import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import time
import os

# 设置随机种子
np.random.seed(100)
torch.manual_seed(100)

def rel_error_l2(u, uanal):
    """相对L2误差计算"""
    if abs(uanal) >= 10 * np.finfo(float).eps:
        return np.sqrt((u - uanal)**2 / uanal**2)
    else:
        return abs(u - uanal)

class U0Network(nn.Module):
    """u0网络：近似初始解值"""
    def __init__(self, d, hls):
        super(U0Network, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(d, hls),
            nn.ReLU(),
            nn.Linear(hls, hls),
            nn.ReLU(),
            nn.Linear(hls, 1)
        )

    def forward(self, x):
        return self.network(x)

class SigmaTGradUNetwork(nn.Module):
    """σᵀ∇u网络：每个时间步一个独立网络"""
    def __init__(self, d, hls):
        super(SigmaTGradUNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(d, hls),
            nn.ReLU(),
            nn.Linear(hls, hls),
            nn.ReLU(),
            nn.Linear(hls, hls),
            nn.ReLU(),
            nn.Linear(hls, d)
        )

    def forward(self, x):
        return self.network(x)

class AmericanOptionDeepBSDESolver:
    """100维度美式期权DeepBSDE求解器"""
    
    def __init__(self, d=100, option_type='put', strike_price=50.0, penalty_param=1000.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.d = d
        self.option_type = option_type  # 'call' 或 'put'
        self.K = strike_price  # 执行价格
        self.lambda_penalty = penalty_param  # 惩罚参数
        self.device = device
        
        # 方程参数
        self.r = 0.05
        self.sigma = 0.4
        
        # 初始条件和时间设置
        self.x0 = torch.tensor([1.0 if i % 2 == 0 else 0.5 for i in range(d)], 
                              dtype=torch.float32, device=device)
        self.tspan = (0.0, 1.0)
        self.dt = 0.25
        self.time_steps = int((self.tspan[1] - self.tspan[0]) / self.dt)
        self.m = 30  # 训练轨迹数
        
        # Legendre变换参数（用于下界计算）
        self.A = torch.linspace(-2.0, 2.0, 401, device=device)
        self.u_domain = torch.linspace(-500.0, 500.0, 10001, device=device)
        
        # 网络初始化 - 增加隐藏层大小以适应100维问题
        self.hls = 20 + d  # 隐藏层大小，增加以适应高维美式期权
        self.u0 = U0Network(d, self.hls).to(device)
        self.sigma_grad_u = nn.ModuleList([
            SigmaTGradUNetwork(d, self.hls).to(device) for _ in range(self.time_steps)
        ])
        
        # 优化器 - 使用更小的学习率以提高稳定性
        self.optimizer = optim.Adam(
            list(self.u0.parameters()) + 
            [param for net in self.sigma_grad_u for param in net.parameters()],
            lr=0.0005  # 降低学习率以适应高维问题
        )
        
        # 训练历史记录
        self.losses = []
        self.u0_history = []

    def immediate_exercise_payoff(self, X):
        """立即行权收益函数"""
        portfolio_value = torch.sum(X**2, dim=1, keepdim=True)
        if self.option_type == 'call':
            return torch.maximum(portfolio_value - self.K, torch.zeros_like(portfolio_value))
        else:  # put option
            return torch.maximum(self.K - portfolio_value, torch.zeros_like(portfolio_value))

    def g(self, X):
        """终端条件：与立即行权收益相同"""
        return self.immediate_exercise_payoff(X)

    def f(self, X, u, sigma_grad_u, t):
        """非线性项：包含美式期权的惩罚项"""
        # 标准Black-Scholes-Barenblatt非线性项
        bs_term = self.r * (u - torch.sum(X * sigma_grad_u, dim=1, keepdim=True))
        
        # 美式期权惩罚项：λ * max(u - h(X), 0)
        # 注意：这里使用负号，因为我们要最小化损失，而惩罚项应该是+λ * max(u - h(X), 0)
        # 在变分不等式中：∂u/∂t + Lu + λ max(u - h(X), 0) = 0
        h_X = self.immediate_exercise_payoff(X)
        penalty_term = self.lambda_penalty * torch.maximum(u - h_X, torch.zeros_like(u))
        
        return bs_term + penalty_term

    def mu_f(self, X, t):
        """漂移项：μ(X, p, t) = 0"""
        return torch.zeros_like(X)

    def sigma_f(self, X, t):
        """扩散项：σ(X, p, t) = Diagonal(sigma * X)"""
        if len(X.shape) == 1:
            # 一维情况：返回2D对角矩阵
            return torch.diag(self.sigma * X)
        else:
            # 二维情况：返回3D对角矩阵批次
            batch_size = X.shape[0]
            return torch.diag_embed(self.sigma * X)

    def generate_trajectories(self, batch_size=None):
        """生成轨迹"""
        if batch_size is None:
            batch_size = self.m
            
        X = self.x0.repeat(batch_size, 1)
        u = self.u0(X)
        
        # 检查初始点是否应该立即行权
        h_X = self.immediate_exercise_payoff(X)
        u = torch.maximum(u, h_X)  # 确保初始价值不低于立即行权收益
        
        ts = torch.arange(self.tspan[0], self.tspan[1] + self.dt/2, self.dt, device=self.device)
        
        for i in range(len(ts) - 1):
            t = ts[i].item()
            
            sigma_grad_u_val = self.sigma_grad_u[i](X)
            dW = torch.randn(batch_size, self.d, device=self.device) * np.sqrt(self.dt)
            
            # 更新u
            f_val = self.f(X, u, sigma_grad_u_val, t)
            u = u - f_val * self.dt + torch.sum(sigma_grad_u_val * dW, dim=1, keepdim=True)
            
            # 确保u不低于立即行权收益（美式期权约束）
            h_X = self.immediate_exercise_payoff(X)
            u = torch.maximum(u, h_X)
            
            # 更新X
            mu_val = self.mu_f(X, t)
            sigma_val = self.sigma_f(X, t)
            
            if len(sigma_val.shape) == 2:  # 单样本情况
                X = X + mu_val * self.dt + torch.matmul(dW, sigma_val)
            else:  # 批次情况
                dW_expanded = dW.unsqueeze(-1)  # [batch_size, d, 1]
                sigma_val_expanded = sigma_val  # [batch_size, d, d]
                X_update = torch.matmul(sigma_val_expanded, dW_expanded).squeeze(-1)  # [batch_size, d]
                X = X + mu_val * self.dt + X_update
        
        # 终端时刻确保u不低于行权收益
        h_X_final = self.immediate_exercise_payoff(X)
        u = torch.maximum(u, h_X_final)
        
        return X, u

    def loss_function(self):
        """损失函数"""
        X_final, u_final = self.generate_trajectories()
        g_X = self.g(X_final)
        loss = torch.mean((g_X - u_final) ** 2)
        return loss

    def train(self, maxiters=200, abstol=1e-8, verbose=True):
        """训练过程"""
        for epoch in range(maxiters):
            self.optimizer.zero_grad()
            loss = self.loss_function()
            loss.backward()
            self.optimizer.step()
            
            self.losses.append(loss.item())
            current_u0 = self.u0(self.x0.unsqueeze(0))[0, 0].item()
            
            # 确保u0不低于立即行权收益
            h_x0 = self.immediate_exercise_payoff(self.x0.unsqueeze(0))[0, 0].item()
            current_u0 = max(current_u0, h_x0)
            
            self.u0_history.append(current_u0)
            
            if verbose and (epoch % 10 == 0 or epoch == maxiters - 1):
                print(f'Epoch {epoch}, Loss: {loss.item():.6f}, u0: {current_u0:.6f}')
            
            if loss.item() < abstol:
                if verbose:
                    print(f'Converged at epoch {epoch}')
                break

    def compute_upper_bound(self, trajectories=1000, maxiters_limits=10, verbose=True):
        """计算上界 - 注意：美式期权的上界计算需要特殊处理"""
        if verbose:
            print("美式期权上界计算...")
        
        # 创建独立的网络副本
        u0_high = U0Network(self.d, self.hls).to(self.device)
        u0_high.load_state_dict(self.u0.state_dict())
        
        sigma_grad_u_high = nn.ModuleList([
            SigmaTGradUNetwork(self.d, self.hls).to(self.device) for _ in range(self.time_steps)
        ])
        for i, net in enumerate(sigma_grad_u_high):
            net.load_state_dict(self.sigma_grad_u[i].state_dict())
        
        # 优化器
        high_opt = optim.Adam(
            list(u0_high.parameters()) + 
            [param for net in sigma_grad_u_high for param in net.parameters()],
            lr=0.01
        )
        
        ts = torch.arange(self.tspan[0], self.tspan[1] + self.dt/2, self.dt, device=self.device)
        
        def upper_bound_loss():
            """上界损失函数 - 包含美式期权约束"""
            total = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            for _ in range(trajectories):
                # 正向SDE模拟
                X = self.x0.clone().unsqueeze(0)  # [1, d]
                X_trajectory = [X.clone()]
                
                # 正向过程 - 不需要梯度
                with torch.no_grad():
                    for i in range(len(ts) - 1):
                        t = ts[i].item()
                        dW = torch.randn(1, self.d, device=self.device) * np.sqrt(self.dt)
                        mu_val = self.mu_f(X, t)
                        sigma_val = self.sigma_f(X, t)
                        
                        if len(sigma_val.shape) == 2:  # 2D矩阵
                            X = X + mu_val * self.dt + torch.matmul(dW, sigma_val)
                        else:  # 3D张量
                            dW_expanded = dW.unsqueeze(-1)  # [1, d, 1]
                            X_update = torch.matmul(sigma_val, dW_expanded).squeeze(-1)  # [1, d]
                            X = X + mu_val * self.dt + X_update
                        
                        X_trajectory.append(X.clone())
                
                # 反向计算上界 - 需要梯度
                U = self.g(X)
                
                for i in range(len(ts)-2, -1, -1):
                    t = ts[i].item()
                    X_prev = X_trajectory[i]
                    sigma_grad_u_val = sigma_grad_u_high[i](X_prev)
                    dW = torch.randn(1, self.d, device=self.device) * np.sqrt(self.dt)
                    
                    # 包含美式期权约束的f值
                    h_X_prev = self.immediate_exercise_payoff(X_prev)
                    u_prev = U + self.f(X_prev, U, sigma_grad_u_val, t) * self.dt - torch.sum(sigma_grad_u_val * dW, dim=1, keepdim=True)
                    
                    # 确保不低于立即行权收益
                    U = torch.maximum(u_prev, h_X_prev)
                
                total = total + U
            
            return total / trajectories

        # 优化上界
        for i in range(maxiters_limits):
            high_opt.zero_grad()
            
            # 计算损失 - 我们希望最大化上界，所以最小化负值
            upper_bound = upper_bound_loss()
            loss = -upper_bound  # 负号因为我们想最大化上界
            loss.backward()
            high_opt.step()
            
            if verbose and (i % 2 == 0 or i == maxiters_limits - 1):
                with torch.no_grad():
                    current_bound = -upper_bound_loss().item()
                print(f'Upper bound optimization {i}: {current_bound:.6f}')
        
        # 计算最终上界估计
        with torch.no_grad():
            final_upper_bound = upper_bound_loss()
            u_high = final_upper_bound.item()
        
        if verbose:
            print(f"Upper bound: {u_high:.6f}")
        
        return u_high

    def compute_lower_bound(self, trajectories=1000, verbose=True):
        """计算下界 - 使用Legendre变换"""
        if verbose:
            print("美式期权下界计算...")
        
        ts = torch.arange(self.tspan[0], self.tspan[1] + self.dt/2, self.dt, device=self.device)
        total_lower = torch.tensor(0.0, device=self.device)
        
        for _ in range(trajectories):
            # 初始化u和X
            u = self.u0(self.x0.unsqueeze(0))[0, 0]  # 标量
            X = self.x0.clone()  # 一维张量 [d]
            I = torch.tensor(0.0, device=self.device)
            Q = torch.tensor(0.0, device=self.device)
            
            for i in range(len(ts) - 1):
                t = ts[i].item()
                
                # 获取σᵀ∇u
                sigma_grad_u_val = self.sigma_grad_u[i](X.unsqueeze(0)).squeeze(0)  # [d]
                dW = torch.randn(self.d, device=self.device) * np.sqrt(self.dt)  # [d]
                
                # 更新u
                X_2d = X.unsqueeze(0)  # [1, d]
                u_2d = u.unsqueeze(0).unsqueeze(-1)  # [1, 1]
                sigma_grad_u_val_2d = sigma_grad_u_val.unsqueeze(0)  # [1, d]
                
                # 包含美式期权约束的f值
                f_val = self.f(X_2d, u_2d, sigma_grad_u_val_2d, t)[0, 0]  # 标量
                
                # 点积计算
                dot_product = torch.dot(sigma_grad_u_val, dW)
                u = u - f_val * self.dt + dot_product
                
                # 确保u不低于立即行权收益
                h_X = self.immediate_exercise_payoff(X.unsqueeze(0))[0, 0]
                u = torch.max(u, h_X)
                
                # 更新X
                mu_val = self.mu_f(X, t)  # [d]
                sigma_val = self.sigma_f(X, t)  # [d, d]
                
                # 矩阵向量乘法
                X_update = torch.matmul(sigma_val, dW.unsqueeze(-1)).squeeze(-1)  # [d]
                X = X + mu_val * self.dt + X_update
                
                # Legendre变换
                # 计算点积 sum(X * sigma_grad_u_val)
                X_dot_sigma_grad_u = torch.sum(X * sigma_grad_u_val)
                
                # 计算f矩阵：对于u_domain中的每个u值，计算f(X, u, sigma_grad_u_val, t)
                f_matrix = self.r * (self.u_domain - X_dot_sigma_grad_u)  # 形状: [n_u]
                
                # 计算Legendre变换值：对于A中的每个a，计算 max_u (a*u - f(u))
                a_expanded = self.A.unsqueeze(1)  # 形状: [n_A, 1]
                u_expanded = self.u_domain.unsqueeze(0)  # 形状: [1, n_u]
                f_expanded = f_matrix.unsqueeze(0)  # 形状: [1, n_u]
                
                # 计算 a*u - f(u) 对于所有a和u的组合
                le_matrix = a_expanded * u_expanded - f_expanded  # 形状: [n_A, n_u]
                
                # 对每个a取最大值（在u维度上）
                legendre_values, _ = torch.max(le_matrix, dim=1)  # 形状: [n_A]
                
                # 寻找最优控制a：最大化 a*u - F(a)
                a_u_minus_F = self.A * u - legendre_values
                optimal_idx = torch.argmax(a_u_minus_F)
                a_optimal = self.A[optimal_idx]
                
                # 获取最优a对应的Legendre变换值
                F_optimal = legendre_values[optimal_idx]
                
                # 累积对偶过程
                I = I + a_optimal * self.dt
                Q = Q + torch.exp(I) * F_optimal
            
            # 终端时刻确保u不低于行权收益
            g_X = self.g(X.unsqueeze(0))[0, 0]  # 添加批次维度计算g(X)
            u_T = max(g_X.item(), self.immediate_exercise_payoff(X.unsqueeze(0))[0, 0].item())
            
            total_lower = total_lower + torch.exp(I) * u_T - Q
        
        u_low = (total_lower / trajectories).item()
        
        if verbose:
            print(f"Lower bound: {u_low:.6f}")
        
        return u_low

    def solve(self, limits=False, trajectories_upper=1000, trajectories_lower=1000, 
              maxiters_limits=10, verbose=True, save_everystep=False):
        """主求解函数"""
        
        # 训练主网络
        self.train(verbose=verbose)
        
        # 获取点估计
        u0_estimate = self.u0(self.x0.unsqueeze(0))[0, 0].item()
        
        # 确保点估计不低于立即行权收益
        h_x0 = self.immediate_exercise_payoff(self.x0.unsqueeze(0))[0, 0].item()
        u0_estimate = max(u0_estimate, h_x0)
        
        if not limits:
            # 不计算上下界
            if verbose:
                print(f"点估计: {u0_estimate:.6f}")
                print(f"立即行权收益: {h_x0:.6f}")
            
            # 返回结果
            class PIDESolution:
                def __init__(self, X0, ts, losses, u0_estimate, u0_network, limits=None):
                    self.X0 = X0
                    self.ts = ts
                    self.losses = losses
                    self.us = u0_estimate
                    self.u0 = u0_network
                    self.limits = limits
            
            ts_array = torch.arange(self.tspan[0], self.tspan[1] + self.dt/2, self.dt).cpu().numpy()
            
            if save_everystep:
                return PIDESolution(self.x0.cpu().numpy(), ts_array, self.losses, self.u0_history, self.u0)
            else:
                return PIDESolution(self.x0.cpu().numpy(), ts_array, self.losses, u0_estimate, self.u0)
        
        else:
            # 计算上下界
            u_high = self.compute_upper_bound(
                trajectories=trajectories_upper, 
                maxiters_limits=maxiters_limits, 
                verbose=verbose
            )
            
            u_low = self.compute_lower_bound(
                trajectories=trajectories_lower,
                verbose=verbose
            )
            
            if verbose:
                print(f"\n解的上下界:")
                print(f"Lower bound: {u_low:.6f}")
                print(f"Point estimate: {u0_estimate:.6f}") 
                print(f"Upper bound: {u_high:.6f}")
                print(f"立即行权收益: {h_x0:.6f}")
                print(f"Within bounds: {u_low <= u0_estimate <= u_high}")
                print(f"期权价值 >= 立即行权收益: {u0_estimate >= h_x0}")
        
        # 返回结果
        class PIDESolution:
            def __init__(self, X0, ts, losses, u0_estimate, u0_network, limits=None):
                self.X0 = X0
                self.ts = ts
                self.losses = losses
                self.us = u0_estimate
                self.u0 = u0_network
                self.limits = limits
        
        ts_array = torch.arange(self.tspan[0], self.tspan[1] + self.dt/2, self.dt).cpu().numpy()
        
        if save_everystep:
            return PIDESolution(self.x0.cpu().numpy(), ts_array, self.losses, self.u0_history, self.u0, (u_low, u_high))
        else:
            return PIDESolution(self.x0.cpu().numpy(), ts_array, self.losses, u0_estimate, self.u0, (u_low, u_high))


# 测试代码
if __name__ == "__main__":
    # 创建求解器实例
    solver = AmericanOptionDeepBSDESolver(
        d=100, 
        option_type='put', 
        strike_price=50.0, 
        penalty_param=1000.0
    )
    
    print("=== 100维度美式期权DeepBSDE求解 ===")
    print(f"期权类型: {solver.option_type}")
    print(f"执行价格: {solver.K}")
    print(f"维度: {solver.d}")
    
    # 求解期权
    result = solver.solve(limits=True, verbose=True)
    
    # 绘制结果
    import matplotlib.pyplot as plt
    
    # 创建Figures目录
    figures_dir = "Figures"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    # 绘制训练损失
    plt.figure(figsize=(10, 6))
    plt.semilogy(solver.losses, 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss for 100D American Option')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{figures_dir}/AmericanOption_100D_Loss.png")
    plt.close()
    
    # 绘制期权价值估计历史
    plt.figure(figsize=(10, 6))
    plt.plot(solver.u0_history, 'r-')
    plt.xlabel('Epoch')
    plt.ylabel('Option Value Estimate')
    plt.title('Option Value Estimate During Training')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{figures_dir}/AmericanOption_100D_ValueHistory.png")
    plt.close()
    
    print("\n求解完成！")
    print(f"最终期权价值估计: {result.us:.6f}")
    if hasattr(result, 'limits') and result.limits is not None:
        print(f"上下界: [{result.limits[0]:.6f}, {result.limits[1]:.6f}]")
    print("图形已保存到Figures目录。")
