import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any

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

class BlackScholesBarenblattSolver:
    """Black-Scholes-Barenblatt方程求解器"""
    
    def __init__(self, d=30, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.d = d
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
        
        # Legendre变换参数
        self.A = torch.linspace(-2.0, 2.0, 401, device=device)  # 控制变量范围 - 与Julia原文件一致
        # 与Julia原文件一致的设置（之前: torch.linspace(-10.0, 10.0, 201, device=device)）
        self.u_domain = torch.linspace(-500.0, 500.0, 10001, device=device)  # u的取值范围
        
        # 网络初始化
        self.hls = 10 + d  # 隐藏层大小
        self.u0 = U0Network(d, self.hls).to(device)
        self.sigma_grad_u = nn.ModuleList([
            SigmaTGradUNetwork(d, self.hls).to(device) for _ in range(self.time_steps)
        ])
        
        # 优化器
        self.optimizer = optim.Adam(
            list(self.u0.parameters()) + 
            [param for net in self.sigma_grad_u for param in net.parameters()],
            lr=0.001
        )
        
        # 训练历史记录
        self.losses = []
        self.u0_history = []

    def g(self, X):
        """终端条件：g(X) = sum(X^2)"""
        return torch.sum(X**2, dim=1, keepdim=True)

    def f(self, X, u, sigma_grad_u, t):
        """非线性项：f(X, u, σᵀ∇u, p, t) = r * (u - sum(X * σᵀ∇u))"""
        return self.r * (u - torch.sum(X * sigma_grad_u, dim=1, keepdim=True))

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
        
        ts = torch.arange(self.tspan[0], self.tspan[1] + self.dt/2, self.dt, device=self.device)
        
        for i in range(len(ts) - 1):
            t = ts[i].item()
            
            sigma_grad_u_val = self.sigma_grad_u[i](X)
            dW = torch.randn(batch_size, self.d, device=self.device) * np.sqrt(self.dt)
            
            # 更新u
            f_val = self.f(X, u, sigma_grad_u_val, t)
            u = u - f_val * self.dt + torch.sum(sigma_grad_u_val * dW, dim=1, keepdim=True)
            
            # 更新X - 修复矩阵乘法维度问题
            mu_val = self.mu_f(X, t)
            sigma_val = self.sigma_f(X, t)
            
            if len(sigma_val.shape) == 2:  # 单样本情况
                X = X + mu_val * self.dt + torch.matmul(dW, sigma_val)
            else:  # 批次情况
                dW_expanded = dW.unsqueeze(-1)  # [batch_size, d, 1]
                sigma_val_expanded = sigma_val  # [batch_size, d, d]
                X_update = torch.matmul(sigma_val_expanded, dW_expanded).squeeze(-1)  # [batch_size, d]
                X = X + mu_val * self.dt + X_update
        
        return X, u

    def loss_function(self):
        """损失函数"""
        X_final, u_final = self.generate_trajectories()
        g_X = self.g(X_final)
        loss = torch.mean((g_X - u_final) ** 2)
        return loss

    def train(self, maxiters=150, abstol=1e-8, verbose=True):
        """训练过程"""
        for epoch in range(maxiters):
            self.optimizer.zero_grad()
            loss = self.loss_function()
            loss.backward()
            self.optimizer.step()
            
            self.losses.append(loss.item())
            current_u0 = self.u0(self.x0.unsqueeze(0))[0, 0].item()
            self.u0_history.append(current_u0)
            
            if verbose and (epoch % 10 == 0 or epoch == maxiters - 1):
                print(f'Epoch {epoch}, Loss: {loss.item():.6f}, u0: {current_u0:.6f}')
            
            if loss.item() < abstol:
                if verbose:
                    print(f'Converged at epoch {epoch}')
                break

    def analytical_solution(self, x, t):
        """解析解"""
        T = self.tspan[1]
        exponent = (self.r + self.sigma**2) * (T - t)
        return torch.exp(torch.tensor(exponent, device=x.device)) * torch.sum(x**2)

    def compute_upper_bound(self, trajectories=1000, maxiters_limits=10, verbose=True):
        """计算上界"""
        if verbose:
            print("Calculating upper bound...")
        
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
            """上界损失函数"""
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
                    
                    f_val = self.f(X_prev, U, sigma_grad_u_val, t)
                    U = U + f_val * self.dt - torch.sum(sigma_grad_u_val * dW, dim=1, keepdim=True)
                
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
        """计算下界 - 使用Legendre变换，与Julia代码完全一致"""
        if verbose:
            print("Calculating lower bound with Legendre transform...")
        
        ts = torch.arange(self.tspan[0], self.tspan[1] + self.dt/2, self.dt, device=self.device)
        total_lower = torch.tensor(0.0, device=self.device)
        
        for _ in range(trajectories):
            # 与Julia代码完全一致：初始化u和X
            u = self.u0(self.x0.unsqueeze(0))[0, 0]  # 标量
            X = self.x0.clone()  # 一维张量 [d]
            I = torch.tensor(0.0, device=self.device)
            Q = torch.tensor(0.0, device=self.device)
            
            for i in range(len(ts) - 1):
                t = ts[i].item()
                
                # 获取σᵀ∇u - 与Julia代码一致
                # 网络期望输入是二维的，所以需要添加批次维度
                sigma_grad_u_val = self.sigma_grad_u[i](X.unsqueeze(0)).squeeze(0)  # [d]
                dW = torch.randn(self.d, device=self.device) * np.sqrt(self.dt)  # [d]
                
                # 更新u - 与Julia代码完全一致
                # f期望输入是二维的，所以需要添加批次维度
                X_2d = X.unsqueeze(0)  # [1, d]
                u_2d = u.unsqueeze(0).unsqueeze(-1)  # [1, 1]
                sigma_grad_u_val_2d = sigma_grad_u_val.unsqueeze(0)  # [1, d]
                
                f_val = self.f(X_2d, u_2d, sigma_grad_u_val_2d, t)[0, 0]  # 标量
                
                # 点积计算：sigma_grad_u_val 和 dW 都是一维的
                dot_product = torch.dot(sigma_grad_u_val, dW)
                u = u - f_val * self.dt + dot_product
                
                # 更新X - 修复维度问题
                mu_val = self.mu_f(X, t)  # [d]
                sigma_val = self.sigma_f(X, t)  # [d, d]，因为X是一维的
                
                # 矩阵向量乘法：sigma_val @ dW
                # sigma_val形状是[d, d]，dW形状是[d]，需要将dW变为列向量[d, 1]
                X_update = torch.matmul(sigma_val, dW.unsqueeze(-1)).squeeze(-1)  # [d]
                X = X + mu_val * self.dt + X_update
                
                # Legendre变换 - 与Julia代码完全一致
                # 计算点积 sum(X * sigma_grad_u_val)
                X_dot_sigma_grad_u = torch.sum(X * sigma_grad_u_val)
                
                # 计算f矩阵：对于u_domain中的每个u值，计算f(X, u, sigma_grad_u_val, t)
                # f = r * (u - X_dot_sigma_grad_u)
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
                
                # 累积对偶过程 - 与Julia代码完全一致
                I = I + a_optimal * self.dt
                Q = Q + torch.exp(I) * F_optimal
            
            g_X = self.g(X.unsqueeze(0))[0, 0]  # 添加批次维度计算g(X)
            total_lower = total_lower + torch.exp(I) * g_X - Q
        
        u_low = (total_lower / trajectories).item()
        
        if verbose:
            print(f"Lower bound: {u_low:.6f}")
        
        return u_low

    def solve(self, limits=False, trajectories_upper=1000, trajectories_lower=1000, 
              maxiters_limits=10, verbose=True, save_everystep=False):
        """主求解函数 - 与Julia的solve函数接口完全一致"""
        
        # 训练主网络
        self.train(verbose=verbose)
        
        # 获取点估计
        u0_estimate = self.u0(self.x0.unsqueeze(0))[0, 0].item()
        
        # 计算解析解
        u_analytical = self.analytical_solution(self.x0, self.tspan[0]).item()
        
        if not limits:
            # 不计算上下界
            if verbose:
                print(f"Point estimate: {u0_estimate:.6f}")
                print(f"Analytical solution: {u_analytical:.6f}")
                error = rel_error_l2(u0_estimate, u_analytical)
                print(f"Relative error: {error:.6f}")
            
            # 返回与Julia代码一致的格式
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
            # 计算上下界 - 与Julia代码完全一致的条件分支
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
                print(f"\nSolution bounds:")
                print(f"Lower bound: {u_low:.6f}")
                print(f"Point estimate: {u0_estimate:.6f}") 
                print(f"Upper bound: {u_high:.6f}")
                print(f"Analytical solution: {u_analytical:.6f}")
                print(f"Within bounds: {u_low <= u0_estimate <= u_high}")
            
            error = rel_error_l2(u0_estimate, u_analytical)
            
            if verbose:
                print(f"Relative error: {error:.6f}")
            
            # 返回与Julia代码一致的格式
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

def main():
    """主测试函数"""
    print("=== 30维Black-Scholes-Barenblatt方程求解 ===")
    
    # 测试标准版本（limits=false）
    print("\n1. 标准DeepBSDE算法:")
    solver_std = BlackScholesBarenblattSolver(d=30)
    result_std = solver_std.solve(limits=False, verbose=True)
    
    # 验证标准版本结果
    u_pred_std = result_std.us if hasattr(result_std.us, '__len__') else result_std.us
    u_anal_std = solver_std.analytical_solution(solver_std.x0, solver_std.tspan[0]).item()
    if hasattr(u_pred_std, '__len__'):
        error_std = rel_error_l2(u_pred_std[-1], u_anal_std)
    else:
        error_std = rel_error_l2(u_pred_std, u_anal_std)
    
    print(f"标准算法误差: {error_std:.6f}")
    
    # 测试带Legendre变换的版本（limits=true）
    print("\n2. 带Legendre变换对偶方法的DeepBSDE:")
    solver_limits = BlackScholesBarenblattSolver(d=30)
    result_limits = solver_limits.solve(
        limits=True, 
        trajectories_upper=1000,  # 与Julia原文件一致 (之前: 100)
        trajectories_lower=1000,  # 与Julia原文件一致 (之前: 100)
        maxiters_limits=10,       # 与Julia原文件一致 (之前: 5)
        verbose=True
    )
    
    # 验证带界限版本结果
    u_pred_limits = result_limits.us if hasattr(result_limits.us, '__len__') else result_limits.us
    u_anal_limits = solver_limits.analytical_solution(solver_limits.x0, solver_limits.tspan[0]).item()
    if hasattr(u_pred_limits, '__len__'):
        error_limits = rel_error_l2(u_pred_limits[-1], u_anal_limits)
    else:
        error_limits = rel_error_l2(u_pred_limits, u_anal_limits)
    
    print(f"对偶方法误差: {error_limits:.6f}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.semilogy(solver_std.losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (Standard)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.semilogy(solver_limits.losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (With Limits)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    if hasattr(result_limits, 'limits') and result_limits.limits is not None:
        u_low, u_high = result_limits.limits
        if hasattr(result_limits.us, '__len__'):
            u_point = result_limits.us[-1]
        else:
            u_point = result_limits.us
        u_anal = u_anal_limits
        
        plt.axhline(y=u_anal, color='green', linestyle='--', label='Analytical', alpha=0.7)
        plt.axhline(y=u_point, color='blue', linestyle='-', label='Point Estimate', alpha=0.7)
        plt.axhspan(u_low, u_high, alpha=0.3, color='red', label='Confidence Interval')
        plt.ylabel('Solution Value')
        plt.title('Solution with Bounds')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 验证结果
    print(f"\n验证结果:")
    print(f"标准算法误差: {error_std:.6f} {'✓ < 1.0' if error_std < 1.0 else '✗ >= 1.0'}")
    print(f"对偶方法误差: {error_limits:.6f} {'✓ < 1.0' if error_limits < 1.0 else '✗ >= 1.0'}")
    
    if hasattr(result_limits, 'limits') and result_limits.limits is not None:
        u_low, u_high = result_limits.limits
        if hasattr(result_limits.us, '__len__'):
            u_point = result_limits.us[-1]
        else:
            u_point = result_limits.us
        if u_low <= u_point <= u_high:
            print("✓ 点估计在上下界范围内")
        else:
            print("✗ 点估计超出上下界范围")
    
    return solver_std, solver_limits, result_std, result_limits

if __name__ == "__main__":
    solver_std, solver_limits, result_std, result_limits = main()
