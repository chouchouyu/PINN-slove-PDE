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

class AmericanOptionSolver:
    """100维美式期权求解器"""
    
    def __init__(self, d=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.d = d
        self.device = device
        
        # 美式期权参数
        self.r = 0.05  # 无风险利率
        self.sigma = 0.2  # 波动率（美式期权通常较低）
        self.K = 1.0  # 行权价
        
        # 初始条件和时间设置
        self.S0 = torch.tensor([1.0 for _ in range(d)], dtype=torch.float32, device=device)  # 初始股价
        self.tspan = (0.0, 1.0)  # 时间区间
        self.dt = 0.01  # 更小的时间步长（美式期权需要更精细的时间离散）
        self.time_steps = int((self.tspan[1] - self.tspan[0]) / self.dt)
        self.m = 50  # 增加训练轨迹数
        
        # 美式期权特定参数
        self.exercise_dates = torch.linspace(0.1, 1.0, 10, device=device)  # 行权日期
        
        # 网络结构增强
        self.hls = 128  # 增大隐藏层大小
        self.u0 = U0Network(d, self.hls).to(device)
        self.sigma_grad_u = nn.ModuleList([
            SigmaTGradUNetwork(d, self.hls).to(device) for _ in range(self.time_steps)
        ])
        
        # 优化器调整
        self.optimizer = optim.Adam(
            list(self.u0.parameters()) + 
            [param for net in self.sigma_grad_u for param in net.parameters()],
            lr=0.0005  # 降低学习率
        )
        
        self.losses = []
        self.u0_history = []

    def payoff_function(self, S):
        """美式期权收益函数 - 看涨期权"""
        # 多种收益函数可选
        # 1. 算术平均看涨
        arithmetic_mean = torch.mean(S, dim=1, keepdim=True)
        payoff_arithmetic = torch.relu(arithmetic_mean - self.K)
        
        # 2. 几何平均看涨  
        geometric_mean = torch.exp(torch.mean(torch.log(S + 1e-8), dim=1, keepdim=True))
        payoff_geometric = torch.relu(geometric_mean - self.K)
        
        # 3. 最大值的看涨
        max_price, _ = torch.max(S, dim=1, keepdim=True)
        payoff_max = torch.relu(max_price - self.K)
        
        # 使用算术平均（最常用）
        return payoff_arithmetic

    def g(self, X):
        """终端条件：美式期权到期收益"""
        return self.payoff_function(X)

    def f(self, X, u, sigma_grad_u, t):
        """美式期权PDE的非线性项：f = r*u - r*sum(X*σᵀ∇u)"""
        return self.r * (u - torch.sum(X * sigma_grad_u, dim=1, keepdim=True))

    def mu_f(self, X, t):
        """股价漂移项：dX = r*X*dt + σ*X*dW"""
        return self.r * X

    def sigma_f(self, X, t):
        """股价扩散项"""
        if len(X.shape) == 1:
            return torch.diag(self.sigma * X)
        else:
            batch_size = X.shape[0]
            return torch.diag_embed(self.sigma * X)

    def american_exercise_condition(self, X, u, t):
        """美式期权行权条件判断"""
        immediate_payoff = self.payoff_function(X)
        continuation_value = u
        
        # 如果立即行权收益大于继续持有价值，则行权
        exercise_mask = (immediate_payoff > continuation_value) & (
            torch.any(torch.isclose(t, self.exercise_dates), dim=0) if t.dim() > 0 
            else torch.any(torch.isclose(torch.tensor(t), self.exercise_dates))
        )
        
        return torch.where(exercise_mask, immediate_payoff, continuation_value), exercise_mask

    def generate_trajectories(self, batch_size=None, training=True):
        """生成轨迹，包含美式期权行权判断"""
        if batch_size is None:
            batch_size = self.m
            
        X = self.S0.repeat(batch_size, 1)
        u = self.u0(X)
        
        ts = torch.arange(self.tspan[0], self.tspan[1] + self.dt/2, self.dt, device=self.device)
        exercise_flags = torch.zeros(batch_size, 1, device=self.device, dtype=torch.bool)
        
        for i in range(len(ts) - 1):
            t = ts[i]
            
            # 检查是否到达行权日期
            if torch.any(torch.isclose(t, self.exercise_dates)):
                u, exercised = self.american_exercise_condition(X, u, t)
                exercise_flags = exercise_flags | exercised
            
            # 只有未行权的路径继续演化
            active_mask = ~exercise_flags.squeeze()
            if not training or torch.any(active_mask):
                sigma_grad_u_val = self.sigma_grad_u[i](X)
                dW = torch.randn(batch_size, self.d, device=self.device) * np.sqrt(self.dt)
                
                # 更新u（只更新活跃路径）
                if training:
                    f_val = self.f(X, u, sigma_grad_u_val, t)
                    u_update = -f_val * self.dt + torch.sum(sigma_grad_u_val * dW, dim=1, keepdim=True)
                    u = torch.where(active_mask.unsqueeze(1), u + u_update, u)
                
                # 更新X
                mu_val = self.mu_f(X, t)
                sigma_val = self.sigma_f(X, t)
                
                if len(sigma_val.shape) == 2:
                    X_update = torch.matmul(dW, sigma_val)
                else:
                    dW_expanded = dW.unsqueeze(-1)
                    X_update = torch.matmul(sigma_val, dW_expanded).squeeze(-1)
                
                X = torch.where(active_mask.unsqueeze(1), X + mu_val * self.dt + X_update, X)
        
        return X, u, exercise_flags

    def loss_function(self):
        """美式期权损失函数：考虑提前行权"""
        X_final, u_final, exercise_flags = self.generate_trajectories()
        
        # 到期收益或提前行权收益
        terminal_payoff = self.g(X_final)
        final_values = torch.where(exercise_flags, u_final, terminal_payoff)
        
        # 比较网络预测值与实际收益
        predicted_values = self.u0(self.S0.unsqueeze(0)).repeat(X_final.shape[0], 1)
        loss = torch.mean((final_values - predicted_values) ** 2)
        
        return loss

    def train(self, maxiters=200, abstol=1e-6, verbose=True):
        """训练过程"""
        for epoch in range(maxiters):
            self.optimizer.zero_grad()
            loss = self.loss_function()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(
                list(self.u0.parameters()) + 
                [param for net in self.sigma_grad_u for param in net.parameters()],
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            self.losses.append(loss.item())
            current_u0 = self.u0(self.S0.unsqueeze(0))[0, 0].item()
            self.u0_history.append(current_u0)
            
            if verbose and (epoch % 20 == 0 or epoch == maxiters - 1):
                print(f'Epoch {epoch}, Loss: {loss.item():.8f}, Option Price: {current_u0:.6f}')
            
            if loss.item() < abstol:
                if verbose:
                    print(f'Converged at epoch {epoch}')
                break

    def monte_carlo_benchmark(self, num_paths=100000, num_time_steps=100):
        """蒙特卡洛基准（美式期权）"""
        print("Running Monte Carlo benchmark for American option...")
        
        dt = (self.tspan[1] - self.tspan[0]) / num_time_steps
        times = torch.linspace(self.tspan[0], self.tspan[1], num_time_steps + 1, device=self.device)
        
        # 生成路径
        S_paths = torch.zeros(num_paths, num_time_steps + 1, self.d, device=self.device)
        S_paths[:, 0, :] = self.S0
        
        for i in range(num_time_steps):
            dW = torch.randn(num_paths, self.d, device=self.device) * np.sqrt(dt)
            drift = self.r * S_paths[:, i, :] * dt
            diffusion = self.sigma * S_paths[:, i, :] * dW
            S_paths[:, i + 1, :] = S_paths[:, i, :] + drift + diffusion
        
        # 最小二乘蒙特卡洛（LSM）方法
        cash_flows = self.payoff_function(S_paths[:, -1, :])
        exercise_times = torch.full((num_paths,), num_time_steps, device=self.device)
        
        # 向后递归
        for i in range(num_time_steps - 1, 0, -1):
            # 在行权日判断
            if torch.any(torch.isclose(times[i], self.exercise_dates)):
                immediate_payoff = self.payoff_function(S_paths[:, i, :])
                continuation_value = cash_flows * torch.exp(-self.r * (times[-1] - times[i]))
                
                # 使用回归估计继续持有价值
                exercise_mask = immediate_payoff > continuation_value
                cash_flows = torch.where(exercise_mask.squeeze(), immediate_payoff, cash_flows)
                exercise_times = torch.where(exercise_mask.squeeze(), i, exercise_times)
        
        # 贴现到期权价值
        option_price = torch.mean(cash_flows * torch.exp(-self.r * times[exercise_times.long()]))
        
        return option_price.item()

    def solve(self, verbose=True, mc_benchmark=True):
        """主求解函数"""
        print(f"=== {self.d}维美式期权定价 ===")
        
        # 训练网络
        self.train(verbose=verbose)
        
        # 获取期权价格估计
        option_price = self.u0(self.S0.unsqueeze(0))[0, 0].item()
        
        print(f"\n深度学习估计的美式期权价格: {option_price:.6f}")
        
        # 蒙特卡洛基准
        if mc_benchmark:
            mc_price = self.monte_carlo_benchmark()
            print(f"蒙特卡洛基准价格: {mc_price:.6f}")
            print(f"价格差异: {abs(option_price - mc_price):.6f}")
        
        # 绘制结果
        self.plot_results(option_price, mc_price if mc_benchmark else None)
        
        return option_price

    def plot_results(self, dl_price, mc_price=None):
        """绘制结果"""
        plt.figure(figsize=(15, 5))
        
        # 训练损失
        plt.subplot(1, 3, 1)
        plt.semilogy(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        
        # 期权价格收敛
        plt.subplot(1, 3, 2)
        plt.plot(self.u0_history)
        plt.xlabel('Epoch')
        plt.ylabel('Option Price')
        plt.title('Option Price Convergence')
        if mc_price is not None:
            plt.axhline(y=mc_price, color='r', linestyle='--', label=f'MC Benchmark: {mc_price:.4f}')
            plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 价格比较
        plt.subplot(1, 3, 3)
        methods = ['Deep Learning']
        prices = [dl_price]
        
        if mc_price is not None:
            methods.append('Monte Carlo')
            prices.append(mc_price)
        
        plt.bar(methods, prices, alpha=0.7)
        plt.ylabel('Option Price')
        plt.title('Price Comparison')
        
        for i, v in enumerate(prices):
            plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

# 使用示例
def main():
    """主测试函数"""
    # 100维美式期权
    solver = AmericanOptionSolver(d=100)
    option_price = solver.solve(verbose=True, mc_benchmark=True)
    
    print(f"\n最终结果:")
    print(f"100维美式期权价格: {option_price:.6f}")
    
    return solver, option_price

if __name__ == "__main__":
    solver, price = main()
