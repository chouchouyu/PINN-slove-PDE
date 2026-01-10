import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from .Models import U0Network, SigmaTGradUNetwork
from .BlackScholesBarenblatt import BlackScholesBarenblatt

# 设置随机种子
np.random.seed(100)
torch.manual_seed(100)

def rel_error_l2(u, uanal):
    """相对L2误差计算"""
    if abs(uanal) >= 10 * np.finfo(float).eps:
        return np.sqrt((u - uanal)**2 / uanal**2)
    else:
        return abs(u - uanal)

class BlackScholesBarenblattSolver(BlackScholesBarenblatt):
    """Black-Scholes-Barenblatt方程求解器"""
    
    def __init__(self, d=30, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # 调用父类构造函数
        super().__init__(d)
        
        self.device = device
        
        # 初始条件和时间设置
        self.x0 = torch.tensor([1.0 if i % 2 == 0 else 0.5 for i in range(d)], 
                              dtype=torch.float32, device=device)
        self.tspan = (0.0, 1.0)
        self.dt = 0.25
        self.time_steps = int((self.tspan[1] - self.tspan[0]) / self.dt)
        self.m = 30  # 训练轨迹数
        
        # Legendre变换参数
        self.A = torch.linspace(-2.0, 2.0, 401, device=device)  # 控制变量范围
        self.u_domain = torch.linspace(-10.0, 10.0, 201, device=device)  # u的取值范围
        
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
            mu_val = self.mu_tf(X, t)
            sigma_val = self.sigma_tf(X, t)
            
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
        g_X = self.g_tf(X_final)
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
            # 计算上下界 - 使用独立函数
            from .BoundsCalculator import compute_upper_bound, compute_lower_bound
            
            u_high = compute_upper_bound(
                self,
                trajectories=trajectories_upper, 
                maxiters_limits=maxiters_limits, 
                verbose=verbose
            )
            
            u_low = compute_lower_bound(
                self,
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


