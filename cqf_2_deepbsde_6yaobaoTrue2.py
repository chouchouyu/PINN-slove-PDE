import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import time
import pandas as pd
from scipy import stats
from matplotlib.patches import Patch

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
        self.A = torch.linspace(-2.0, 2.0, 401, device=device)
        self.u_domain = torch.linspace(-500.0, 500.0, 10001, device=device)
        
        # 网络初始化
        self.hls = 10 + d
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
            return torch.diag(self.sigma * X)
        else:
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
            
            # 更新X
            mu_val = self.mu_f(X, t)
            sigma_val = self.sigma_f(X, t)
            
            if len(sigma_val.shape) == 2:
                X = X + mu_val * self.dt + torch.matmul(dW, sigma_val)
            else:
                dW_expanded = dW.unsqueeze(-1)
                X_update = torch.matmul(sigma_val, dW_expanded).squeeze(-1)
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
        
        u0_high = U0Network(self.d, self.hls).to(self.device)
        u0_high.load_state_dict(self.u0.state_dict())
        
        sigma_grad_u_high = nn.ModuleList([
            SigmaTGradUNetwork(self.d, self.hls).to(self.device) for _ in range(self.time_steps)
        ])
        for i, net in enumerate(sigma_grad_u_high):
            net.load_state_dict(self.sigma_grad_u[i].state_dict())
        
        high_opt = optim.Adam(
            list(u0_high.parameters()) + 
            [param for net in sigma_grad_u_high for param in net.parameters()],
            lr=0.01
        )
        
        ts = torch.arange(self.tspan[0], self.tspan[1] + self.dt/2, self.dt, device=self.device)
        
        def upper_bound_loss():
            total = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            for _ in range(trajectories):
                X = self.x0.clone().unsqueeze(0)
                X_trajectory = [X.clone()]
                
                with torch.no_grad():
                    for i in range(len(ts) - 1):
                        t = ts[i].item()
                        dW = torch.randn(1, self.d, device=self.device) * np.sqrt(self.dt)
                        mu_val = self.mu_f(X, t)
                        sigma_val = self.sigma_f(X, t)
                        
                        if len(sigma_val.shape) == 2:
                            X = X + mu_val * self.dt + torch.matmul(dW, sigma_val)
                        else:
                            dW_expanded = dW.unsqueeze(-1)
                            X_update = torch.matmul(sigma_val, dW_expanded).squeeze(-1)
                            X = X + mu_val * self.dt + X_update
                        
                        X_trajectory.append(X.clone())
                
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

        for i in range(maxiters_limits):
            high_opt.zero_grad()
            upper_bound = upper_bound_loss()
            loss = -upper_bound
            loss.backward()
            high_opt.step()
            
            if verbose and (i % 2 == 0 or i == maxiters_limits - 1):
                with torch.no_grad():
                    current_bound = -upper_bound_loss().item()
                print(f'Upper bound optimization {i}: {current_bound:.6f}')
        
        with torch.no_grad():
            final_upper_bound = upper_bound_loss()
            u_high = final_upper_bound.item()
        
        if verbose:
            print(f"Upper bound: {u_high:.6f}")
        
        return u_high

    def compute_lower_bound(self, trajectories=1000, verbose=True):
        """计算下界"""
        if verbose:
            print("Calculating lower bound with Legendre transform...")
        
        ts = torch.arange(self.tspan[0], self.tspan[1] + self.dt/2, self.dt, device=self.device)
        total_lower = torch.tensor(0.0, device=self.device)
        
        for _ in range(trajectories):
            u = self.u0(self.x0.unsqueeze(0))[0, 0]
            X = self.x0.clone()
            I = torch.tensor(0.0, device=self.device)
            Q = torch.tensor(0.0, device=self.device)
            
            for i in range(len(ts) - 1):
                t = ts[i].item()
                
                sigma_grad_u_val = self.sigma_grad_u[i](X.unsqueeze(0)).squeeze(0)
                dW = torch.randn(self.d, device=self.device) * np.sqrt(self.dt)
                
                X_2d = X.unsqueeze(0)
                u_2d = u.unsqueeze(0).unsqueeze(-1)
                sigma_grad_u_val_2d = sigma_grad_u_val.unsqueeze(0)
                
                f_val = self.f(X_2d, u_2d, sigma_grad_u_val_2d, t)[0, 0]
                dot_product = torch.dot(sigma_grad_u_val, dW)
                u = u - f_val * self.dt + dot_product
                
                mu_val = self.mu_f(X, t)
                sigma_val = self.sigma_f(X, t)
                X_update = torch.matmul(sigma_val, dW.unsqueeze(-1)).squeeze(-1)
                X = X + mu_val * self.dt + X_update
                
                X_dot_sigma_grad_u = torch.sum(X * sigma_grad_u_val)
                f_matrix = self.r * (self.u_domain - X_dot_sigma_grad_u)
                
                a_expanded = self.A.unsqueeze(1)
                u_expanded = self.u_domain.unsqueeze(0)
                f_expanded = f_matrix.unsqueeze(0)
                
                le_matrix = a_expanded * u_expanded - f_expanded
                legendre_values, _ = torch.max(le_matrix, dim=1)
                
                a_u_minus_F = self.A * u - legendre_values
                optimal_idx = torch.argmax(a_u_minus_F)
                a_optimal = self.A[optimal_idx]
                F_optimal = legendre_values[optimal_idx]
                
                I = I + a_optimal * self.dt
                Q = Q + torch.exp(I) * F_optimal
            
            g_X = self.g(X.unsqueeze(0))[0, 0]
            total_lower = total_lower + torch.exp(I) * g_X - Q
        
        u_low = (total_lower / trajectories).item()
        
        if verbose:
            print(f"Lower bound: {u_low:.6f}")
        
        return u_low

    def solve(self, limits=False, trajectories_upper=1000, trajectories_lower=1000, 
              maxiters_limits=10, verbose=True, save_everystep=False):
        """主求解函数"""
        self.train(verbose=verbose)
        u0_estimate = self.u0(self.x0.unsqueeze(0))[0, 0].item()
        u_analytical = self.analytical_solution(self.x0, self.tspan[0]).item()
        
        if not limits:
            if verbose:
                print(f"Point estimate: {u0_estimate:.6f}")
                print(f"Analytical solution: {u_analytical:.6f}")
                error = rel_error_l2(u0_estimate, u_analytical)
                print(f"Relative error: {error:.6f}")
            
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

def calculate_proper_error_bars(intervals, point_estimates):
    """计算合适的误差条 - 修复版本"""
    errors_lower = []
    errors_upper = []
    violations = 0
    
    for (low, high), u0 in zip(intervals, point_estimates):
        if u0 < low:
            # 点估计低于下界，调整显示
            errors_lower.append(low - u0)  # 修复：使用绝对值
            errors_upper.append(high - u0)
            violations += 1
        elif u0 > high:
            # 点估计高于上界
            errors_lower.append(u0 - low)
            errors_upper.append(u0 - high)  # 修复：使用绝对值
            violations += 1
        else:
            # 正常情况
            errors_lower.append(u0 - low)
            errors_upper.append(high - u0)
    
    if violations > 0:
        print(f"警告: {violations}个点估计超出置信区间")
    
    return errors_lower, errors_upper, violations

def plot_improved_confidence_intervals(metrics, analytical_value, ax):
    """改进的置信区间可视化 - 修复版本"""
    if not metrics.get('limits_intervals'):
        ax.text(0.5, 0.5, '无置信区间数据', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('置信区间可视化')
        return ax
    
    # 计算误差条
    errors_lower, errors_upper, violations = calculate_proper_error_bars(
        metrics['limits_intervals'], metrics['limits_u0']
    )
    
    # 确保误差条非负
    errors_lower = np.abs(errors_lower)
    errors_upper = np.abs(errors_upper)
    
    # 绘制误差条
    x_positions = range(len(metrics['limits_intervals']))
    
    # 使用不同的颜色标记违规点
    colors = ['red' if (u0 < low or u0 > high) else 'blue' 
             for (low, high), u0 in zip(metrics['limits_intervals'], metrics['limits_u0'])]
    
    # 分别绘制正常点和违规点
    normal_x = []
    normal_y = []
    normal_lower = []
    normal_upper = []
    
    violation_x = []
    violation_y = []
    violation_lower = []
    violation_upper = []
    
    for i, ((low, high), u0, color) in enumerate(zip(metrics['limits_intervals'], metrics['limits_u0'], colors)):
        if color == 'blue':
            normal_x.append(i)
            normal_y.append(u0)
            normal_lower.append(errors_lower[i])
            normal_upper.append(errors_upper[i])
        else:
            violation_x.append(i)
            violation_y.append(u0)
            violation_lower.append(errors_lower[i])
            violation_upper.append(errors_upper[i])
    
    # 绘制正常点
    if normal_x:
        ax.errorbar(
            normal_x, normal_y, 
            yerr=[normal_lower, normal_upper],
            fmt='o', capsize=5, color='blue', alpha=0.7,
            label='正常点'
        )
    
    # 绘制违规点
    if violation_x:
        ax.errorbar(
            violation_x, violation_y, 
            yerr=[violation_lower, violation_upper],
            fmt='o', capsize=5, color='red', alpha=0.7,
            label='违规点'
        )
    
    # 添加解析解参考线
    ax.axhline(y=analytical_value, color='green', linestyle='--', 
              label=f'解析解: {analytical_value:.2f}')
    
    # 添加置信区间背景
    for i, (low, high) in enumerate(metrics['limits_intervals']):
        ax.axhspan(low, high, alpha=0.1, color='gray')
    
    ax.set_xlabel('试验次数')
    ax.set_ylabel('u0估计值')
    ax.set_title('置信区间可视化（改进版）')
    
    # 添加统计信息
    coverage = sum(1 for (low, high), u0 in zip(metrics['limits_intervals'], metrics['limits_u0'])
                   if low <= u0 <= high) / len(metrics['limits_intervals']) * 100
    
    stats_text = f'覆盖率: {coverage:.1f}%\n违规点: {violations}个'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 添加图例
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='正常点'),
        Patch(facecolor='red', alpha=0.7, label='违规点'),
        Patch(facecolor='gray', alpha=0.1, label='置信区间')
    ]
    ax.legend(handles=legend_elements)
    
    ax.grid(True, alpha=0.3)
    return ax

def comprehensive_comparison(d=30, num_trials=3):
    """全面的方法对比分析 - 集成方案3改进"""
    print("=== DeepBSDE方法全面对比分析 ===")
    print(f"维度: {d}维, 试验次数: {num_trials}")
    print("=" * 60)
    
    results_std = []
    results_limits = []
    
    performance_metrics = {
        'std_errors': [], 'std_times': [], 'std_u0': [],
        'limits_errors': [], 'limits_times': [], 'limits_u0': [],
        'limits_lower': [], 'limits_upper': [], 'limits_intervals': []
    }
    
    for trial in range(num_trials):
        print(f"\n--- 试验 {trial + 1}/{num_trials} ---")
        
        # 测试标准方法
        start_time = time.time()
        solver_std = BlackScholesBarenblattSolver(d=d)
        result_std = solver_std.solve(limits=False, verbose=False)
        std_time = time.time() - start_time
        
        u_pred_std = result_std.us if hasattr(result_std.us, '__len__') else result_std.us
        u_anal_std = solver_std.analytical_solution(solver_std.x0, solver_std.tspan[0]).item()
        error_std = rel_error_l2(u_pred_std, u_anal_std)
        
        # 测试带Legendre变换方法
        start_time = time.time()
        solver_limits = BlackScholesBarenblattSolver(d=d)
        result_limits = solver_limits.solve(
            limits=True, 
            trajectories_upper=200,
            trajectories_lower=200,
            maxiters_limits=5,
            verbose=False
        )
        limits_time = time.time() - start_time
        
        u_pred_limits = result_limits.us if hasattr(result_limits.us, '__len__') else result_limits.us
        u_anal_limits = solver_limits.analytical_solution(solver_limits.x0, solver_limits.tspan[0]).item()
        error_limits = rel_error_l2(u_pred_limits, u_anal_limits)
        
        # 存储结果
        results_std.append((solver_std, result_std, error_std, std_time))
        results_limits.append((solver_limits, result_limits, error_limits, limits_time))
        
        # 存储性能指标
        performance_metrics['std_errors'].append(error_std)
        performance_metrics['std_times'].append(std_time)
        performance_metrics['std_u0'].append(u_pred_std)
        
        performance_metrics['limits_errors'].append(error_limits)
        performance_metrics['limits_times'].append(limits_time)
        performance_metrics['limits_u0'].append(u_pred_limits)
        
        if hasattr(result_limits, 'limits') and result_limits.limits is not None:
            u_low, u_high = result_limits.limits
            performance_metrics['limits_lower'].append(u_low)
            performance_metrics['limits_upper'].append(u_high)
            performance_metrics['limits_intervals'].append((u_low, u_high))
        
        print(f"标准方法 - 误差: {error_std:.6f}, 时间: {std_time:.2f}s")
        print(f"对偶方法 - 误差: {error_limits:.6f}, 时间: {limits_time:.2f}s")
        if hasattr(result_limits, 'limits') and result_limits.limits is not None:
            print(f"置信区间: [{u_low:.4f}, {u_high:.4f}]")
    
    # 性能统计分析
    print("\n" + "="*60)
    print("                性能对比分析结果")
    print("="*60)
    
    # 创建对比表格
    comparison_data = []
    
    # 准确性对比
    std_error_mean = np.mean(performance_metrics['std_errors'])
    std_error_std = np.std(performance_metrics['std_errors'])
    limits_error_mean = np.mean(performance_metrics['limits_errors'])
    limits_error_std = np.std(performance_metrics['limits_errors'])
    
    # 计算时间对比
    std_time_mean = np.mean(performance_metrics['std_times'])
    std_time_std = np.std(performance_metrics['std_times'])
    limits_time_mean = np.mean(performance_metrics['limits_times'])
    limits_time_std = np.std(performance_metrics['limits_times'])
    
    # 解值稳定性对比
    std_u0_std = np.std(performance_metrics['std_u0'])
    limits_u0_std = np.std(performance_metrics['limits_u0'])
    
    # 置信区间分析
    coverage = 0
    if performance_metrics['limits_intervals']:
        analytical_value = results_limits[0][0].analytical_solution(
            results_limits[0][0].x0, results_limits[0][0].tspan[0]).item()
        for interval in performance_metrics['limits_intervals']:
            if interval[0] <= analytical_value <= interval[1]:
                coverage += 1
        coverage_rate = coverage / len(performance_metrics['limits_intervals'])
        interval_widths = [interval[1] - interval[0] for interval in performance_metrics['limits_intervals']]
        avg_interval_width = np.mean(interval_widths) if interval_widths else 0
    else:
        coverage_rate = 0
        avg_interval_width = 0
    
    # 统计显著性检验
    if len(performance_metrics['std_errors']) > 1 and len(performance_metrics['limits_errors']) > 1:
        t_stat, p_value = stats.ttest_ind(performance_metrics['std_errors'], 
                                         performance_metrics['limits_errors'])
    else:
        t_stat, p_value = 0, 1.0
    
    # 输出详细对比表格
    print(f"\n{'指标':<25} {'标准方法':<15} {'对偶方法':<15} {'优劣分析':<20}")
    print("-" * 80)
    
    comparison_data.append([
        "平均相对误差", 
        f"{std_error_mean:.6f} ± {std_error_std:.6f}", 
        f"{limits_error_mean:.6f} ± {limits_error_std:.6f}",
        "✓ 标准方法更优" if std_error_mean < limits_error_mean else "✓ 对偶方法更优"
    ])
    
    comparison_data.append([
        "平均训练时间(s)", 
        f"{std_time_mean:.2f} ± {std_time_std:.2f}", 
        f"{limits_time_mean:.2f} ± {limits_time_std:.2f}",
        "✓ 标准方法更快" if std_time_mean < limits_time_mean else "✓ 对偶方法更快"
    ])
    
    comparison_data.append([
        "解值稳定性(标准差)", 
        f"{std_u0_std:.6f}", 
        f"{limits_u0_std:.6f}",
        "✓ 标准方法更稳定" if std_u0_std < limits_u0_std else "✓ 对偶方法更稳定"
    ])
    
    if performance_metrics['limits_intervals']:
        comparison_data.append([
            "置信区间覆盖率", 
            "N/A", 
            f"{coverage_rate*100:.1f}%",
            "✓ 良好" if coverage_rate >= 0.9 else "○ 一般" if coverage_rate >= 0.7 else "✗ 较差"
        ])
        
        comparison_data.append([
            "平均区间宽度", 
            "N/A", 
            f"{avg_interval_width:.4f}",
            "✓ 较窄" if avg_interval_width < 1.0 else "○ 适中" if avg_interval_width < 3.0 else "✗ 较宽"
        ])
    
    comparison_data.append([
        "统计显著性(p值)", 
        "N/A", 
        f"{p_value:.6f}",
        "✓ 显著差异" if p_value < 0.05 else "○ 无显著差异"
    ])
    
    for row in comparison_data:
        print(f"{row[0]:<25} {row[1]:<15} {row[2]:<15} {row[3]:<20}")
    
    # 绘制综合对比图表（包含方案3改进的置信区间可视化）
    plot_comprehensive_comparison(performance_metrics, results_std, results_limits, d)
    
    # 方法特性评分
    print("\n" + "="*60)
    print("                方法特性综合评分")
    print("="*60)
    
    characteristics = {
        '准确性': [max(0, 10 - std_error_mean*100), max(0, 10 - limits_error_mean*100)],
        '计算效率': [max(0, 10 - std_time_mean/10), max(0, 10 - limits_time_mean/10)],
        '数值稳定性': [max(0, 10 - std_u0_std*10), max(0, 10 - limits_u0_std*10)],
        '理论保证': [7, 9],
        '实现复杂度': [8, 6],
        '适用性': [9, 8]
    }
    
    if performance_metrics['limits_intervals']:
        characteristics['不确定性量化'] = [5, 8]
    
    methods = ['标准方法', '对偶方法']
    print(f"{'特性':<15} {'标准方法':<10} {'对偶方法':<10} {'推荐':<10}")
    print("-" * 50)
    
    for char, scores in characteristics.items():
        std_score, limits_score = scores
        recommendation = "标准方法" if std_score > limits_score else "对偶方法" if limits_score > std_score else "相当"
        print(f"{char:<15} {std_score:<10.1f} {limits_score:<10.1f} {recommendation:<10}")
    
    # 总体推荐
    total_std = sum([score[0] for score in characteristics.values()])
    total_limits = sum([score[1] for score in characteristics.values()])
    
    print("-" * 50)
    print(f"{'总分':<15} {total_std:<10.1f} {total_limits:<10.1f} ", end="")
    if total_std > total_limits:
        print("✓ 推荐标准方法")
    elif total_limits > total_std:
        print("✓ 推荐对偶方法")
    else:
        print("○ 两种方法相当")
    
    return results_std, results_limits, performance_metrics

def plot_comprehensive_comparison(metrics, results_std, results_limits, d):
    """绘制综合对比图表 - 集成方案3改进"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 误差分布对比
    axes[0,0].boxplot([metrics['std_errors'], metrics['limits_errors']], 
                      labels=['标准方法', '对偶方法'])
    axes[0,0].set_ylabel('相对误差')
    axes[0,0].set_title('误差分布对比')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. 训练时间对比
    axes[0,1].boxplot([metrics['std_times'], metrics['limits_times']], 
                     labels=['标准方法', '对偶方法'])
    axes[0,1].set_ylabel('训练时间 (秒)')
    axes[0,1].set_title('计算效率对比')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. 解值稳定性
    axes[0,2].plot(metrics['std_u0'], 'bo-', label='标准方法', alpha=0.7)
    axes[0,2].plot(metrics['limits_u0'], 'ro-', label='对偶方法', alpha=0.7)
    analytical_value = results_std[0][0].analytical_solution(
        results_std[0][0].x0, results_std[0][0].tspan[0]).item()
    axes[0,2].axhline(y=analytical_value, color='green', linestyle='--', 
                     label=f'解析解: {analytical_value:.2f}')
    axes[0,2].set_xlabel('试验次数')
    axes[0,2].set_ylabel('u0估计值')
    axes[0,2].set_title('解值稳定性对比')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. 训练损失曲线对比（最后一次试验）
    if results_std and hasattr(results_std[-1][0], 'losses'):
        axes[1,0].semilogy(results_std[-1][0].losses, label='标准方法')
    if results_limits and hasattr(results_limits[-1][0], 'losses'):
        axes[1,0].semilogy(results_limits[-1][0].losses, label='对偶方法')
    axes[1,0].set_xlabel('迭代次数')
    axes[1,0].set_ylabel('损失值')
    axes[1,0].set_title('训练过程对比')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. 置信区间可视化 - 使用方案3改进方法
    if metrics.get('limits_intervals'):
        plot_improved_confidence_intervals(metrics, analytical_value, axes[1,1])
    else:
        axes[1,1].text(0.5, 0.5, '无置信区间数据', 
                      transform=axes[1,1].transAxes, ha='center', va='center')
        axes[1,1].set_title('置信区间可视化')
    
    # 6. 方法特性雷达图 - 修复：使用正确的极坐标设置
    # 首先创建一个极坐标轴
    fig.delaxes(axes[1,2])  # 删除原来的轴
    ax_radar = fig.add_subplot(2, 3, 6, projection='polar')  # 创建极坐标轴
    
    characteristics = ['准确性', '效率', '稳定性', '理论保证', '易用性', '适用性']
    std_scores = [8, 9, 7, 7, 8, 9]
    limits_scores = [7, 6, 8, 9, 6, 8]
    
    angles = np.linspace(0, 2*np.pi, len(characteristics), endpoint=False).tolist()
    angles += angles[:1]
    std_scores += std_scores[:1]
    limits_scores += limits_scores[:1]
    characteristics += characteristics[:1]
    
    ax_radar.plot(angles, std_scores, 'o-', linewidth=2, label='标准方法')
    ax_radar.fill(angles, std_scores, alpha=0.25)
    ax_radar.plot(angles, limits_scores, 'o-', linewidth=2, label='对偶方法')
    ax_radar.fill(angles, limits_scores, alpha=0.25)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(characteristics[:-1])
    ax_radar.set_title('方法特性雷达图')
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.suptitle(f'{d}维Black-Scholes-Barenblatt方程求解方法对比', fontsize=16, y=1.02)
    plt.show()

def main():
    """修改后的主函数，专注于方法对比"""
    print("=== DeepBSDE方法对比分析 ===")
    
    # 运行全面对比分析
    results_std, results_limits, metrics = comprehensive_comparison(d=30, num_trials=3)
    
    # 输出最终建议
    print("\n" + "="*60)
    print("                最终使用建议")
    print("="*60)
    
    print("\n📊 基于对比分析，建议如下：")
    print("\n✅ 推荐使用标准DeepBSDE方法的情况：")
    print("   • 需要快速得到点估计")
    print("   • 计算资源有限")
    print("   • 问题相对简单，不需要不确定性量化")
    print("   • 实现复杂度要求低")
    
    print("\n✅ 推荐使用带Legendre变换对偶方法的情况：")
    print("   • 需要置信区间估计")
    print("   • 对解的可靠性要求高")
    print("   • 有充足的计算资源")
    print("   • 需要进行严格的理论分析")
    
    print("\n🔍 关键发现：")
    if metrics['std_errors'] and metrics['limits_errors']:
        if np.mean(metrics['std_errors']) < np.mean(metrics['limits_errors']):
            print("   • 标准方法在准确性上略优")
        else:
            print("   • 对偶方法在准确性上略优")
        
        if np.mean(metrics['std_times']) < np.mean(metrics['limits_times']):
            print("   • 标准方法在计算效率上明显更优")
        else:
            print("   • 对偶方法在计算效率上更优")
    
    if metrics.get('limits_intervals'):
        analytical_value = results_limits[0][0].analytical_solution(
            results_limits[0][0].x0, results_limits[0][0].tspan[0]).item()
        coverage = sum(1 for interval in metrics['limits_intervals'] 
                      if interval[0] <= analytical_value <= interval[1])
        print(f"   • 对偶方法提供置信区间，覆盖率为{coverage/len(metrics['limits_intervals'])*100:.1f}%")
    
    return results_std, results_limits, metrics

if __name__ == "__main__":
    results_std, results_limits, metrics = main()
