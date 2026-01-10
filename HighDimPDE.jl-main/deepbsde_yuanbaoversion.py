import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import matplotlib.pyplot as plt

class DeepBSDENetwork(nn.Module):
    """DeepBSDE网络结构 - 与Julia代码完全对应"""
    def __init__(self, d, time_steps):
        super(DeepBSDENetwork, self).__init__()
        self.d = d
        self.time_steps = time_steps
        self.hls = 10 + d  # 隐藏层大小：40 (10 + 30)
        
        # u0网络：近似初始解值
        self.u0 = nn.Sequential(
            nn.Linear(d, self.hls),
            nn.ReLU(),
            nn.Linear(self.hls, self.hls),
            nn.ReLU(), 
            nn.Linear(self.hls, 1)
        )
        
        # σᵀ∇u网络数组：每个时间步一个独立网络
        self.sigma_grad_u = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, self.hls),
                nn.ReLU(),
                nn.Linear(self.hls, self.hls),
                nn.ReLU(),
                nn.Linear(self.hls, self.hls),  # 额外隐藏层（与Julia代码一致）
                nn.ReLU(),
                nn.Linear(self.hls, d)
            ) for _ in range(time_steps)
        ])
    
    def forward(self, X0):
        return self.u0(X0)

class BlackScholesBarenblattSolver:
    """Black-Scholes-Barenblatt方程求解器"""
    def __init__(self, d=30, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.d = d
        self.device = device
        self.r = 0.05
        self.sigma = 0.4
        
        # 与Julia代码完全相同的参数
        self.x0 = torch.tensor([1.0 if i % 2 == 0 else 0.5 for i in range(d)], 
                              dtype=torch.float32, device=device)
        self.tspan = (0.0, 1.0)
        self.dt = 0.25
        self.time_steps = int((self.tspan[1] - self.tspan[0]) / self.dt)
        self.m = 30  # 轨迹数
        
        # 初始化网络
        self.model = DeepBSDENetwork(d, self.time_steps).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # 存储训练历史
        self.losses = []
        self.iters = []
    
    def g(self, X):
        """终端条件：g(X) = sum(X^2)"""
        return torch.sum(X ** 2, dim=1, keepdim=True)
    
    def f(self, X, u, sigma_grad_u, t):
        """非线性项：f(X, u, σᵀ∇u, p, t) = r * (u - sum(X * σᵀ∇u))"""
        return self.r * (u - torch.sum(X * sigma_grad_u, dim=1, keepdim=True))
    
    def mu_f(self, X, t):
        """漂移项：μ(X, p, t) = 0"""
        return torch.zeros_like(X)
    
    def sigma_f(self, X, t):
        """扩散项：σ(X, p, t) = Diagonal(sigma * X)"""
        batch_size = X.shape[0]
        # 修复：正确处理批次维度
        if len(X.shape) == 1:
            return torch.diag(self.sigma * X)
        else:
            return torch.diag_embed(self.sigma * X)
    
    def generate_trajectories(self, batch_size=None):
        """生成轨迹 - 与Julia代码逻辑完全一致"""
        if batch_size is None:
            batch_size = self.m
            
        # 初始化
        u = self.model.u0(self.x0.repeat(batch_size, 1))
        X = self.x0.repeat(batch_size, 1)
        
        ts = torch.arange(self.tspan[0], self.tspan[1] + self.dt, self.dt, device=self.device)
        
        for i in range(len(ts) - 1):
            t = ts[i].item()  # 获取标量值
            
            # 获取当前时间步的梯度网络
            sigma_grad_u_val = self.model.sigma_grad_u[i](X)
            
            # 生成随机噪声
            dW = torch.randn(batch_size, self.d, device=self.device) * np.sqrt(self.dt)
            
            # 更新u（BSDE离散形式）
            u = u - self.f(X, u, sigma_grad_u_val, t) * self.dt + torch.sum(sigma_grad_u_val * dW, dim=1, keepdim=True)
            
            # 更新X（SDE离散形式）- 修复扩散项计算
            sigma_matrix = self.sigma_f(X, t)
            if len(X.shape) == 1:
                # 单样本情况
                dW_expanded = dW.squeeze(0) if batch_size == 1 else dW
                X = X + self.mu_f(X, t) * self.dt + torch.matmul(sigma_matrix, dW_expanded)
            else:
                # 批次情况
                dW_expanded = dW.unsqueeze(-1)
                X_update = torch.matmul(sigma_matrix, dW_expanded).squeeze(-1)
                X = X + self.mu_f(X, t) * self.dt + X_update
        
        return X, u
    
    def loss_function(self):
        """损失函数：终端条件匹配损失"""
        X_final, u_final = self.generate_trajectories()
        g_X = self.g(X_final)
        loss = torch.mean((g_X - u_final) ** 2)
        return loss
    
    def analytical_solution(self, x, t):
        """解析解：u_analytical(x, t) = exp((r + sigma^2) * (T - t)) * sum(x^2)"""
        T = self.tspan[1]
        # 修复：将浮点数转换为张量
        exponent = torch.tensor((self.r + self.sigma**2) * (T - t), 
                               dtype=x.dtype, device=x.device)
        return torch.exp(exponent) * torch.sum(x**2)
    
    def rel_error_l2(self, u, u_anal):
        """相对L2误差计算"""
        u_tensor = torch.tensor(u, dtype=torch.float32, device=self.device)
        u_anal_tensor = torch.tensor(u_anal, dtype=torch.float32, device=self.device)
        
        if abs(u_anal_tensor) >= 10 * torch.finfo(torch.float32).eps:
            return torch.sqrt(((u_tensor - u_anal_tensor) ** 2) / (u_anal_tensor ** 2))
        else:
            return torch.abs(u_tensor - u_anal_tensor)
    
    def train(self, maxiters=150, abstol=1e-8, verbose=True):
        """训练过程 - 与Julia代码逻辑一致"""
        for epoch in range(maxiters):
            self.optimizer.zero_grad()
            loss = self.loss_function()
            loss.backward()
            self.optimizer.step()
            
            self.losses.append(loss.item())
            current_u0 = self.model.u0(self.x0.unsqueeze(0))[0,0].item()
            self.iters.append(current_u0)
            
            if verbose and (epoch % 10 == 0 or epoch == maxiters - 1):
                print(f'Epoch {epoch}, Loss: {loss.item():.6f}, u0: {current_u0:.6f}')
            
            if loss.item() < abstol:
                print(f'Converged at epoch {epoch}')
                break
    
    def test(self):
        """测试函数 - 计算相对误差"""
        u_pred = self.model.u0(self.x0.unsqueeze(0))[0,0].item()
        u_anal = self.analytical_solution(self.x0, self.tspan[0]).item()  # 现在应该能正常工作
        
        error = self.rel_error_l2(u_pred, u_anal).item()
        
        print(f"Predicted u0: {u_pred:.6f}")
        print(f"Analytical u0: {u_anal:.6f}")
        print(f"Relative L2 Error: {error:.6f}")
        
        return error

def main():
    """主函数 - 运行完整的测试流程"""
    # 设置随机种子以保证可重复性
    torch.manual_seed(100)
    np.random.seed(100)
    
    # 初始化求解器（30维问题）
    solver = BlackScholesBarenblattSolver(d=30)
    
    print("=== 30维Black-Scholes-Barenblatt方程求解 ===")
    print(f"维度 d: {solver.d}")
    print(f"初始点 x0: {solver.x0.cpu().numpy()}")
    print(f"时间区间: {solver.tspan}")
    print(f"时间步长 dt: {solver.dt}")
    print(f"时间步数: {solver.time_steps}")
    print(f"轨迹数 m: {solver.m}")
    print(f"网络隐藏层大小: {solver.model.hls}")
    
    # 训练模型
    print("\n开始训练...")
    solver.train(maxiters=150, verbose=True)
    
    # 测试结果
    print("\n测试结果:")
    error = solver.test()
    
    # 验证误差是否小于1.0（与Julia测试用例相同标准）
    if error < 1.0:
        print("✓ 测试通过：相对误差 < 1.0")
    else:
        print("✗ 测试未通过：相对误差 >= 1.0")
    
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(solver.losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(solver.iters)
    plt.xlabel('Epoch')
    plt.ylabel('u0 estimate')
    plt.title('u0 Convergence')
    
    plt.tight_layout()
    plt.show()
    
    return solver, error

if __name__ == "__main__":
    solver, error = main()