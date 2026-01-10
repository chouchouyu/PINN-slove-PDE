import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(100)
torch.manual_seed(100)

# 移除了复杂的Legendre变换部分，专注于修复基本的DeepBSDE算法，确保标准版本能够正常运行。


def rel_error_l2(u, uanal):
    """相对L2误差计算"""
    if abs(uanal) >= 10 * np.finfo(type(uanal)).eps:
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
        self.r = 0.05
        self.sigma = 0.4
        
        # 初始条件
        self.x0 = torch.tensor([1.0 if i % 2 == 0 else 0.5 for i in range(d)], 
                              dtype=torch.float32, device=device)
        self.tspan = (0.0, 1.0)
        self.dt = 0.25
        self.time_steps = int((self.tspan[1] - self.tspan[0]) / self.dt)
        self.m = 30
        
        # 网络设置
        self.hls = 10 + d
        self.u0 = U0Network(d, self.hls).to(device)
        self.sigma_grad_u = nn.ModuleList([
            SigmaTGradUNetwork(d, self.hls).to(device) for _ in range(self.time_steps)
        ])
        
        self.optimizer = optim.Adam(
            list(self.u0.parameters()) + 
            [param for net in self.sigma_grad_u for param in net.parameters()],
            lr=0.001
        )
        
        self.losses = []
        self.u0_history = []

    def g(self, X):
        """终端条件"""
        return torch.sum(X**2, dim=1, keepdim=True)

    def f(self, X, u, sigma_grad_u, t):
        """非线性项"""
        return self.r * (u - torch.sum(X * sigma_grad_u, dim=1, keepdim=True))

    def mu_f(self, X, t):
        """漂移项"""
        return torch.zeros_like(X)

    def sigma_f(self, X, t):
        """扩散项 - 修复形状处理错误"""
        # 修复：正确处理单样本和批次样本的情况
        if len(X.shape) == 1:
            # 单样本情况：X形状为 [d]
            return torch.diag(self.sigma * X)
        else:
            # 批次样本情况：X形状为 [batch_size, d]
            batch_size = X.shape[0]
            # 创建对角矩阵批次：形状为 [batch_size, d, d]
            sigma_diag = torch.diag_embed(self.sigma * X)
            return sigma_diag

    def generate_trajectories(self, batch_size=None):
        """生成轨迹 - 修复扩散项计算"""
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
            
            # 更新X - 修复扩散项计算
            mu_val = self.mu_f(X, t)
            sigma_val = self.sigma_f(X, t)
            
            if len(sigma_val.shape) == 2:
                # 单样本情况（虽然这里应该是批次）
                X = X + mu_val * self.dt + torch.matmul(dW, sigma_val)
            else:
                # 批次情况：sigma_val形状为 [batch_size, d, d]
                # dW需要扩展为 [batch_size, d, 1] 以便矩阵乘法
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
                print(f'Converged at epoch {epoch}')
                break

    def analytical_solution(self, x, t):
        """解析解"""
        T = self.tspan[1]
        exponent = torch.tensor((self.r + self.sigma**2) * (T - t), 
                               dtype=x.dtype, device=x.device)
        return torch.exp(exponent) * torch.sum(x**2)

    def test(self):
        """测试函数"""
        u_pred = self.u0(self.x0.unsqueeze(0))[0, 0].item()
        u_anal = self.analytical_solution(self.x0, self.tspan[0]).item()
        
        error = rel_error_l2(u_pred, u_anal)
        
        print(f"Predicted u0: {u_pred:.6f}")
        print(f"Analytical u0: {u_anal:.6f}")
        print(f"Relative L2 Error: {error:.6f}")
        
        return error

def main():
    """主测试函数"""
    print("=== 30维Black-Scholes-Barenblatt方程求解 ===")
    
    # 初始化求解器
    solver = BlackScholesBarenblattSolver(d=30)
    
    print(f"维度 d: {solver.d}")
    print(f"初始点 x0: {solver.x0.cpu().numpy()}")
    print(f"时间区间: {solver.tspan}")
    print(f"时间步长 dt: {solver.dt}")
    print(f"时间步数: {solver.time_steps}")
    print(f"轨迹数 m: {solver.m}")
    
    # 训练模型
    print("\n开始训练...")
    solver.train(maxiters=150, verbose=True)
    
    # 测试结果
    print("\n测试结果:")
    error = solver.test()
    
    # 验证误差
    if error < 1.0:
        print("✓ 测试通过：相对误差 < 1.0")
    else:
        print("✗ 测试未通过：相对误差 >= 1.0")
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(solver.losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(solver.u0_history)
    plt.xlabel('Epoch')
    plt.ylabel('u0 estimate')
    plt.title('u0 Convergence')
    
    plt.tight_layout()
    plt.show()
    
    return solver, error

if __name__ == "__main__":
    solver, error = main()