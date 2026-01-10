import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Callable, Tuple, List, Optional
import matplotlib.pyplot as plt




class DeepBSDESolver:
    """
    深度BSDE求解器，用于求解高维抛物型偏微分方程
    
    参数:
    - d: 空间维度
    - T: 终止时间
    - dt: 时间步长
    - n_time_steps: 时间步数
    - n_trajectories: 模拟轨迹数
    - n_iterations: 训练迭代次数
    - hidden_size: 神经网络隐藏层大小
    - learning_rate: 学习率
    - device: 计算设备 ('cpu' 或 'cuda')
    """
    
    def __init__(self, d: int, T: float = 1.0, dt: float = 0.01,
                 n_trajectories: int = 100, n_iterations: int = 1000,
                 hidden_size: int = 20, learning_rate: float = 0.001,
                 device: str = 'cpu'):
        self.d = d
        self.T = T
        self.dt = dt
        self.n_time_steps = int(T / dt)
        self.n_trajectories = n_trajectories
        self.n_iterations = n_iterations
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        
        # 时间网格
        self.ts = torch.linspace(0, T, self.n_time_steps + 1, device=self.device)
        
        # 初始化神经网络
        self.u0_net = self._build_u0_network()
        self.sigma_nets = nn.ModuleList([
            self._build_sigma_network() for _ in range(self.n_time_steps)
        ])
        
        # 优化器
        self.optimizer = optim.Adam(
            list(self.u0_net.parameters()) + list(self.sigma_nets.parameters()),
            lr=learning_rate
        )
        
        # 损失历史
        self.loss_history = []
        
    def _build_u0_network(self) -> nn.Module:
        """构建初始值网络"""
        return nn.Sequential(
            nn.Linear(self.d, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        ).to(self.device)
    
    def _build_sigma_network(self) -> nn.Module:
        """构建sigma转置梯度网络"""
        return nn.Sequential(
            nn.Linear(self.d, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.d)
        ).to(self.device)
    
    def simulate_trajectories(self, X0: torch.Tensor, 
                             mu_func: Callable, 
                             sigma_func: Callable,
                             f_func: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        模拟轨迹，返回终端位置和终端值
        
        参数:
        - X0: 初始位置，形状为 (n_trajectories, d)
        - mu_func: 漂移系数函数
        - sigma_func: 扩散系数函数
        - f_func: 非线性项函数
        """
        batch_size = X0.shape[0]
        X = X0.clone()
        u = self.u0_net(X0)
        
        for i in range(self.n_time_steps):
            t = self.ts[i]
            
            # 计算sigma转置梯度
            sigma_T_grad_u = self.sigma_nets[i](X)
            
            # 生成布朗运动增量
            dW = torch.sqrt(torch.tensor(self.dt, device=self.device)) * torch.randn_like(X)
            
            # 更新u (BSDE离散)
            f_value = f_func(X, u, sigma_T_grad_u, t)
            u = u - f_value * self.dt + torch.sum(sigma_T_grad_u * dW, dim=1, keepdim=True)
            
            # 更新X (SDE离散)
            mu_value = mu_func(X, t)
            sigma_value = sigma_func(X, t)
            X = X + mu_value * self.dt + sigma_value * dW
        
        return X, u
    
    def compute_loss(self, terminal_X: torch.Tensor, terminal_u: torch.Tensor,
                    g_func: Callable) -> torch.Tensor:
        """
        计算损失函数 (终端条件与模拟值之间的均方误差)
        """
        g_value = g_func(terminal_X)
        loss = torch.mean((g_value - terminal_u.squeeze())**2)
        return loss
    
    def train(self, X0: torch.Tensor,
              mu_func: Callable,
              sigma_func: Callable,
              f_func: Callable,
              g_func: Callable,
              verbose: bool = True) -> List[float]:
        """
        训练神经网络
        
        参数:
        - X0: 初始位置，形状为 (1, d) 或 (d,)
        - mu_func: 漂移系数函数
        - sigma_func: 扩散系数函数
        - f_func: 非线性项函数
        - g_func: 终端条件函数
        - verbose: 是否打印训练信息
        """
        # 确保X0形状正确
        if len(X0.shape) == 1:
            X0 = X0.unsqueeze(0)
        if X0.shape[0] == 1:
            X0 = X0.repeat(self.n_trajectories, 1)
        
        X0 = X0.to(self.device)
        self.loss_history = []
        
        for iteration in range(self.n_iterations):
            # 前向传播
            terminal_X, terminal_u = self.simulate_trajectories(X0, mu_func, sigma_func, f_func)
            
            # 计算损失
            loss = self.compute_loss(terminal_X, terminal_u, g_func)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 记录损失
            self.loss_history.append(loss.item())
            
            # 打印训练信息
            if verbose and (iteration + 1) % 100 == 0:
                print(f'迭代 {iteration + 1}/{self.n_iterations}, 损失: {loss.item():.6f}')
        
        return self.loss_history
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        使用训练好的网络预测初始值
        """
        with torch.no_grad():
            return self.u0_net(X.to(self.device))
    
    def plot_training_history(self):
        """绘制训练损失历史"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.xlabel('迭代次数')
        plt.ylabel('损失')
        plt.title('训练损失历史')
        plt.grid(True, alpha=0.3)
        plt.show()


# 示例：Black-Scholes-Barenblatt 方程
def black_scholes_mu(X: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Black-Scholes 漂移系数"""
    return 0.0 * X  # 无风险利率为0

def black_scholes_sigma(X: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Black-Scholes 波动率"""
    return 0.4 * X  # 波动率为0.4

def black_scholes_f(X: torch.Tensor, u: torch.Tensor, 
                   sigma_T_grad_u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Black-Scholes 非线性项"""
    return -0.0 * u  # 无风险利率为0

def black_scholes_g(X: torch.Tensor) -> torch.Tensor:
    """Black-Scholes 终端条件 (欧式看涨期权 payoff)"""
    strike = 1.0
    return torch.clamp(X - strike, min=0.0)


# 测试函数
def test_black_scholes():
    """测试 Black-Scholes 方程求解"""
    print("测试 Black-Scholes 方程求解...")
    
    # 设置参数
    d = 1  # 空间维度
    T = 1.0  # 终止时间
    dt = 0.01  # 时间步长
    n_trajectories = 100  # 轨迹数
    n_iterations = 1000  # 迭代次数
    
    # 创建求解器
    solver = DeepBSDESolver(
        d=d,
        T=T,
        dt=dt,
        n_trajectories=n_trajectories,
        n_iterations=n_iterations,
        hidden_size=20,
        learning_rate=0.001,
        device='cpu'
    )
    
    # 初始位置
    X0 = torch.ones((1, d)) * 1.0  # 初始价格为1.0
    
    # 训练
    print("开始训练...")
    loss_history = solver.train(
        X0=X0,
        mu_func=black_scholes_mu,
        sigma_func=black_scholes_sigma,
        f_func=black_scholes_f,
        g_func=black_scholes_g,
        verbose=True
    )
    
    # 预测
    test_X = torch.tensor([[0.8], [0.9], [1.0], [1.1], [1.2]], dtype=torch.float32)
    predictions = solver.predict(test_X)
    
    print("\n预测结果:")
    for i, (x, pred) in enumerate(zip(test_X, predictions)):
        print(f"初始价格: {x.item():.2f}, 期权价值: {pred.item():.6f}")
#     初始价格: 0.80, 期权价值: 0.156258
# 初始价格: 0.90, 期权价值: 0.150484
# 初始价格: 1.00, 期权价值: 0.143877
# 初始价格: 1.10, 期权价值: 0.133801
# 初始价格: 1.20, 期权价值: 0.123139
    # 绘制训练历史
    solver.plot_training_history()
    
    return solver, loss_history


# Hamilton-Jacobi-Bellman 方程示例
def hjb_mu(X: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """HJB 方程的漂移系数"""
    return torch.zeros_like(X)

def hjb_sigma(X: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """HJB 方程的扩散系数"""
    return torch.ones_like(X) * 0.1

def hjb_f(X: torch.Tensor, u: torch.Tensor, 
         sigma_T_grad_u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """HJB 方程的非线性项"""
    # 这里简化处理，实际HJB方程有更复杂的非线性项
    return -torch.norm(sigma_T_grad_u, dim=1, keepdim=True)**2 / 2.0

def hjb_g(X: torch.Tensor) -> torch.Tensor:
    """HJB 方程的终端条件"""
    return torch.sin(torch.norm(X, dim=1, keepdim=True))


def test_hjb_equation():
    """测试 HJB 方程求解"""
    print("\n测试 Hamilton-Jacobi-Bellman 方程求解...")
    
    # 设置参数
    d = 2  # 更高维度
    T = 1.0
    dt = 0.05
    n_trajectories = 200
    n_iterations = 1500
    
    # 创建求解器
    solver = DeepBSDESolver(
        d=d,
        T=T,
        dt=dt,
        n_trajectories=n_trajectories,
        n_iterations=n_iterations,
        hidden_size=30,
        learning_rate=0.0005,
        device='cpu'
    )
    
    # 初始位置
    X0 = torch.ones((1, d)) * 0.5
    
    # 训练
    print("开始训练...")
    loss_history = solver.train(
        X0=X0,
        mu_func=hjb_mu,
        sigma_func=hjb_sigma,
        f_func=hjb_f,
        g_func=hjb_g,
        verbose=True
    )
    
    # 测试不同初始点
    test_points = torch.tensor([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0]
    ], dtype=torch.float32)
    
    predictions = solver.predict(test_points)
    
    print("\nHJB方程预测结果:")
    for i, (point, pred) in enumerate(zip(test_points, predictions)):
        print(f"初始点: {point.numpy()}, 值函数: {pred.item():.6f}")
    
    return solver, loss_history


# 主函数
if __name__ == "__main__":
    print("=" * 60)
    print("深度BSDE求解器 - Python实现")
    print("=" * 60)
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    print("\n1. 测试 Black-Scholes 方程:")
    bs_solver, bs_loss = test_black_scholes()
    
    print("\n" + "=" * 60)
    print("\n2. 测试 Hamilton-Jacobi-Bellman 方程:")
    hjb_solver, hjb_loss = test_hjb_equation()
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
