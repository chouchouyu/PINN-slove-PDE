import torch
import torch.nn as nn
import numpy as np

class FBSDESolverExample:
    """
    前向-后向随机微分方程求解器示例
    展示如何从单个初始状态生成多个轨迹的初始状态
    """
    def __init__(self, Xi, M, D, device='cpu'):
        """
        初始化FBSDE求解器示例
        
        参数:
            Xi: 初始状态张量，形状为 1 x D
            M: 轨迹数量（批次大小）
            D: 状态维度
            device: 计算设备
        """
        # 确保Xi是张量
        if not isinstance(Xi, torch.Tensor):
            Xi = torch.tensor(Xi, dtype=torch.float32, device=device)
        
        # 检查Xi的形状
        if Xi.dim() == 1:
            Xi = Xi.unsqueeze(0)  # 如果是1D，转换为2D
        
        self.Xi = Xi
        self.M = M
        self.D = D
        self.device = device
        
        print(f"初始状态Xi的形状: {self.Xi.shape}")
        print(f"轨迹数量M: {self.M}")
        print(f"状态维度D: {self.D}")
    
    def expand_initial_state(self):
        """
        扩展初始状态以生成多个轨迹
        
        返回:
            X0: 扩展后的初始状态，形状为 M x D
        """
        # 关键代码行：将初始状态Xi重复M次，然后重塑为(M, D)形状
        # repeat(1) 表示在第一个维度（批量维度）不重复
        # repeat(self.M, 1) 表示在第二个维度重复M次
        # view(self.M, self.D) 将结果重塑为M行D列
        X0 = self.Xi.repeat(self.M, 1).view(self.M, self.D)
        
        return X0
    
    def simulate_trajectories(self, num_steps=10, dt=0.1):
        """
        模拟多个轨迹的前向传播
        
        参数:
            num_steps: 时间步数
            dt: 时间步长
            
        返回:
            trajectories: 所有轨迹的模拟结果，形状为 (num_steps+1, M, D)
        """
        # 扩展初始状态
        X0 = self.expand_initial_state()
        
        # 存储所有时间步的状态
        trajectories = torch.zeros(num_steps + 1, self.M, self.D, device=self.device)
        trajectories[0] = X0
        
        # 简单的前向欧拉模拟：dX = μ dt + σ dW
        current_X = X0
        for step in range(1, num_steps + 1):
            # 漂移项（示例：零漂移）
            drift = torch.zeros_like(current_X)
            
            # 扩散项（示例：单位扩散）
            diffusion = torch.ones_like(current_X)
            
            # 布朗运动增量
            dW = torch.randn(self.M, self.D, device=self.device) * np.sqrt(dt)
            
            # 欧拉离散化
            current_X = current_X + drift * dt + diffusion * dW
            
            # 存储当前状态
            trajectories[step] = current_X
        
        return trajectories
    
    def test_initial_state_expansion(self):
        """
        测试初始状态扩展功能
        """
        print("\n" + "="*60)
        print("初始状态扩展测试")
        print("="*60)
        
        # 方法1: 使用expand_initial_state方法
        X0 = self.expand_initial_state()
        print(f"\n扩展后的初始状态形状: {X0.shape}")
        print(f"期望形状: ({self.M}, {self.D})")
        
        # 验证形状
        if X0.shape == (self.M, self.D):
            print("✓ 形状正确")
        else:
            print(f"✗ 形状错误: 期望({self.M}, {self.D})，实际{X0.shape}")
        
        # 验证内容
        print("\n验证内容一致性:")
        print(f"原始Xi: {self.Xi.flatten()[:5].tolist()}... (显示前5个元素)")
        print(f"扩展后X0的第一行: {X0[0, :5].tolist()}... (显示前5个元素)")
        
        # 检查所有行是否相同
        is_consistent = torch.allclose(X0[0], X0[1], rtol=1e-6)
        if is_consistent:
            print("✓ 所有轨迹的初始状态相同")
        else:
            print("✗ 不同轨迹的初始状态不同")
        
        # 方法2: 手动计算验证
        print("\n手动计算验证:")
        # 使用repeat然后view
        manual_X0 = self.Xi.repeat(self.M, 1).view(self.M, self.D)
        
        # 使用expand和reshape
        alt_X0 = self.Xi.expand(self.M, -1).reshape(self.M, self.D)
        
        # 比较两种方法的结果
        if torch.allclose(manual_X0, alt_X0, rtol=1e-6):
            print("✓ 两种扩展方法结果一致")
        else:
            print("✗ 两种扩展方法结果不同")
        
        return X0
    
    def visualize_trajectories(self, trajectories, num_trajectories=5):
        """
        可视化前几个轨迹
        
        参数:
            trajectories: 轨迹数据，形状为 (num_steps+1, M, D)
            num_trajectories: 要显示的最大轨迹数量
        """
        import matplotlib.pyplot as plt
        
        num_steps = trajectories.shape[0] - 1
        time_points = np.linspace(0, num_steps * 0.1, num_steps + 1)
        
        # 只显示前几个轨迹
        num_to_show = min(num_trajectories, self.M)
        
        plt.figure(figsize=(10, 6))
        
        for i in range(num_to_show):
            # 只显示第一个维度
            trajectory = trajectories[:, i, 0].cpu().numpy()
            plt.plot(time_points, trajectory, label=f'轨迹 {i+1}', alpha=0.7)
        
        plt.xlabel('时间')
        plt.ylabel('状态值')
        plt.title(f'FBSDE轨迹模拟 (显示前{num_to_show}个轨迹)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def main():
    """
    主函数：演示FBSDE初始状态扩展的使用
    """
    # 设置随机种子以确保结果可重复
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("FBSDE初始状态扩展演示")
    print("="*60)
    
    # 测试用例1: 简单2D状态
    print("\n测试用例1: 2D状态，5个轨迹")
    Xi_1d = torch.tensor([[1.0, 2.0]])  # 1 x 2
    solver_1d = FBSDESolverExample(Xi=Xi_1d, M=5, D=2)
    X0_1d = solver_1d.test_initial_state_expansion()
    
    # 测试用例2: 更高维度状态
    print("\n\n测试用例2: 5D状态，3个轨迹")
    Xi_5d = torch.randn(1, 5)  # 1 x 5
    solver_5d = FBSDESolverExample(Xi=Xi_5d, M=3, D=5)
    X0_5d = solver_5d.test_initial_state_expansion()
    
    # 测试用例3: 模拟轨迹
    print("\n\n测试用例3: 轨迹模拟")
    Xi_sim = torch.tensor([[0.0]])  # 1 x 1
    solver_sim = FBSDESolverExample(Xi=Xi_sim, M=10, D=1)
    
    # 模拟轨迹
    trajectories = solver_sim.simulate_trajectories(num_steps=20, dt=0.1)
    print(f"\n模拟轨迹形状: {trajectories.shape}")
    print(f"时间步数: {trajectories.shape[0] - 1}")
    print(f"轨迹数量: {trajectories.shape[1]}")
    print(f"状态维度: {trajectories.shape[2]}")
    
    # 可视化轨迹
    try:
        solver_sim.visualize_trajectories(trajectories, num_trajectories=5)
    except ImportError:
        print("\n注意: matplotlib未安装，无法可视化轨迹")
        print("可以使用 pip install matplotlib 安装")
    
    # 演示代码行的不同写法
    print("\n" + "="*60)
    print("代码行不同写法对比")
    print("="*60)
    
    Xi = torch.tensor([[1.0, 2.0, 3.0]])  # 1 x 3
    M, D = 4, 3
    
    # 方法1: 原始写法
    X0_method1 = Xi.repeat(M, 1).view(M, D)
    print(f"\n方法1 (原始): Xi.repeat({M}, 1).view({M}, {D})")
    print(f"形状: {X0_method1.shape}")
    
    # 方法2: 使用expand
    X0_method2 = Xi.expand(M, -1).reshape(M, D)
    print(f"\n方法2 (expand): Xi.expand({M}, -1).reshape({M}, {D})")
    print(f"形状: {X0_method2.shape}")
    
    # 方法3: 使用repeat_interleave
    X0_method3 = Xi.repeat_interleave(M, dim=0)
    print(f"\n方法3 (repeat_interleave): Xi.repeat_interleave({M}, dim=0)")
    print(f"形状: {X0_method3.shape}")
    
    # 比较结果
    if torch.allclose(X0_method1, X0_method2, rtol=1e-6) and \
       torch.allclose(X0_method1, X0_method3, rtol=1e-6):
        print("\n✓ 所有方法结果一致")
    else:
        print("\n✗ 方法结果不一致")
    
    return solver_1d, solver_5d, solver_sim


if __name__ == "__main__":
    main()
