import torch
import torch.nn as nn
import numpy as np

class FBSNN(nn.Module):
    """
    前向-后向随机微分方程神经网络求解器
    用于求解高维偏微分方程
    """
    def __init__(self, Xi, T, M, N, D, device='cpu'):
        """
        初始化FBSNN求解器
        
        参数:
            Xi: 初始状态，维度为 1 x D
            T: 终止时间
            M: 轨迹数量（批次大小）
            N: 时间步数
            D: 维度
            device: 计算设备
        """
        super().__init__()
        self.Xi = torch.tensor(Xi, dtype=torch.float32, device=device)
        self.T = T
        self.M = M
        self.N = N
        self.D = D
        self.device = device
        
        # 创建简单的神经网络模型
        self.model = nn.Sequential(
            nn.Linear(D + 1, 50),  # 输入维度: 空间D + 时间1
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)       # 输出维度: 1
        ).to(device)
    
    def net_u(self, t, X):
        """
        计算神经网络在给定时间和状态下的输出及其梯度
        
        参数:
            t: 时间批次，维度为 M x 1
            X: 状态批次，维度为 M x D
            
        返回:
            u: 神经网络输出，维度为 M x 1
            Du: 输出关于X的梯度，维度为 M x D
        """
        # 拼接时间和空间变量
        inputs = torch.cat([t, X], dim=1)
        u = self.model(inputs)
        
        # 计算梯度
        Du = torch.autograd.grad(
            outputs=[u],
            inputs=[X],
            grad_outputs=torch.ones_like(u),
            allow_unused=True,
            retain_graph=True,
            create_graph=True
        )[0]
        
        return u, Du
    
    def mu_tf(self, t, X, Y, Z):
        """
        前向随机微分方程的漂移项
        
        参数:
            t: 时间批次，维度为 M x 1
            X: 状态批次，维度为 M x D
            Y: 值函数批次，维度为 M x 1
            Z: 梯度批次，维度为 M x D
            
        返回:
            mu: 漂移项，维度为 M x D
        """
        # 示例实现: 零漂移
        return torch.zeros_like(X)
    
    def sigma_tf(self, t, X, Y):
        """
        前向随机微分方程的扩散项
        
        参数:
            t: 时间批次，维度为 M x 1
            X: 状态批次，维度为 M x D
            Y: 值函数批次，维度为 M x 1
            
        返回:
            sigma: 扩散项，维度为 M x D x D
        """
        # 示例实现: 单位矩阵
        M, D = X.shape
        return torch.eye(D, device=self.device).unsqueeze(0).repeat(M, 1, 1)
    
    def phi_tf(self, t, X, Y, Z):
        """
        后向随机微分方程的漂移项
        
        参数:
            t: 时间批次，维度为 M x 1
            X: 状态批次，维度为 M x D
            Y: 值函数批次，维度为 M x 1
            Z: 梯度批次，维度为 M x D
            
        返回:
            phi: 漂移项，维度为 M x 1
        """
        # 示例实现: 零漂移
        return torch.zeros_like(Y)
    
    def g_tf(self, X):
        """
        终端条件函数
        
        参数:
            X: 状态批次，维度为 M x D
            
        返回:
            g: 终端条件值，维度为 M x 1
        """
        # 示例实现: 计算X的L2范数的平方
        return torch.sum(X**2, dim=1, keepdim=True)
    
    def Dg_tf(self, X):
        """
        计算函数g关于输入X的梯度
        
        参数:
            X: 状态批次，维度为 M x D
            
        返回:
            Dg: 梯度，维度为 M x D
        """
        g = self.g_tf(X)
        Dg = torch.autograd.grad(
            outputs=[g],
            inputs=[X],
            grad_outputs=torch.ones_like(g),
            allow_unused=True,
            retain_graph=True,
            create_graph=True
        )[0]
        return Dg
    
    def loss_function(self, t, W, Xi):
        """
        计算神经网络的损失
        
        参数:
            t: 时间实例的批次，维度为 M x (N+1) x 1
            W: 布朗运动增量的批次，维度为 M x (N+1) x D
            Xi: 初始状态，维度为 1 x D
            
        返回:
            loss: 总损失值
            X: 所有时间步的状态，维度为 M x (N+1) x D
            Y: 所有时间步的网络输出，维度为 M x (N+1) x 1
            Y0_pred: 初始时刻的网络输出，维度为标量
        """
        # 初始化损失为零
        loss = 0
        
        # 用于存储每个时间步的状态
        X_list = []
        
        # 用于存储每个时间步的网络输出
        Y_list = []
        
        # 初始时间和布朗运动增量
        t0 = t[:, 0, :]
        W0 = W[:, 0, :]
        
        # 所有轨迹的初始状态
        X0 = Xi.repeat(self.M, 1).view(self.M, self.D)  # 维度: M x D
        
        # 获取初始状态下的网络输出及其梯度
        Y0, Z0 = self.net_u(t0, X0)
        
        # 存储初始状态和网络输出
        X_list.append(X0)
        Y_list.append(Y0)
        
        # 迭代每个时间步
        for n in range(0, self.N):
            # 下一个时间步和布朗运动增量
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]
            
            # 使用欧拉-丸山方法计算下一个状态
            X1 = X0 + self.mu_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.squeeze(
                torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1)), dim=-1)
            
            # 计算在下一个状态的预测值 (Y1_tilde)
            Y1_tilde = Y0 + self.phi_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.sum(
                Z0 * torch.squeeze(torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1))), 
                dim=1, keepdim=True)
            
            # 获取在下一个状态的网络输出及其梯度
            Y1, Z1 = self.net_u(t1, X1)
            
            # 将Y1和Y1_tilde的平方差添加到损失中
            loss += torch.sum(torch.pow(Y1 - Y1_tilde, 2))
            
            # 更新变量用于下一次迭代
            t0, W0, X0, Y0, Z0 = t1, W1, X1, Y1, Z1
            
            # 存储当前状态和网络输出
            X_list.append(X0)
            Y_list.append(Y0)
        
        # 将终端条件添加到损失中: 网络输出与最终状态目标值之间的差异
        loss += torch.sum(torch.pow(Y1 - self.g_tf(X1), 2))
        
        # 将网络梯度与g在最终状态的梯度之间的差异添加到损失中
        loss += torch.sum(torch.pow(Z1 - self.Dg_tf(X1), 2))
        
        # 将所有时间步的状态和网络输出堆叠起来
        X = torch.stack(X_list, dim=1)
        Y = torch.stack(Y_list, dim=1)
        
        # 返回损失、各时间步的状态和输出
        # 最后返回的元素是网络输出的第一个元素，供参考或进一步使用
        return loss, X, Y, Y[0, 0, 0]


def test_loss_function():
    """
    测试损失函数的计算
    """
    # 设置随机种子以确保结果可重复
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=" * 60)
    print("FBSDE损失函数测试")
    print("=" * 60)
    
    # 设置参数
    D = 3           # 维度
    M = 5           # 轨迹数量
    N = 4           # 时间步数
    T = 1.0         # 终止时间
    
    # 创建初始状态
    Xi = torch.randn(1, D)
    
    # 创建模型
    model = FBSNN(
        Xi=Xi.numpy(),
        T=T,
        M=M,
        N=N,
        D=D,
        device='cpu'
    )
    
    # 生成测试数据
    dt = T / N
    
    # 生成时间批次
    t_batch = torch.zeros(M, N + 1, 1)
    for i in range(N + 1):
        t_batch[:, i, 0] = i * dt
    
    # 生成布朗运动增量
    W_batch = torch.zeros(M, N + 1, D)
    for i in range(1, N + 1):
        W_batch[:, i, :] = torch.randn(M, D) * np.sqrt(dt)
    W_batch = torch.cumsum(W_batch, dim=1)
    
    # 计算损失
    loss, X_states, Y_outputs, Y0_pred = model.loss_function(t_batch, W_batch, Xi)
    
    # 打印结果
    print(f"\n测试参数:")
    print(f"  维度 D: {D}")
    print(f"  轨迹数 M: {M}")
    print(f"  时间步数 N: {N}")
    print(f"  终止时间 T: {T}")
    
    print(f"\n计算结果:")
    print(f"  损失值: {loss.item():.6f}")
    print(f"  初始预测值 Y0: {Y0_pred.item():.6f}")
    print(f"  状态张量形状: {X_states.shape}")
    print(f"  输出张量形状: {Y_outputs.shape}")
    
    # 验证输出一致性
    print(f"\n验证:")
    print(f"  时间批次形状: {t_batch.shape}")
    print(f"  布朗运动批次形状: {W_batch.shape}")
    print(f"  初始状态形状: {Xi.shape}")
    
    # 验证损失计算是否正确
    if not torch.isnan(loss) and not torch.isinf(loss):
        print("✓ 损失计算成功完成")
    else:
        print("✗ 损失计算失败")
    
    return model, loss, X_states, Y_outputs, Y0_pred


def test_with_different_dimensions():
    """
    测试不同维度下的损失函数计算
    """
    print("\n" + "=" * 60)
    print("不同维度测试")
    print("=" * 60)
    
    test_cases = [
        {"D": 1, "M": 2, "N": 2},
        {"D": 2, "M": 3, "N": 3},
        {"D": 5, "M": 4, "N": 4},
        {"D": 10, "M": 5, "N": 5},
    ]
    
    T = 1.0
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}:")
        print(f"  维度 D: {case['D']}")
        print(f"  轨迹数 M: {case['M']}")
        print(f"  时间步数 N: {case['N']}")
        
        # 创建初始状态
        Xi = torch.randn(1, case['D'])
        
        # 创建模型
        model = FBSNN(
            Xi=Xi.numpy(),
            T=T,
            M=case['M'],
            N=case['N'],
            D=case['D'],
            device='cpu'
        )
        
        # 生成测试数据
        dt = T / case['N']
        
        t_batch = torch.zeros(case['M'], case['N'] + 1, 1)
        for j in range(case['N'] + 1):
            t_batch[:, j, 0] = j * dt
        
        W_batch = torch.zeros(case['M'], case['N'] + 1, case['D'])
        for j in range(1, case['N'] + 1):
            W_batch[:, j, :] = torch.randn(case['M'], case['D']) * np.sqrt(dt)
        W_batch = torch.cumsum(W_batch, dim=1)
        
        # 计算损失
        loss, X_states, Y_outputs, Y0_pred = model.loss_function(t_batch, W_batch, Xi)
        
        print(f"  损失值: {loss.item():.6f}")
        print(f"  初始预测值: {Y0_pred.item():.6f}")
        
        if not torch.isnan(loss) and not torch.isinf(loss):
            print("  ✓ 计算成功")
        else:
            print("  ✗ 计算失败")


if __name__ == "__main__":
    # 运行主测试
    model, loss, X_states, Y_outputs, Y0_pred = test_loss_function()
    
    # 运行不同维度测试
    test_with_different_dimensions()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
