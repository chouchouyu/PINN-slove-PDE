import torch
import torch.nn as nn
import numpy as np

class GradientNetwork(nn.Module):
    """
    用于计算神经网络输出及其对输入状态X的梯度的类
    这个类专门用于物理信息神经网络（PINN）中求解偏微分方程
    """
    def __init__(self, input_dim, hidden_dims, output_dim=1):
        """
        初始化神经网络模型
        
        参数:
            input_dim: 输入维度（包括时间和空间变量）
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度，默认为1
        """
        super(GradientNetwork, self).__init__()
        
        # 构建网络层
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Tanh())
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def net_u(self, t, X):
        """
        计算神经网络的输出及其对输入状态X的梯度
        
        参数:
            t: 时间实例的批次，维度为 M x 1
            X: 状态变量的批次，维度为 M x D
        
        返回:
            u: 神经网络在每组输入(t, X)处的值函数，维度为 M x 1
            Du: 输出u关于状态变量X的梯度，维度为 M x D
        """
        # 沿第二维拼接时间和状态变量，形成神经网络的输入
        input_tensor = torch.cat((t, X), 1)
        
        # 将拼接后的输入传入神经网络模型
        # 输出u是一个维度为M x 1的张量，表示每组输入(t, X)处的值函数
        u = self.model(input_tensor)
        
        # 计算输出u关于状态变量X的梯度
        # 梯度是针对批次中的每个输入计算的，结果是一个维度为M x D的张量
        Du = torch.autograd.grad(
            outputs=[u], 
            inputs=[X], 
            grad_outputs=torch.ones_like(u), 
            allow_unused=True, 
            retain_graph=True, 
            create_graph=True
        )[0]
        
        return u, Du
    
    def compute_second_derivative(self, t, X):
        """
        计算二阶导数（Hessian矩阵的对角线元素）
        
        参数:
            t: 时间实例的批次，维度为 M x 1
            X: 状态变量的批次，维度为 M x D
        
        返回:
            D2u: 输出u关于状态变量X的二阶导数，维度为 M x D
        """
        # 先计算一阶导数
        u, Du = self.net_u(t, X)
        
        # 计算二阶导数
        D2u = []
        for i in range(X.shape[1]):
            # 对每个维度计算二阶偏导
            grad_i = torch.autograd.grad(
                outputs=Du[:, i], 
                inputs=[X], 
                grad_outputs=torch.ones_like(Du[:, i]), 
                retain_graph=(i < X.shape[1] - 1), 
                create_graph=False
            )[0][:, i]
            D2u.append(grad_i.unsqueeze(1))
        
        D2u = torch.cat(D2u, dim=1)
        return u, Du, D2u

def test_gradient_calculation():
    """
    测试梯度计算功能的函数
    """
    # 设置随机种子以确保结果可重复
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=" * 60)
    print("神经网络梯度计算测试")
    print("=" * 60)
    
    # 创建模型实例
    # 输入维度: 1(时间) + 3(空间) = 4
    # 隐藏层: [20, 20, 20]
    # 输出维度: 1
    model = GradientNetwork(input_dim=4, hidden_dims=[20, 20, 20], output_dim=1)
    
    # 生成测试数据
    batch_size = 5
    spatial_dim = 3
    
    # 创建时间批次: M x 1
    t_batch = torch.randn(batch_size, 1, requires_grad=True)
    
    # 创建空间状态批次: M x D
    X_batch = torch.randn(batch_size, spatial_dim, requires_grad=True)
    
    print(f"\n测试数据形状:")
    print(f"时间批次 t: {t_batch.shape}")
    print(f"空间批次 X: {X_batch.shape}")
    
    # 计算一阶导数
    print(f"\n计算一阶导数...")
    u, Du = model.net_u(t_batch, X_batch)
    
    print(f"\n结果形状:")
    print(f"值函数 u: {u.shape}")
    print(f"梯度 Du: {Du.shape}")
    
    # 验证梯度计算
    print(f"\n验证梯度计算:")
    print(f"u的前3个值: {u[:3].detach().numpy().flatten()}")
    print(f"Du的范数: {torch.norm(Du).item():.6f}")
    print(f"Du的平均值: {torch.mean(Du).item():.6f}")
    print(f"Du的标准差: {torch.std(Du).item():.6f}")
    
    # 计算二阶导数
    print(f"\n计算二阶导数...")
    u2, Du2, D2u = model.compute_second_derivative(t_batch, X_batch)
    
    print(f"\n二阶导数结果形状:")
    print(f"值函数 u: {u2.shape}")
    print(f"一阶梯度 Du: {Du2.shape}")
    print(f"二阶导数 D2u: {D2u.shape}")
    
    # 验证二阶导数计算
    print(f"\n验证二阶导数计算:")
    print(f"D2u的范数: {torch.norm(D2u).item():.6f}")
    print(f"D2u的平均值: {torch.mean(D2u).item():.6f}")
    print(f"D2u的标准差: {torch.std(D2u).item():.6f}")
    
    # 检查一致性
    print(f"\n检查一致性:")
    u_diff = torch.norm(u - u2).item()
    Du_diff = torch.norm(Du - Du2).item()
    print(f"两次计算u的差异: {u_diff:.6f}")
    print(f"两次计算Du的差异: {Du_diff:.6f}")
    
    if u_diff < 1e-6 and Du_diff < 1e-6:
        print("✓ 梯度计算一致性检查通过")
    else:
        print("✗ 梯度计算存在不一致")
    
    return model, t_batch, X_batch, u, Du, D2u

def test_with_different_dimensions():
    """
    测试不同维度下的梯度计算
    """
    print("\n" + "=" * 60)
    print("不同维度测试")
    print("=" * 60)
    
    test_cases = [
        {"batch_size": 2, "spatial_dim": 1, "hidden_dims": [10]},
        {"batch_size": 3, "spatial_dim": 2, "hidden_dims": [10, 10]},
        {"batch_size": 4, "spatial_dim": 5, "hidden_dims": [20, 20, 20]},
        {"batch_size": 5, "spatial_dim": 10, "hidden_dims": [30, 30, 30, 30]},
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}:")
        print(f"  批次大小: {case['batch_size']}")
        print(f"  空间维度: {case['spatial_dim']}")
        print(f"  隐藏层: {case['hidden_dims']}")
        
        # 创建模型
        input_dim = 1 + case['spatial_dim']  # 时间 + 空间
        model = GradientNetwork(
            input_dim=input_dim, 
            hidden_dims=case['hidden_dims'], 
            output_dim=1
        )
        
        # 生成测试数据
        t = torch.randn(case['batch_size'], 1, requires_grad=True)
        X = torch.randn(case['batch_size'], case['spatial_dim'], requires_grad=True)
        
        # 计算梯度
        u, Du = model.net_u(t, X)
        
        print(f"  输出形状: u={u.shape}, Du={Du.shape}")
        
        # 检查梯度是否存在
        if Du is not None:
            print(f"  梯度计算成功，Du范数: {torch.norm(Du).item():.6f}")
        else:
            print(f"  梯度计算失败，Du为None")

if __name__ == "__main__":
    # 运行主测试
    model, t_batch, X_batch, u, Du, D2u = test_gradient_calculation()
    
    # 运行不同维度测试
    test_with_different_dimensions()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
