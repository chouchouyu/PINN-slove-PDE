import torch
import torch.nn as nn
import numpy as np

class ExampleNetwork(nn.Module):
    """
    示例神经网络，用于展示梯度计算
    """
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def g_tf(self, X):
        """
        计算终端条件函数g(X)
        
        参数:
            X: 状态变量的批次，维度为 M x D
            
        返回:
            g: 终端条件函数值，维度为 M x 1
        """
        # 示例：计算向量各元素的平方和
        return torch.sum(X**2, dim=1, keepdim=True)
    
    def Dg_tf(self, X):
        """
        计算函数g关于输入X的梯度
        
        参数:
            X: 状态变量的批次，维度为 M x D
            
        返回:
            Dg: 函数g关于输入X的梯度，维度为 M x D
        
        详细说明:
            1. 首先调用g_tf方法计算函数g在输入X处的值
            2. 使用torch.autograd.grad计算g关于X的梯度
            3. grad_outputs参数设置为与g形状相同的全1张量，表示对g的所有分量平等对待
            4. allow_unused=True允许输入变量未在计算中被使用的情况
            5. retain_graph=True保留计算图，允许后续的梯度计算
            6. create_graph=True创建梯度计算图，使返回的梯度本身可继续求导
        """
        # 计算函数g在输入X处的值
        g = self.g_tf(X)  # 维度: M x 1
        
        # 计算g关于输入X的梯度
        # 梯度是针对批次中的每个输入计算的，结果是一个维度为M x D的张量
        Dg = torch.autograd.grad(
            outputs=[g], 
            inputs=[X], 
            grad_outputs=torch.ones_like(g), 
            allow_unused=True, 
            retain_graph=True, 
            create_graph=True
        )[0]
        
        return Dg
    
    def compute_second_derivative(self, X):
        """
        计算函数g的二阶导数（Hessian矩阵的对角线元素）
        
        参数:
            X: 状态变量的批次，维度为 M x D
            
        返回:
            D2g: 函数g关于输入X的二阶导数，维度为 M x D
        """
        # 先计算一阶导数
        g = self.g_tf(X)
        Dg = self.Dg_tf(X)
        
        # 计算二阶导数
        D2g_list = []
        for i in range(X.shape[1]):
            # 对每个维度计算二阶偏导
            grad_i = torch.autograd.grad(
                outputs=Dg[:, i], 
                inputs=[X], 
                grad_outputs=torch.ones_like(Dg[:, i]), 
                retain_graph=(i < X.shape[1] - 1), 
                create_graph=False
            )[0][:, i]
            D2g_list.append(grad_i.unsqueeze(1))
        
        D2g = torch.cat(D2g_list, dim=1)
        return g, Dg, D2g

def test_gradient_calculations():
    """
    测试梯度计算功能的函数
    """
    # 设置随机种子确保结果可重复
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=" * 60)
    print("梯度计算功能测试")
    print("=" * 60)
    
    # 创建模型实例
    model = ExampleNetwork(input_dim=5, hidden_dim=10, output_dim=1)
    
    # 生成测试数据
    batch_size = 3
    input_dim = 5
    
    # 创建输入张量，并设置requires_grad=True以便计算梯度
    X = torch.randn(batch_size, input_dim, requires_grad=True)
    print(f"输入X的形状: {X.shape}")
    
    # 测试g_tf函数
    print(f"\n测试g_tf函数:")
    g_value = model.g_tf(X)
    print(f"g(X)的值: {g_value.detach().numpy().flatten()}")
    
    # 测试Dg_tf函数
    print(f"\n测试Dg_tf函数:")
    Dg_value = model.Dg_tf(X)
    print(f"Dg的形状: {Dg_value.shape}")
    print(f"Dg的范数: {torch.norm(Dg_value).item():.6f}")
    print(f"Dg的平均值: {torch.mean(Dg_value).item():.6f}")
    
    # 测试二阶导数计算
    print(f"\n测试二阶导数计算:")
    g_val, Dg_val, D2g_val = model.compute_second_derivative(X)
    print(f"D2g的形状: {D2g_val.shape}")
    print(f"D2g的范数: {torch.norm(D2g_val).item():.6f}")
    
    # 验证梯度计算的正确性
    print(f"\n验证梯度计算正确性:")
    
    # 使用有限差分法验证梯度
    def finite_difference_gradient(X, model, epsilon=1e-4):
        """使用中心差分法计算数值梯度"""
        n = X.shape[1]
        grad_numerical = torch.zeros_like(X)
        
        for i in range(n):
            X_plus = X.clone()
            X_minus = X.clone()
            X_plus[:, i] += epsilon
            X_minus[:, i] -= epsilon
            
            g_plus = model.g_tf(X_plus)
            g_minus = model.g_tf(X_minus)
            
            grad_numerical[:, i] = (g_plus - g_minus).squeeze() / (2 * epsilon)
        
        return grad_numerical
    
    # 计算数值梯度
    grad_numerical = finite_difference_gradient(X, model)
    
    # 比较自动微分和数值梯度的差异
    max_diff = torch.max(torch.abs(Dg_value - grad_numerical)).item()
    mean_diff = torch.mean(torch.abs(Dg_value - grad_numerical)).item()
    
    print(f"最大差异: {max_diff:.6e}")
    print(f"平均差异: {mean_diff:.6e}")
    
    if max_diff < 1e-4:
        print("✓ 梯度计算正确")
    else:
        print("⚠ 梯度计算可能存在误差")
    
    return model, X, g_value, Dg_value, D2g_val

if __name__ == "__main__":
    # 运行测试
    model, X, g, Dg, D2g = test_gradient_calculations()
    
    print(f"\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
