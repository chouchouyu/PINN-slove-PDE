import torch
import torch.nn as nn
import torch.nn.functional as F

# https://www.youtube.com/watch?v=o1gp_lfIrdk
# Spectral Normalization For GAN @GAN

class NAISNetProjectionLayer:
    """
    NAIS-Net投影层的简化实现，展示代码中的数学运算逻辑
    """
    def __init__(self, epsilon=0.01):
        """
        初始化投影层参数
        
        参数:
            epsilon: 正则化参数，控制单位矩阵的缩放程度
        """
        self.epsilon = epsilon
    
    def project(self, layer, out):
        """
        对线性层进行投影变换
        对权重矩阵的Gram矩阵进行约束，可以控制权重列向量之间的相关性，避免过于冗余或奇异的权重。
        参数:
            layer: nn.Linear层，假设输入输出维度相同
            out: 输入张量
            
        返回:
            变换后的输出张量
        """
        # 1. 获取权重矩阵 W ∈ R^{n×n} (假设方阵)
        weights = layer.weight  # shape: (out_features, in_features)
        
        # 2. 计算 delta 值，用于约束矩阵范数
        # delta = 1 - 2ε，注意，如果epsilon是正数，那么delta会小于1。
        delta = 1 - 2 * self.epsilon
        
        # 3. 计算权重矩阵的Gram矩阵: R^T R
        # 这里 R = weights，计算 W^T W 权重列向量之间的内积关系。
        RtR = torch.matmul(weights.t(), weights)  # 方阵 shape: (in_features, in_features)
        
        # 4. 计算Gram矩阵的Frobenius范数
        norm = torch.norm(RtR)  # 默认计算Frobenius范数: sqrt(Σ|a_ij|²)
        
        # 5. 范数约束：如果范数超过阈值，进行缩放
        # 确保缩放后的矩阵范数为 √δ
        if norm > delta:
            # 计算缩放因子: c = √δ / √norm
            # 使得缩放后的矩阵范数变为 √δ
            scaling_factor = (delta ** 0.5) / (norm ** 0.5)
            RtR = scaling_factor * RtR
        
        # 6. 构建正则化矩阵: A = R^T R + εI
        # 添加单位矩阵的 ε 倍，确保矩阵正定性
        identity = torch.eye(RtR.shape[0], device=RtR.device)  # 单位矩阵
        A = RtR + identity * self.epsilon
        
        # 7. 应用线性变换: output = -A * out + bias
        # 使用负的A矩阵，创建近似反对称结构
        # 另外，F.linear是PyTorch中的函数，它执行线性变换：y = x @ weight.t() + bias。
        # 但这里传入的权重是-A，而A的形状是(in_features, in_features)，注意F.linear要求权重的形状是(out_features, in_features)。
        # 这里out的输入形状可能是(batch_size, in_features)，那么-A的形状是(in_features, in_features)，所以输出的形状是(batch_size, in_features)。
        # 因此，这个线性变换实际上是将输入从in_features维映射到in_features维，而不是改变维度。
        # 这更像是一个特征变换。
        # layer是nn.Linear(in_features, out_features)，
        # 那么layer.weight是(out_features, in_features)，
        # layer.bias是(out_features,)。
        # 而A是(in_features, in_features)。
        return F.linear(out, -A, layer.bias)


# 测试代码
def test_projection_layer():
    """测试投影层的功能"""
    # 设置随机种子确保可重复性
    torch.manual_seed(42)
    
    # 创建测试数据
    batch_size = 4
    feature_size = 5  # 假设输入输出维度相同
    
    # 创建一个线性层
    linear_layer = nn.Linear(feature_size, feature_size, bias=True)
    
    # 创建投影层实例
    projector = NAISNetProjectionLayer(epsilon=0.01)
    
    # 创建输入张量
    input_tensor = torch.randn(batch_size, feature_size)
    
    # 应用投影变换
    output = projector.project(linear_layer, input_tensor)
    
    # 验证输出形状
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
    print(f"线性层权重形状: {linear_layer.weight.shape}")
    print(f"线性层偏置形状: {linear_layer.bias.shape}")
    
    # 验证数学性质
    print(f"\n验证数学性质:")
    
    # 计算变换矩阵 A
    weights = linear_layer.weight
    RtR = torch.matmul(weights.t(), weights)
    norm = torch.norm(RtR)
    delta = 1 - 2 * 0.01
    
    if norm > delta:
        scaling_factor = (delta ** 0.5) / (norm ** 0.5)
        RtR = scaling_factor * RtR
    
    A = RtR + torch.eye(feature_size) * 0.01
    
    # 验证 F.linear 的等价计算
    expected_output = torch.matmul(input_tensor, -A.t()) + linear_layer.bias
    
    # 比较结果
    tolerance = 1e-6
    if torch.allclose(output, expected_output, rtol=tolerance):
        print("✓ 投影变换计算正确")
    else:
        print("✗ 计算结果不一致")
        diff = torch.abs(output - expected_output).max()
        print(f"最大差异: {diff.item()}")


if __name__ == "__main__":
    test_projection_layer()
