import torch
import torch.nn as nn
import numpy as np
import random

def set_seed(seed=42):
    """设置所有随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"已设置随机种子: {seed}")

class SimpleModel(nn.Module):
    """简单的神经网络模型"""
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def test_without_seed():
    """不设置随机种子的测试"""
    print("\n=== 测试1: 不设置随机种子 ===")
    
    # 创建模型
    model1 = SimpleModel()
    
    # 创建输入数据
    x = torch.randn(1, 10)
    
    # 前向传播
    output1 = model1(x)
    print(f"第一次运行结果: {output1}")
    
    # 重新创建模型（权重会重新随机初始化）
    model2 = SimpleModel()
    output2 = model2(x)
    print(f"第二次运行结果: {output2}")
    
    # 比较结果
    if torch.allclose(output1, output2, rtol=1e-3):
        print("两次结果相同（概率极低）")
    else:
        print("两次结果不同（正常情况）")

def test_with_seed():
    """设置随机种子的测试"""
    print("\n=== 测试2: 设置随机种子 ===")
    
    # 第一次运行
    set_seed(42)  # 设置随机种子
    model1 = SimpleModel()
    x = torch.randn(1, 10)
    output1 = model1(x)
    print(f"第一次运行结果: {output1}")
    
    # 第二次运行
    set_seed(42)  # 设置相同的随机种子
    model2 = SimpleModel()
    output2 = model2(x)
    print(f"第二次运行结果: {output2}")
    
    # 比较结果
    if torch.allclose(output1, output2, rtol=1e-6):
        print("✓ 两次结果完全相同（随机种子生效）")
    else:
        print("✗ 两次结果不同")

def test_across_devices():
    """测试不同设备/环境下的结果一致性"""
    print("\n=== 测试3: 跨环境一致性验证 ===")
    
    # 设置随机种子
    set_seed(42)
    
    # 创建模型和输入
    model = SimpleModel()
    x = torch.randn(1, 10)
    
    # 获取权重以便验证
    print("\n模型权重（前5个值）:")
    print(f"layer1.weight[0, :5]: {model.layer1.weight[0, :5]}")
    print(f"layer1.bias[:5]: {model.layer1.bias[:5]}")
    print(f"layer2.weight[0, :5]: {model.layer2.weight[0, :5]}")
    print(f"layer2.bias: {model.layer2.bias}")
    
    # 前向传播
    output = model(x)
    print(f"\n模型输出: {output}")
    
    # 手动计算验证
    with torch.no_grad():
        # 第一层
        h = x @ model.layer1.weight.T + model.layer1.bias
        h_relu = torch.relu(h)
        # 第二层
        output_manual = h_relu @ model.layer2.weight.T + model.layer2.bias
        
        print(f"\n手动计算结果: {output_manual}")
        
        if torch.allclose(output, output_manual, rtol=1e-6):
            print("✓ 自动计算与手动计算结果一致")
        else:
            print("✗ 计算结果不一致")

def main():
    """主函数"""
    print("神经网络结果可复现性测试")
    print("=" * 50)
    
    # 测试不同情况
    test_without_seed()
    test_with_seed()
    test_across_devices()
    
    print("\n" + "=" * 50)
    print("总结:")
    print("1. 不设置随机种子时，每次运行结果都可能不同")
    print("2. 设置相同的随机种子可以确保结果可复现")
    print("3. 不同设备/环境下，只要随机种子相同，结果应该一致")
    print("4. 您的差异可能是由于：")
    print("   - 未设置随机种子")
    print("   - PyTorch/CUDA版本不同")
    print("   - 硬件差异（CPU/GPU）")

if __name__ == "__main__":
    main()
