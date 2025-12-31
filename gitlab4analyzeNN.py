"""
使用self.modules()进行权重初始化的完整示例
演示如何正确初始化神经网络中的所有线性层
包括简单网络和复杂网络结构
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

# 示例1: 简单的多层感知机，使用self.modules()初始化
class SimpleMLP(nn.Module):
    """简单的多层感知机，使用self.modules()进行初始化"""
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, 
                 activation: nn.Module = nn.ReLU()):
        super(SimpleMLP, self).__init__()
        
        # 存储参数
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        
        # 创建网络层
        layers = []
        prev_size = input_size
        
        # 添加隐藏层
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation)
            prev_size = hidden_size
        
        # 添加输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        # 创建网络
        self.network = nn.Sequential(*layers)
        
        # 使用self.modules()初始化所有权重
        self._init_weights_using_modules()
    
    def _init_weights_using_modules(self):
        """使用self.modules()遍历所有模块并初始化权重"""
        print("正在使用self.modules()初始化权重...")
        initialized_count = 0
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 根据激活函数类型选择初始化方法
                if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.ELU)):
                    # 使用Kaiming初始化（针对ReLU族激活函数）
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    print(f"  初始化线性层: {module.weight.shape} -> Kaiming初始化")
                elif isinstance(self.activation, (nn.Tanh, nn.Sigmoid)):
                    # 使用Xavier初始化（针对tanh/sigmoid激活函数）
                    nn.init.xavier_uniform_(module.weight)
                    print(f"  初始化线性层: {module.weight.shape} -> Xavier初始化")
                else:
                    # 默认使用Xavier初始化
                    nn.init.xavier_uniform_(module.weight)
                    print(f"  初始化线性层: {module.weight.shape} -> 默认Xavier初始化")
                
                # 初始化偏置
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                
                initialized_count += 1
        
        print(f"总共初始化了 {initialized_count} 个线性层")
    
    def forward(self, x):
        return self.network(x)

# 示例2: 复杂的嵌套网络结构
class ComplexNestedNet(nn.Module):
    """复杂的嵌套网络结构，包含多个子模块"""
    def __init__(self, input_size: int, num_blocks: int = 3):
        super(ComplexNestedNet, self).__init__()
        
        self.input_size = input_size
        
        # 创建多个残差块
        self.residual_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.residual_blocks.append(ResidualBlock(input_size if i == 0 else 64))
        
        # 创建多个并行分支
        self.branches = nn.ModuleList()
        for i in range(2):
            self.branches.append(BranchModule(64))
        
        # 最后的分类头
        self.classifier = ClassificationHead(64, 10)
        
        # 使用self.modules()初始化所有权重
        self._init_all_weights()
    
    def _init_all_weights(self):
        """初始化网络中的所有线性层权重"""
        print("\n初始化ComplexNestedNet中的所有线性层...")
        linear_count = 0
        conv_count = 0
        batch_norm_count = 0
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 初始化线性层
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                linear_count += 1
                print(f"  找到并初始化线性层: {module.weight.shape}")
            
            elif isinstance(module, nn.Conv2d):
                # 初始化卷积层
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                conv_count += 0
                print(f"  找到并初始化卷积层: {module.weight.shape}")
            
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                # 初始化批归一化层
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                batch_norm_count += 0
        
        print(f"总共初始化了: {linear_count} 个线性层, {conv_count} 个卷积层, {batch_norm_count} 个批归一化层")
    
    def forward(self, x):
        for block in self.residual_blocks:
            x = block(x)
        
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        x = torch.mean(torch.stack(branch_outputs), dim=0)
        return self.classifier(x)

class ResidualBlock(nn.Module):
    """残差块，包含两个线性层和跳跃连接"""
    def __init__(self, in_features: int, out_features: int = 64):
        super(ResidualBlock, self).__init__()
        
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        # 如果输入输出维度不匹配，需要投影
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.linear2(out)
        out = self.bn2(out)
        
        identity = self.shortcut(identity)
        out += identity
        out = self.relu(out)
        
        return out

class BranchModule(nn.Module):
    """分支模块，包含多个子层"""
    def __init__(self, in_features: int):
        super(BranchModule, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 64)
        )
    
    def forward(self, x):
        return self.layers(x)

class ClassificationHead(nn.Module):
    """分类头"""
    def __init__(self, in_features: int, num_classes: int):
        super(ClassificationHead, self).__init__()
        
        self.fc1 = nn.Linear(in_features, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 示例3: 带有自定义初始化策略的网络
class CustomInitializedNet(nn.Module):
    """带有自定义初始化策略的网络"""
    def __init__(self, layer_sizes: List[int], activation_type: str = "relu"):
        super(CustomInitializedNet, self).__init__()
        
        self.layer_sizes = layer_sizes
        self.activation_type = activation_type
        
        # 选择激活函数
        if activation_type.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation_type.lower() == "tanh":
            self.activation = nn.Tanh()
        elif activation_type.lower() == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
        
        # 创建网络层
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # 最后一层不加激活函数
                layers.append(self.activation)
        
        self.network = nn.Sequential(*layers)
        
        # 使用自定义初始化策略
        self._custom_init_weights()
    
    def _custom_init_weights(self):
        """自定义权重初始化策略"""
        print(f"\n使用自定义初始化策略，激活函数: {self.activation_type}")
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 根据激活函数选择初始化方法
                if self.activation_type.lower() == "relu":
                    # Kaiming初始化，针对ReLU优化
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    init_method = "Kaiming"
                elif self.activation_type.lower() in ["tanh", "sigmoid"]:
                    # Xavier初始化，针对tanh/sigmoid优化
                    nn.init.xavier_uniform_(module.weight)
                    init_method = "Xavier"
                else:
                    # 默认使用正态分布初始化
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    init_method = "Normal"
                
                # 初始化偏置
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                
                print(f"  初始化层 {module.weight.shape}: {init_method}初始化")
    
    def forward(self, x):
        return self.network(x)

# 工具函数：分析网络结构
def analyze_network_structure(model: nn.Module, input_shape: tuple = None):
    """分析网络结构，显示所有模块信息"""
    print("\n" + "="*60)
    print("网络结构分析")
    print("="*60)
    
    total_params = 0
    total_layers = 0
    
    print("网络中的所有模块:")
    for name, module in model.named_modules():
        if name:  # 跳过根模块（空字符串）
            num_params = sum(p.numel() for p in module.parameters())
            total_params += num_params
            
            if isinstance(module, nn.Linear):
                layer_type = "Linear"
                total_layers += 1
            elif isinstance(module, nn.Conv2d):
                layer_type = "Conv2d"
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                layer_type = "BatchNorm"
            elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
                layer_type = "Activation"
            else:
                layer_type = "Other"
            
            print(f"  {name}: {type(module).__name__} ({layer_type}), 参数数量: {num_params}")
    
    print(f"\n总结:")
    print(f"  总参数数量: {total_params:,}")
    print(f"  线性层数量: {total_layers}")
    
    # 如果需要，可以模拟前向传播获取各层输出形状
    if input_shape is not None:
        print(f"\n前向传播形状跟踪 (输入: {input_shape}):")
        
        # 注册钩子来捕获各层输出
        def hook_fn(module, input, output):
            module_name = str(module.__class__.__name__)
            print(f"    {module_name}: {input[0].shape if input else 'N/A'} -> {output.shape}")
        
        handles = []
        for name, module in model.named_modules():
            if name:  # 跳过根模块
                handles.append(module.register_forward_hook(hook_fn))
        
        # 创建模拟输入
        if len(input_shape) == 2:
            x = torch.randn(*input_shape)
        else:
            x = torch.randn(1, *input_shape)
        
        # 前向传播
        with torch.no_grad():
            _ = model(x)
        
        # 移除钩子
        for handle in handles:
            handle.remove()
    
    return total_params, total_layers

# 主测试函数
def main():
    """主测试函数，演示self.modules()的使用"""
    print("="*60)
    print("使用self.modules()进行权重初始化的完整示例")
    print("="*60)
    
    # 测试1: 简单的MLP网络
    print("\n测试1: 简单的MLP网络")
    print("-"*40)
    
    simple_mlp = SimpleMLP(
        input_size=10,
        hidden_sizes=[64, 32, 16],
        output_size=1,
        activation=nn.ReLU()
    )
    
    # 分析网络结构
    total_params, total_layers = analyze_network_structure(simple_mlp, input_shape=(2, 10))
    
    # 测试前向传播
    test_input = torch.randn(5, 10)
    output = simple_mlp(test_input)
    print(f"\n测试输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试2: 复杂嵌套网络
    print("\n\n测试2: 复杂嵌套网络")
    print("-"*40)
    
    complex_net = ComplexNestedNet(input_size=128, num_blocks=2)
    
    # 分析网络结构
    total_params_complex, total_layers_complex = analyze_network_structure(
        complex_net, 
        input_shape=(2, 128)
    )
    
    # 测试前向传播
    test_input_complex = torch.randn(4, 128)
    output_complex = complex_net(test_input_complex)
    print(f"\n测试输入形状: {test_input_complex.shape}")
    print(f"输出形状: {output_complex.shape}")
    
    # 测试3: 自定义初始化策略
    print("\n\n测试3: 不同激活函数的自定义初始化")
    print("-"*40)
    
    # 测试ReLU激活函数
    print("\n1. 使用ReLU激活函数:")
    relu_net = CustomInitializedNet(
        layer_sizes=[20, 64, 32, 10, 1],
        activation_type="relu"
    )
    
    # 测试tanh激活函数
    print("\n2. 使用tanh激活函数:")
    tanh_net = CustomInitializedNet(
        layer_sizes=[20, 64, 32, 10, 1],
        activation_type="tanh"
    )
    
    # 测试sigmoid激活函数
    print("\n3. 使用sigmoid激活函数:")
    sigmoid_net = CustomInitializedNet(
        layer_sizes=[20, 64, 32, 10, 1],
        activation_type="sigmoid"
    )
    
    # 测试4: 验证权重初始化效果
    print("\n\n测试4: 验证权重初始化效果")
    print("-"*40)
    
    def analyze_weight_distribution(model: nn.Module, model_name: str):
        """分析模型权重的分布"""
        all_weights = []
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                all_weights.append(module.weight.data.cpu().numpy().flatten())
        
        if all_weights:
            all_weights = np.concatenate(all_weights)
            
            mean = np.mean(all_weights)
            std = np.std(all_weights)
            min_val = np.min(all_weights)
            max_val = np.max(all_weights)
            
            print(f"\n{model_name}权重统计:")
            print(f"  均值: {mean:.6f}")
            print(f"  标准差: {std:.6f}")
            print(f"  最小值: {min_val:.6f}")
            print(f"  最大值: {max_val:.6f}")
            print(f"  范围: {max_val - min_val:.6f}")
            
            return all_weights
        return None
    
    # 分析各个网络的权重分布
    relu_weights = analyze_weight_distribution(relu_net, "ReLU网络")
    tanh_weights = analyze_weight_distribution(tanh_net, "Tanh网络")
    sigmoid_weights = analyze_weight_distribution(sigmoid_net, "Sigmoid网络")
    
    # 绘制权重分布对比图
    if relu_weights is not None and tanh_weights is not None and sigmoid_weights is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].hist(relu_weights, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_xlabel('权重值')
        axes[0].set_ylabel('频数')
        axes[0].set_title('ReLU网络权重分布 (Kaiming初始化)')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(tanh_weights, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1].set_xlabel('权重值')
        axes[1].set_ylabel('频数')
        axes[1].set_title('Tanh网络权重分布 (Xavier初始化)')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].hist(sigmoid_weights, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[2].set_xlabel('权重值')
        axes[2].set_ylabel('频数')
        axes[2].set_title('Sigmoid网络权重分布 (Xavier初始化)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('weight_distributions_comparison.png', dpi=150, bbox_inches='tight')
        print("\n权重分布对比图已保存为 'weight_distributions_comparison.png'")
        plt.show()
    
    # 总结
    print("\n" + "="*60)
    print("使用self.modules()进行权重初始化的优势总结:")
    print("="*60)
    
    advantages = [
        "1. 全面性: 自动遍历网络中的所有模块，包括嵌套模块",
        "2. 健壮性: 无论网络结构多复杂，都能确保所有层被正确初始化",
        "3. 灵活性: 可以根据模块类型（Linear, Conv2d等）应用不同的初始化策略",
        "4. 可维护性: 初始化逻辑集中在一处，便于修改和调试",
        "5. 兼容性: 与nn.Sequential、nn.ModuleList等容器兼容",
        "6. 扩展性: 容易添加对新模块类型的初始化支持"
    ]
    
    for advantage in advantages:
        print(advantage)
    
    print("\n关键注意事项:")
    print("1. 使用isinstance()检查模块类型，针对不同类型应用不同初始化")
    print("2. 注意初始化偏置项，通常初始化为0")
    print("3. 根据激活函数类型选择Kaiming或Xavier初始化")
    print("4. 在__init__方法的最后调用初始化函数")

if __name__ == "__main__":
    main()
