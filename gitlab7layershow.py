"""
神经网络结构分析工具
提供PyTorch模型的结构分析、参数统计和可视化功能
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import sys
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class NetworkAnalyzer:
    """神经网络结构分析器"""
    
    def __init__(self, model: nn.Module):
        """
        初始化分析器
        
        参数:
            model: 要分析的PyTorch模型
        """
        self.model = model
        self.hook_handles = []
    
    def analyze_structure(self, input_shape: Optional[Tuple] = None, 
                         verbose: bool = True) -> Dict[str, Any]:
        """
        分析网络结构，显示所有模块信息
        
        参数:
            input_shape: 输入形状，用于前向传播形状跟踪
            verbose: 是否打印详细信息
            
        返回:
            包含网络结构信息的字典
        """
        if verbose:
            self._print_header("网络结构分析")
        
        total_params = 0
        total_layers = 0
        layer_info = []
        
        if verbose:
            print("网络中的所有模块:")
        
        for name, module in self.model.named_modules():
            if name:  # 跳过根模块（空字符串）
                num_params = sum(p.numel() for p in module.parameters())
                total_params += num_params
                
                # 确定层类型
                if isinstance(module, nn.Linear):
                    layer_type = "Linear"
                    total_layers += 1
                elif isinstance(module, nn.Conv2d):
                    layer_type = "Conv2d"
                elif isinstance(module, nn.Conv1d):
                    layer_type = "Conv1d"
                elif isinstance(module, nn.Conv3d):
                    layer_type = "Conv3d"
                elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    layer_type = "BatchNorm"
                elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, 
                                        nn.InstanceNorm2d, nn.InstanceNorm3d)):
                    layer_type = "Normalization"
                elif isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, 
                                        nn.Softmax, nn.GELU, nn.SiLU, nn.Mish)):
                    layer_type = "Activation"
                elif isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                    layer_type = "Dropout"
                elif isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
                                        nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
                                        nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, 
                                        nn.AdaptiveAvgPool3d, nn.AdaptiveMaxPool1d,
                                        nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d)):
                    layer_type = "Pooling"
                elif isinstance(module, nn.Flatten):
                    layer_type = "Flatten"
                elif isinstance(module, nn.Embedding):
                    layer_type = "Embedding"
                elif isinstance(module, nn.LSTM):
                    layer_type = "LSTM"
                elif isinstance(module, nn.GRU):
                    layer_type = "GRU"
                elif isinstance(module, nn.RNN):
                    layer_type = "RNN"
                elif isinstance(module, nn.Transformer):
                    layer_type = "Transformer"
                elif isinstance(module, nn.MultiheadAttention):
                    layer_type = "MultiheadAttention"
                else:
                    layer_type = "Other"
                
                layer_info.append({
                    'name': name,
                    'type': type(module).__name__,
                    'layer_type': layer_type,
                    'num_params': num_params,
                    'module': module
                })
                
                if verbose:
                    print(f"  {name}: {type(module).__name__} ({layer_type}), 参数数量: {num_params}")
        
        if verbose:
            self._print_header("总结")
            print(f"  总参数数量: {total_params:,}")
            print(f"  可训练层数量: {total_layers}")
            print(f"  总模块数量: {len(layer_info)}")
        
        result = {
            'total_params': total_params,
            'total_layers': total_layers,
            'total_modules': len(layer_info),
            'layer_info': layer_info
        }
        
        # 如果需要，可以模拟前向传播获取各层输出形状
        if input_shape is not None:
            shape_info = self._analyze_forward_shapes(input_shape, verbose)
            result['shape_info'] = shape_info
        
        return result
    
    def _analyze_forward_shapes(self, input_shape: Tuple, verbose: bool = True) -> List[Dict]:
        """
        分析前向传播形状
        
        参数:
            input_shape: 输入形状
            verbose: 是否打印详细信息
            
        返回:
            形状信息列表
        """
        shape_info = []
        
        if verbose:
            print(f"\n前向传播形状跟踪 (输入: {input_shape}):")
        
        # 注册钩子来捕获各层输出
        def hook_fn(module, input, output, name):
            input_shape = input[0].shape if input and len(input) > 0 else 'N/A'
            output_shape = output.shape if output is not None else 'N/A'
            
            info = {
                'name': name,
                'module_type': type(module).__name__,
                'input_shape': input_shape,
                'output_shape': output_shape
            }
            shape_info.append(info)
            
            if verbose:
                print(f"    {type(module).__name__} ({name}): {input_shape} -> {output_shape}")
        
        # 注册钩子
        for name, module in self.model.named_modules():
            if name:  # 跳过根模块
                handle = module.register_forward_hook(
                    lambda m, inp, out, n=name: hook_fn(m, inp, out, n)
                )
                self.hook_handles.append(handle)
        
        # 创建模拟输入
        try:
            if len(input_shape) == 2:
                x = torch.randn(*input_shape)
            else:
                x = torch.randn(1, *input_shape)
            
            # 移动到模型所在的设备
            x = x.to(next(self.model.parameters()).device)
            
            # 前向传播
            with torch.no_grad():
                _ = self.model(x)
                
        except Exception as e:
            if verbose:
                print(f"  前向传播跟踪失败: {e}")
        finally:
            # 移除钩子
            self._remove_hooks()
        
        return shape_info
    
    def _remove_hooks(self):
        """移除所有注册的钩子"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
    
    def get_parameter_statistics(self) -> Dict[str, Any]:
        """
        获取参数统计信息
        
        返回:
            参数统计信息字典
        """
        param_stats = {
            'total_params': 0,
            'trainable_params': 0,
            'non_trainable_params': 0,
            'param_details': []
        }
        
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            is_trainable = param.requires_grad
            
            param_stats['total_params'] += num_params
            if is_trainable:
                param_stats['trainable_params'] += num_params
            else:
                param_stats['non_trainable_params'] += num_params
            
            param_stats['param_details'].append({
                'name': name,
                'shape': tuple(param.shape),
                'num_params': num_params,
                'requires_grad': is_trainable,
                'dtype': str(param.dtype)
            })
        
        return param_stats
    
    def get_memory_usage(self, input_shape: Tuple) -> Dict[str, Any]:
        """
        估计模型的内存使用情况
        
        参数:
            input_shape: 输入形状
            
        返回:
            内存使用信息字典
        """
        # 计算参数内存
        param_memory = 0
        for param in self.model.parameters():
            param_memory += param.numel() * param.element_size()
        
        # 转换为更友好的单位
        def format_bytes(bytes_num):
            for unit in ['B', 'KB', 'MB', 'GB']:
                if bytes_num < 1024.0:
                    return f"{bytes_num:.2f} {unit}"
                bytes_num /= 1024.0
            return f"{bytes_num:.2f} TB"
        
        # 估计前向传播的内存
        try:
            if len(input_shape) == 2:
                x = torch.randn(*input_shape)
            else:
                x = torch.randn(1, *input_shape)
            
            # 移动到模型所在的设备
            device = next(self.model.parameters()).device
            x = x.to(device)
            
            # 进行一次前向传播
            with torch.no_grad():
                _ = self.model(x)
            
            if device.type == 'cuda':
                # 获取GPU内存使用
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                
                forward_memory = {
                    'allocated': format_bytes(allocated),
                    'reserved': format_bytes(reserved),
                    'allocated_bytes': allocated,
                    'reserved_bytes': reserved
                }
            else:
                forward_memory = {
                    'estimated': format_bytes(param_memory * 3),  # 粗略估计
                    'estimated_bytes': param_memory * 3
                }
                
        except Exception as e:
            forward_memory = {'error': str(e)}
        
        return {
            'parameters': format_bytes(param_memory),
            'parameters_bytes': param_memory,
            'forward_pass': forward_memory
        }
    
    def _print_header(self, title: str):
        """打印标题分隔线"""
        print("\n" + "="*60)
        print(title)
        print("="*60)
    
    def print_summary(self, input_shape: Optional[Tuple] = None):
        """
        打印网络完整摘要
        
        参数:
            input_shape: 输入形状（可选）
        """
        self._print_header("神经网络分析摘要")
        
        # 1. 基本结构分析
        struct_info = self.analyze_structure(input_shape=input_shape, verbose=False)
        
        print(f"模型名称: {self.model.__class__.__name__}")
        print(f"总参数数量: {struct_info['total_params']:,}")
        print(f"可训练层数量: {struct_info['total_layers']}")
        print(f"总模块数量: {struct_info['total_modules']}")
        
        # 2. 参数统计
        param_stats = self.get_parameter_statistics()
        print(f"\n参数统计:")
        print(f"  可训练参数: {param_stats['trainable_params']:,}")
        print(f"  不可训练参数: {param_stats['non_trainable_params']:,}")
        print(f"  参数总数: {param_stats['total_params']:,}")
        
        # 3. 层类型分布
        layer_counts = {}
        for info in struct_info['layer_info']:
            layer_type = info['layer_type']
            layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
        
        print(f"\n层类型分布:")
        for layer_type, count in sorted(layer_counts.items()):
            print(f"  {layer_type}: {count}")
        
        # 4. 内存使用估计
        if input_shape is not None:
            memory_info = self.get_memory_usage(input_shape)
            print(f"\n内存使用估计 (输入形状: {input_shape}):")
            print(f"  参数内存: {memory_info['parameters']}")
            
            if 'forward_pass' in memory_info:
                if 'allocated' in memory_info['forward_pass']:
                    print(f"  前向传播GPU内存: {memory_info['forward_pass']['allocated']} (已分配)")
                    print(f"  前向传播GPU内存: {memory_info['forward_pass']['reserved']} (已保留)")
                elif 'estimated' in memory_info['forward_pass']:
                    print(f"  前向传播估计内存: {memory_info['forward_pass']['estimated']}")
        
        # 5. 设备信息
        try:
            device = next(self.model.parameters()).device
            print(f"\n设备信息:")
            print(f"  模型所在设备: {device}")
            if device.type == 'cuda':
                print(f"  GPU名称: {torch.cuda.get_device_name(device)}")
        except StopIteration:
            print(f"\n设备信息: 模型没有参数")

# 测试函数
def test_network_analyzer():
    """测试网络分析器功能"""
    print("测试神经网络分析器...")
    print("="*60)
    
    # 创建一个简单的测试网络
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2, 2)
            
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2, 2)
            
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(32 * 7 * 7, 128)
            self.relu3 = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, 10)
            self.softmax = nn.Softmax(dim=1)
        
        def forward(self, x):
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = self.flatten(x)
            x = self.dropout(self.relu3(self.fc1(x)))
            x = self.fc2(x)
            return self.softmax(x)
    
    # 创建模型和分析器
    model = SimpleCNN()
    analyzer = NetworkAnalyzer(model)
    
    # 测试1: 基本结构分析
    print("测试1: 基本结构分析")
    struct_info = analyzer.analyze_structure(input_shape=(1, 3, 28, 28), verbose=True)
    
    # 测试2: 参数统计
    print("\n测试2: 参数统计")
    param_stats = analyzer.get_parameter_statistics()
    print(f"总参数: {param_stats['total_params']:,}")
    print(f"可训练参数: {param_stats['trainable_params']:,}")
    print(f"不可训练参数: {param_stats['non_trainable_params']:,}")
    
    # 测试3: 内存使用估计
    print("\n测试3: 内存使用估计")
    memory_info = analyzer.get_memory_usage(input_shape=(1, 3, 28, 28))
    print(f"参数内存: {memory_info['parameters']}")
    
    # 测试4: 完整摘要
    print("\n测试4: 完整摘要")
    analyzer.print_summary(input_shape=(1, 3, 28, 28))
    
    return True

# 使用示例
if __name__ == "__main__":
    print("神经网络结构分析工具")
    print("="*60)
    
    # 运行测试
    success = test_network_analyzer()
    
    if success:
        print("\n" + "="*60)
        print("测试完成!")
        print("="*60)
        
        print("""
使用示例:
    
# 1. 导入模块
from network_analyzer import NetworkAnalyzer

# 2. 创建您的模型
model = YourModel()

# 3. 创建分析器
analyzer = NetworkAnalyzer(model)

# 4. 分析网络结构
analyzer.analyze_structure(input_shape=(batch_size, channels, height, width))

# 5. 获取完整摘要
analyzer.print_summary(input_shape=(batch_size, channels, height, width))

# 6. 获取参数统计
param_stats = analyzer.get_parameter_statistics()

# 7. 获取内存使用估计
memory_info = analyzer.get_memory_usage(input_shape=(batch_size, channels, height, width))
        """)
    else:
        print("测试失败!")
