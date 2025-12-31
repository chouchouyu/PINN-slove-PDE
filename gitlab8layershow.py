import numpy as np
import torch
import torch.nn as nn
from FBSNNs import *
from gitlab7layershow import NetworkAnalyzer #network_analyzer
from CallOption import *

  

# 创建NetworkAnalyzer实例（假设已经导入）
def analyze_fbsnn_network():
    # 设置FBSNN参数
    
    T = 1.0                    # 终止时间
    M = 1                    # 轨迹数
    N = 50                     # 时间步数
    D = 1                     # 维度
    Mm = N ** (1/5)                   # 离散点数
    layers = [D + 1] + 4 * [256] + [1]  # 网络层结构 [输入维度, 隐藏层, 输出维度]
    Xi = np.array([1.0] * D)[None, :]  # 初始条件，2维
    # 测试两种模式
    modes = ["FC", "Naisnet"]
    activations = ["ReLU", "Tanh","Sine"]
    
    for mode in modes:
        for activation in activations:
            print(f"\n{'='*60}")
            print(f"分析 {mode} 模式, 激活函数: {activation}")
            print(f"{'='*60}")
            
            try:
                # 创建FBSNN实例
                fbsnn = CallOption(Xi, T, M, N, D, Mm, layers, mode, activation)
                
                # 创建网络分析器
                analyzer = NetworkAnalyzer(fbsnn.model)
                
                # 分析网络结构
                # 输入形状: (batch_size, input_dim) = (M, D+1) 因为输入是(t, X)的拼接
                input_shape = (M, D+1)  # M x (D+1)
                
                # 打印网络摘要
                analyzer.print_summary(input_shape=input_shape)
                
                # 详细分析结构
                print(f"\n详细层信息:")
                struct_info = analyzer.analyze_structure(input_shape=input_shape, verbose=True)
                
                # 参数统计
                param_stats = analyzer.get_parameter_statistics()
                print(f"\n参数统计:")
                print(f"  总参数: {param_stats['total_params']:,}")
                print(f"  可训练参数: {param_stats['trainable_params']:,}")
                print(f"  不可训练参数: {param_stats['non_trainable_params']:,}")
                
            except Exception as e:
                print(f"创建或分析 {mode} 网络时出错: {e}")

 

  

# 主执行函数
if __name__ == "__main__":
    print("FBSNN网络结构分析")
    print("="*60)
    
    # 分析基本配置
    analyze_fbsnn_network()
    
 