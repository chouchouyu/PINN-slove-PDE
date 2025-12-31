"""
torch.cuda.empty_cache() 对GPU加速影响分析
展示缓存清理的作用、使用场景和性能影响
修复了IndexError: pop index out of range错误
"""

import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import gc
import random

def demonstrate_empty_cache_effect():
    """演示empty_cache的作用效果"""
    print("torch.cuda.empty_cache() 对GPU加速影响分析")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA不可用，请在支持CUDA的GPU上运行此示例")
        return None
    
    device = torch.device("cuda:0")
    
    # 1. 基本原理说明
    print("\n1. 基本原理:")
    print("-"*40)
    
    print("""PyTorch GPU内存管理机制:

1. 分配内存: PyTorch通过CUDA内存分配器请求GPU内存
2. 缓存机制: 已释放的内存块被缓存以便快速重用
3. 内存碎片: 频繁分配释放可能产生内存碎片
4. empty_cache(): 强制清空缓存，释放未使用的内存""")
    
    # 2. 展示缓存清理效果
    print("\n2. 缓存清理效果演示")
    print("-"*40)
    
    def get_gpu_memory_info():
        """获取GPU内存信息"""
        allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(device) / 1024**2    # MB
        return allocated, reserved
    
    print("初始内存状态:")
    init_alloc, init_reserved = get_gpu_memory_info()
    print(f"  已分配内存: {init_alloc:.2f} MB")
    print(f"  已保留内存: {init_reserved:.2f} MB")
    
    # 创建一些大张量占用内存
    print("\n创建大张量占用内存...")
    large_tensors = []
    tensor_sizes = [1000, 2000, 3000, 4000, 5000]
    
    for i, size in enumerate(tensor_sizes):
        tensor = torch.randn(size, size, device=device)
        large_tensors.append(tensor)
        alloc, reserved = get_gpu_memory_info()
        print(f"  创建{size}x{size}张量后: 已分配={alloc:.2f}MB, 已保留={reserved:.2f}MB")
    
    # 释放张量
    print("\n释放所有张量...")
    for tensor in large_tensors:
        del tensor
    
    # 强制垃圾回收
    gc.collect()
    
    # 查看释放后的内存
    alloc_after_del, reserved_after_del = get_gpu_memory_info()
    print(f"  删除后: 已分配={alloc_after_del:.2f}MB, 已保留={reserved_after_del:.2f}MB")
    print(f"  注意: 已保留内存仍然较高，因为PyTorch缓存了内存块")
    
    # 使用empty_cache清理缓存
    print("\n调用torch.cuda.empty_cache()...")
    torch.cuda.empty_cache()
    
    alloc_after_empty, reserved_after_empty = get_gpu_memory_info()
    print(f"  清理后: 已分配={alloc_after_empty:.2f}MB, 已保留={reserved_after_empty:.2f}MB")
    print(f"  内存释放: {reserved_after_del - reserved_after_empty:.2f}MB")
    
    # 3. 性能影响测试
    print("\n3. 性能影响测试")
    print("-"*40)
    
    def performance_test_with_cache():
        """测试带缓存的性能"""
        times = []
        
        for _ in range(10):
            # 创建随机张量
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            
            # 执行矩阵乘法
            start = time.time()
            for _ in range(100):
                _ = torch.matmul(x, y)
            torch.cuda.synchronize()
            end = time.time()
            
            times.append(end - start)
            
            # 清理
            del x, y
            
        return np.mean(times) * 1000  # 转换为毫秒
    
    def performance_test_without_cache(use_empty_cache=True):
        """测试不带缓存的性能"""
        times = []
        
        for _ in range(10):
            # 创建随机张量
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            
            # 执行矩阵乘法
            start = time.time()
            for _ in range(100):
                _ = torch.matmul(x, y)
            torch.cuda.synchronize()
            end = time.time()
            
            times.append(end - start)
            
            # 清理
            del x, y
            
            if use_empty_cache:
                torch.cuda.empty_cache()
            
        return np.mean(times) * 1000  # 转换为毫秒
    
    print("运行性能测试...")
    
    # 先预热GPU
    print("  预热GPU...")
    _ = performance_test_with_cache()
    
    # 测试带缓存的性能
    print("  测试带缓存性能...")
    time_with_cache = performance_test_with_cache()
    
    # 清理缓存
    torch.cuda.empty_cache()
    
    # 测试每次迭代都清理缓存的性能
    print("  测试每次迭代清理缓存性能...")
    time_with_frequent_empty = performance_test_without_cache(use_empty_cache=True)
    
    # 测试不清理缓存的性能
    print("  测试不清理缓存性能...")
    time_without_empty = performance_test_without_cache(use_empty_cache=False)
    
    print(f"\n性能测试结果:")
    print(f"  带缓存: {time_with_cache:.2f} ms/迭代")
    print(f"  频繁清理缓存: {time_with_frequent_empty:.2f} ms/迭代")
    print(f"  不清理缓存: {time_without_empty:.2f} ms/迭代")
    print(f"  性能差异: 频繁清理比带缓存慢 {time_with_frequent_empty/time_with_cache:.2f}倍")
    
    # 4. 内存碎片化测试 - 修复版
    print("\n4. 内存碎片化测试")
    print("-"*40)
    
    def simulate_fragmentation():
        """模拟内存碎片化场景 - 修复索引错误"""
        # 创建不同大小的张量
        tensors = []
        sizes = [100, 500, 200, 800, 300, 700, 400, 600]
        
        for i, size in enumerate(sizes):
            tensor = torch.randn(size, size, device=device)
            tensors.append(tensor)
        
        # 随机删除一些张量 - 修复：只需要删除一次
        num_to_remove = len(tensors) // 2
        
        for _ in range(num_to_remove):
            if tensors:  # 确保列表不为空
                idx = random.randint(0, len(tensors) - 1)
                # 删除张量并从列表中移除 - 只做一次操作
                removed_tensor = tensors.pop(idx)
                del removed_tensor  # 显式删除以释放GPU内存
        
        # 尝试分配大张量
        try:
            large_tensor = torch.randn(1000, 1000, device=device)
            print("  内存碎片化测试: 可以分配大张量")
            del large_tensor
        except RuntimeError as e:
            print(f"  内存碎片化测试: 无法分配大张量 - {e}")
            
            # 尝试清理缓存
            print("  尝试清理缓存...")
            torch.cuda.empty_cache()
            
            try:
                large_tensor = torch.randn(1000, 1000, device=device)
                print("  清理后: 可以分配大张量")
                del large_tensor
            except RuntimeError as e2:
                print(f"  清理后仍然失败: {e2}")
        
        # 清理剩余的张量
        for tensor in tensors:
            del tensor
    
    print("运行内存碎片化测试...")
    simulate_fragmentation()
    
    # 5. 最佳实践建议
    print("\n5. 最佳实践建议")
    print("-"*40)
    
    recommendations = [
        "1. 在长时间训练循环中，定期使用empty_cache():",
        "   场景: 每100-1000次迭代清理一次，避免内存泄漏积累",
        "",
        "2. 在遇到CUDA out of memory错误时使用:",
        "   场景: 捕获内存不足异常，清理缓存后重试",
        "",
        "3. 在模型切换或批处理大小变化时使用:",
        "   场景: 不同模型可能需要不同内存布局，清理缓存减少碎片",
        "",
        "4. 在推理服务中谨慎使用:",
        "   场景: 实时推理对延迟敏感，频繁清理会降低性能",
        "",
        "5. 在内存受限环境中使用:",
        "   场景: GPU内存紧张时，主动清理可释放内存",
        "",
        "6. 避免在训练循环内部频繁调用:",
        "   场景: 每次迭代都清理会显著降低训练速度"
    ]
    
    for rec in recommendations:
        print(rec)
    
    # 6. 在期权定价模型中的应用
    print("\n6. 在期权定价模型中的应用")
    print("-"*40)
    
    print("""在GPUCallOption期权定价模型中:

使用场景:
1. 蒙特卡洛模拟: 每次模拟生成大量随机路径，可能产生内存碎片
2. 批量处理: 大批量训练时，不同批可能大小不同
3. 长时间训练: 训练数万次迭代可能积累未释放内存

建议策略:
1. 每1000次迭代清理一次: 平衡内存使用和性能
2. 在批处理大小变化时清理: 确保内存布局最优
3. 捕获内存异常: 在try-catch中处理OOM错误

示例代码:
# 在训练循环中定期清理缓存
for iteration in range(num_iterations):
    # 训练步骤...
    if iteration % 1000 == 0:
        torch.cuda.empty_cache()
    
# 在批处理大小变化时清理缓存
def change_batch_size(new_batch_size):
    torch.cuda.empty_cache()
    # 重新配置批处理大小...
    
# 处理内存不足异常
try:
    # 内存密集型操作
    result = memory_intensive_operation()
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    # 重试或调整策略""")
    
    # 7. 可视化内存使用情况
    print("\n7. 可视化内存使用情况")
    print("-"*40)
    
    # 模拟内存使用变化
    memory_history = []
    
    # 记录初始内存
    alloc, reserved = get_gpu_memory_info()
    memory_history.append(("初始", alloc, reserved))
    
    # 分配内存
    tensor1 = torch.randn(2000, 2000, device=device)
    alloc, reserved = get_gpu_memory_info()
    memory_history.append(("分配大张量", alloc, reserved))
    
    # 释放但不清理缓存
    del tensor1
    gc.collect()
    alloc, reserved = get_gpu_memory_info()
    memory_history.append(("释放不清理", alloc, reserved))
    
    # 清理缓存
    torch.cuda.empty_cache()
    alloc, reserved = get_gpu_memory_info()
    memory_history.append(("清理缓存", alloc, reserved))
    
    # 绘制内存使用图
    labels = [m[0] for m in memory_history]
    allocated = [m[1] for m in memory_history]
    reserved = [m[2] for m in memory_history]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, allocated, width, label='已分配内存 (MB)', color='skyblue')
    rects2 = ax.bar(x + width/2, reserved, width, label='已保留内存 (MB)', color='lightcoral')
    
    ax.set_xlabel('内存状态')
    ax.set_ylabel('内存使用 (MB)')
    ax.set_title('GPU内存使用变化')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # 添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig('gpu_memory_usage.png', dpi=150, bbox_inches='tight')
    print("内存使用图已保存为 'gpu_memory_usage.png'")
    plt.show()
    
    return memory_history

# 运行演示
if __name__ == "__main__":
    print("开始GPU缓存清理分析...")
    result = demonstrate_empty_cache_effect()
    
    if result is not None:
        print("\n" + "="*60)
        print("分析完成!")
        print("="*60)
        
        print("\n关键结论:")
        print("1. torch.cuda.empty_cache() 会清理未使用的GPU内存缓存")
        print("2. 频繁调用会降低性能（内存分配需要重新请求）")
        print("3. 适当使用可以解决内存碎片和OOM问题")
        print("4. 建议在训练循环中每1000次迭代调用一次")
    else:
        print("分析未完成，CUDA不可用")
