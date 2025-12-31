"""
MPS内存管理函数解析
演示torch.mps.set_per_process_memory_fraction(0.9)的作用和用法
"""

import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

def explain_mps_memory_function():
    """解释MPS内存管理函数的作用"""
    print("torch.mps.set_per_process_memory_fraction(0.9) 详细解析")
    print("="*60)
    
    # 1. 函数基本作用
    print("\n1. 函数基本作用:")
    print("-"*40)
    
    print("""
torch.mps.set_per_process_memory_fraction(0.9) 的作用是：
设置当前PyTorch进程可以使用Apple GPU（M系列芯片）总内存的比例。

参数解释：
- 0.9: 表示允许PyTorch使用最多90%的GPU总内存
- 范围: 0.0到1.0之间，表示可用内存的比例
- 默认值: 通常是1.0（100%），但可能因系统而异

使用场景：
1. 防止单个进程占用全部GPU内存
2. 允许多个进程共享GPU内存
3. 为系统或其他应用预留GPU内存
4. 避免内存溢出导致程序崩溃""")
    
    # 2. 检查MPS可用性
    print("\n2. 检查MPS可用性:")
    print("-"*40)
    
    mps_available = False
    if hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'is_available'):
        mps_available = torch.backends.mps.is_available()
    
    print(f"MPS后端可用: {mps_available}")
    
    if not mps_available:
        print("注意: 当前系统不支持MPS，以下为模拟演示")
        print("MPS仅适用于Apple Silicon芯片（M1、M2、M3等）")
        return None
    
    # 3. 实际使用演示
    print("\n3. 实际使用演示:")
    print("-"*40)
    
    # 获取当前MPS内存统计
    def get_mps_memory_stats() -> Dict[str, float]:
        """获取MPS内存统计信息"""
        stats = {}
        
        try:
            # 获取已分配内存
            if hasattr(torch.mps, 'current_allocated_memory'):
                allocated = torch.mps.current_allocated_memory() / 1024**2  # 转换为MB
                stats['allocated_mb'] = allocated
            
            # 获取驱动程序分配的内存
            if hasattr(torch.mps, 'driver_allocated_memory'):
                driver_allocated = torch.mps.driver_allocated_memory() / 1024**2
                stats['driver_allocated_mb'] = driver_allocated
            
            # 获取总内存
            if hasattr(torch.mps, 'get_memory_info'):
                memory_info = torch.mps.get_memory_info('device')
                stats['total_mb'] = memory_info.total / 1024**2
                stats['free_mb'] = memory_info.free / 1024**2
                stats['used_mb'] = (memory_info.total - memory_info.free) / 1024**2
        
        except Exception as e:
            print(f"获取MPS内存信息失败: {e}")
        
        return stats
    
    # 演示设置不同内存比例
    memory_fractions = [0.3, 0.6, 0.9, 1.0]
    memory_results = []
    
    for fraction in memory_fractions:
        print(f"\n设置内存比例为: {fraction}")
        
        try:
            # 设置内存比例
            if hasattr(torch.mps, 'set_per_process_memory_fraction'):
                torch.mps.set_per_process_memory_fraction(fraction)
                print(f"  ✓ 已设置内存比例为: {fraction*100}%")
            else:
                print(f"  ✗ 当前PyTorch版本不支持set_per_process_memory_fraction")
                break
            
            # 获取内存统计
            stats = get_mps_memory_stats()
            if stats:
                print(f"  已分配内存: {stats.get('allocated_mb', 'N/A'):.2f} MB")
                print(f"  已使用内存: {stats.get('used_mb', 'N/A'):.2f} MB")
                print(f"  空闲内存: {stats.get('free_mb', 'N/A'):.2f} MB")
                print(f"  总内存: {stats.get('total_mb', 'N/A'):.2f} MB")
                
                memory_results.append({
                    'fraction': fraction,
                    'allocated_mb': stats.get('allocated_mb', 0),
                    'used_mb': stats.get('used_mb', 0),
                    'free_mb': stats.get('free_mb', 0),
                    'total_mb': stats.get('total_mb', 0)
                })
            
        except Exception as e:
            print(f"  设置内存比例失败: {e}")
    
    # 4. 内存限制测试
    print("\n4. 内存限制测试:")
    print("-"*40)
    
    def test_memory_limit():
        """测试内存限制是否生效"""
        print("测试内存限制...")
        
        # 设置内存限制为50%
        if hasattr(torch.mps, 'set_per_process_memory_fraction'):
            torch.mps.set_per_process_memory_fraction(0.5)
            print("  设置内存限制为50%")
        
        # 尝试分配大量内存
        try:
            # 获取总内存信息
            if hasattr(torch.mps, 'get_memory_info'):
                memory_info = torch.mps.get_memory_info('device')
                total_memory = memory_info.total
                print(f"  GPU总内存: {total_memory / 1024**3:.2f} GB")
                
                # 计算可用的50%
                available_memory = total_memory * 0.5
                print(f"  可用内存限制: {available_memory / 1024**3:.2f} GB")
                
                # 尝试分配接近限制的内存
                tensor_size = int((available_memory * 0.8) / 4)  # 80%的可用内存，float32
                print(f"  尝试分配: {tensor_size * 4 / 1024**3:.2f} GB")
                
                tensor = torch.zeros(tensor_size, dtype=torch.float32, device='mps')
                print(f"  ✓ 成功分配大张量")
                del tensor
                
        except RuntimeError as e:
            print(f"  ✗ 内存分配失败: {e}")
        except Exception as e:
            print(f"  ✗ 其他错误: {e}")
    
    test_memory_limit()
    
    # 5. 在训练场景中的应用
    print("\n5. 在训练场景中的应用:")
    print("-"*40)
    
    print("""在深度学习训练中，合理设置内存比例非常重要：

推荐设置:
1. 单任务训练: 0.8-0.9 (为系统和其他应用预留10-20%内存)
2. 多任务并行: 0.5-0.7 (为其他训练任务预留空间)
3. 开发调试: 0.7-0.8 (为IDE、浏览器等预留内存)
4. 生产部署: 0.9 (最大化利用，但要监控内存使用)

在您的期权定价模型中的应用:
# 训练开始前设置内存比例
torch.mps.set_per_process_memory_fraction(0.9)  # 使用90%的GPU内存

# 然后创建模型和开始训练
model = GPUCallOption(...)
model.train(...)

注意事项:
1. 这个设置是进程级别的，影响当前Python进程
2. 设置后立即生效，直到进程结束
3. 无法设置超过物理内存的大小
4. 实际可用内存可能略低于设置值（有管理开销）""")
    
    # 6. 与其他内存管理函数的对比
    print("\n6. 与其他内存管理函数的对比:")
    print("-"*40)
    
    comparison = {
        'torch.mps.set_per_process_memory_fraction': {
            '平台': 'macOS (Apple Silicon)',
            '作用': '设置进程可用的GPU内存比例',
            '范围': '0.0-1.0',
            '持久性': '进程生命周期内有效',
        },
        'torch.cuda.set_per_process_memory_fraction': {
            '平台': 'NVIDIA GPU (CUDA)',
            '作用': '设置进程可用的GPU内存比例',
            '范围': '0.0-1.0',
            '持久性': '进程生命周期内有效',
        },
        'torch.cuda.empty_cache': {
            '平台': 'NVIDIA GPU (CUDA)',
            '作用': '清空未使用的缓存内存',
            '范围': 'N/A',
            '持久性': '立即生效',
        },
        'torch.mps.empty_cache': {
            '平台': 'macOS (Apple Silicon)',
            '作用': '清空未使用的缓存内存',
            '范围': 'N/A',
            '持久性': '立即生效',
        }
    }
    
    for func_name, info in comparison.items():
        print(f"\n{func_name}:")
        print(f"  平台: {info['平台']}")
        print(f"  作用: {info['作用']}")
        print(f"  范围: {info['范围']}")
        print(f"  持久性: {info['持久性']}")
    
    # 7. 最佳实践建议
    print("\n7. 最佳实践建议:")
    print("-"*40)
    
    practices = [
        "1. 在训练开始前设置内存比例，而不是训练过程中",
        "2. 为系统预留至少10%的内存（设置为0.9）",
        "3. 在多进程应用中，合理分配内存比例",
        "4. 监控实际内存使用，避免设置过低导致OOM",
        "5. 结合torch.mps.empty_cache()管理内存",
        "6. 在遇到内存不足错误时，考虑降低比例或批次大小"
    ]
    
    for practice in practices:
        print(practice)
    
    return memory_results

def demonstrate_memory_fraction_impact():
    """演示不同内存比例设置的影响"""
    print("\n" + "="*60)
    print("不同内存比例设置的影响演示")
    print("="*60)
    
    # 检查MPS可用性
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("MPS不可用，使用模拟数据演示")
        
        # 创建模拟数据用于可视化
        fractions = [0.3, 0.5, 0.7, 0.9, 1.0]
        allocated_memory = [2.5, 4.2, 5.8, 7.5, 8.3]  # GB
        free_memory = [5.5, 3.8, 2.2, 0.5, 0.0]  # GB
        
        # 可视化内存使用
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # 第一个子图：内存分配
        ax[0].bar([str(f) for f in fractions], allocated_memory, color='skyblue')
        ax[0].set_xlabel('内存比例设置')
        ax[0].set_ylabel('已分配内存 (GB)')
        ax[0].set_title('不同内存比例下的已分配内存')
        ax[0].grid(True, alpha=0.3)
        
        # 第二个子图：内存使用效率
        efficiency = [a/(a+f) for a, f in zip(allocated_memory, free_memory)]
        ax[1].plot(fractions, efficiency, 'o-', linewidth=2, markersize=8)
        ax[1].set_xlabel('内存比例设置')
        ax[1].set_ylabel('内存使用效率')
        ax[1].set_title('内存使用效率 vs 内存比例')
        ax[1].grid(True, alpha=0.3)
        ax[1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('mps_memory_fraction_impact.png', dpi=150, bbox_inches='tight')
        print("内存比例影响图已保存为 'mps_memory_fraction_impact.png'")
        plt.show()
    
    else:
        print("MPS可用，运行实际测试...")
        
        # 实际测试不同内存比例
        test_fractions = [0.3, 0.5, 0.7, 0.9]
        results = []
        
        for fraction in test_fractions:
            print(f"\n测试内存比例: {fraction}")
            
            # 设置内存比例
            torch.mps.set_per_process_memory_fraction(fraction)
            
            # 尝试分配内存
            try:
                # 分配一个较大的张量
                tensor_size = 10000
                tensor = torch.randn(tensor_size, tensor_size, device='mps')
                
                # 获取内存信息
                if hasattr(torch.mps, 'current_allocated_memory'):
                    allocated = torch.mps.current_allocated_memory() / 1024**3
                    results.append((fraction, allocated))
                    print(f"  成功分配，当前已分配: {allocated:.2f} GB")
                
                del tensor
                
            except RuntimeError as e:
                print(f"  内存分配失败: {e}")
                results.append((fraction, 0))
        
        # 可视化结果
        if results:
            fractions, allocated = zip(*results)
            
            plt.figure(figsize=(8, 5))
            plt.plot(fractions, allocated, 'o-', linewidth=2, markersize=8)
            plt.xlabel('内存比例设置')
            plt.ylabel('实际分配内存 (GB)')
            plt.title('MPS内存比例设置 vs 实际分配内存')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('mps_memory_fraction_impact.png', dpi=150, bbox_inches='tight')
            print("内存比例影响图已保存为 'mps_memory_fraction_impact.png'")
            plt.show()
    
    return True

# 主函数
if __name__ == "__main__":
    print("MPS内存管理函数详细解析")
    print("="*60)
    
    # 解析函数作用
    results = explain_mps_memory_function()
    
    # 演示内存比例影响
    demonstrate_memory_fraction_impact()
    
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    
    print("""
关键要点:

1. torch.mps.set_per_process_memory_fraction(0.9) 的作用:
   - 限制当前PyTorch进程可使用的Apple GPU内存比例
   - 参数0.9表示可以使用最多90%的GPU总内存
   - 为系统和其他应用预留10%的内存空间

2. 使用场景:
   - 防止单个进程占用全部GPU内存
   - 允许多个进程共享GPU资源
   - 提高系统稳定性，避免内存溢出

3. 在您的代码中的应用:
   - 在期权定价模型训练开始前设置
   - 建议设置为0.8-0.9，为系统预留内存
   - 结合torch.mps.empty_cache()进行内存管理

4. 注意事项:
   - 仅适用于Apple Silicon芯片（M1、M2、M3等）
   - 设置后在整个进程生命周期内有效
   - 需要PyTorch版本支持MPS后端
    """)
