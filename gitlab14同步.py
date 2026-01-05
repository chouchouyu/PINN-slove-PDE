"""
torch.cuda.synchronize() 作用演示
展示CUDA同步机制的重要性和使用场景
"""

import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

def demonstrate_cuda_synchronize():
    """演示torch.cuda.synchronize()的作用"""
    print("torch.cuda.synchronize() 作用演示")
    print("="*60)
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("CUDA不可用，请在支持CUDA的GPU上运行此示例")
        return None
    
    device = torch.device("cuda:0")
    print(f"使用设备: {device}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    
    # 1. 基本原理解释
    print("\n1. 基本原理:")
    print("-"*40)
    
    print("""
torch.cuda.synchronize() 的作用：

CUDA操作（如内核启动、内存拷贝）通常是异步的：
1. CPU发送指令到GPU后，立即继续执行后续代码
2. GPU在后台并行执行这些操作
3. 如果不等待GPU完成，计时会不准确，可能导致竞争条件

synchronize() 的作用：
- 阻塞CPU执行，直到所有CUDA操作完成
- 确保计时准确性
- 防止数据竞争
- 调试CUDA程序时非常有用""")
    
    # 2. 演示异步性导致的时间测量不准确
    print("\n2. 异步操作导致的时间测量问题")
    print("-"*40)
    
    def test_async_operations():
        """测试异步操作的时间测量"""
        # 创建一些测试数据
        size = 10000
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        print(f"测试矩阵大小: {size}x{size}")
        print(f"矩阵元素数: {size*size:,}")
        
        # 测试1: 不使用synchronize()的时间测量
        print("\n测试1: 不使用synchronize()")
        start_time = time.time()
        for _ in range(10):
            c = torch.matmul(a, b)
        # 注意：这里没有同步，时间可能不准确
        time_without_sync = time.time() - start_time
        print(f"  测量时间: {time_without_sync:.4f}秒")
        
        # 测试2: 使用synchronize()的时间测量
        print("\n测试2: 使用synchronize()")
        start_time = time.time()
        for _ in range(10):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()  # 等待所有CUDA操作完成
        time_with_sync = time.time() - start_time
        print(f"  测量时间: {time_with_sync:.4f}秒")
        
        # 测试3: 每个操作后都使用synchronize()
        print("\n测试3: 每个操作后都使用synchronize()")
        start_time = time.time()
        for _ in range(10):
            c = torch.matmul(a, b)
            torch.cuda.synchronize()  # 每次操作后都同步
        time_with_each_sync = time.time() - start_time
        print(f"  测量时间: {time_with_each_sync:.4f}秒")
        
        return {
            'without_sync': time_without_sync,
            'with_sync': time_with_sync,
            'with_each_sync': time_with_each_sync
        }
    
    time_results = test_async_operations()
    
    # 3. 演示多个流中的同步
    print("\n3. CUDA流中的同步")
    print("-"*40)
    
    def test_cuda_streams():
        """测试CUDA流中的同步"""
        print("创建两个CUDA流...")
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        
        # 创建测试数据
        size = 5000
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.zeros(size, size, device=device)
        
        # 在流1中执行操作
        with torch.cuda.stream(stream1):
            c1 = torch.matmul(a, b)
        
        # 在流2中执行操作
        with torch.cuda.stream(stream2):
            c2 = torch.matmul(b, a)
        
        print("操作已提交到不同流，现在等待流1完成...")
        stream1.synchronize()  # 只等待流1完成
        print("流1操作已完成")
        
        print("等待流2完成...")
        stream2.synchronize()  # 只等待流2完成
        print("流2操作已完成")
        
        print("使用torch.cuda.synchronize()等待所有流完成...")
        torch.cuda.synchronize()  # 等待所有流完成
        print("所有CUDA操作已完成")
    
    test_cuda_streams()
    
    # 4. 演示内存拷贝中的同步
    print("\n4. 内存拷贝中的同步")
    print("-"*40)
    
    def test_memory_copy():
        """测试内存拷贝中的同步"""
        size = 10000
        print(f"测试数据大小: {size}x{size} 矩阵")
        
        # 创建CPU数据
        cpu_data = torch.randn(size, size)
        
        # CPU到GPU拷贝
        print("\nCPU -> GPU 内存拷贝:")
        start_time = time.time()
        gpu_data = cpu_data.to(device)
        torch.cuda.synchronize()  # 等待拷贝完成
        copy_time = time.time() - start_time
        data_size_mb = cpu_data.element_size() * cpu_data.numel() / 1024**2
        print(f"  数据大小: {data_size_mb:.2f} MB")
        print(f"  拷贝时间: {copy_time*1000:.2f} ms")
        print(f"  带宽: {data_size_mb/copy_time:.2f} MB/s")
        
        # GPU到CPU拷贝
        print("\nGPU -> CPU 内存拷贝:")
        start_time = time.time()
        cpu_data_back = gpu_data.cpu()
        torch.cuda.synchronize()  # 等待拷贝完成
        copy_back_time = time.time() - start_time
        print(f"  拷贝时间: {copy_back_time*1000:.2f} ms")
        print(f"  带宽: {data_size_mb/copy_back_time:.2f} MB/s")
        
        return {
            'cpu_to_gpu_time': copy_time,
            'gpu_to_cpu_time': copy_back_time,
            'data_size_mb': data_size_mb
        }
    
    copy_results = test_memory_copy()
    
    # 5. 演示神经网络训练中的同步
    print("\n5. 神经网络训练中的同步")
    print("-"*40)
    
    def test_nn_training_sync():
        """测试神经网络训练中的同步"""
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(1000, 2000)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(2000, 1000)
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                return self.fc2(x)
        
        model = SimpleNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        batch_size = 32
        inputs = torch.randn(batch_size, 1000, device=device)
        targets = torch.randn(batch_size, 1000, device=device)
        
        # 训练步骤
        print(f"神经网络结构: Linear(1000,2000) -> ReLU -> Linear(2000,1000)")
        print(f"批量大小: {batch_size}")
        
        # 预热
        for _ in range(3):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # 测试不同同步策略的训练时间
        strategies = [
            ("无同步", False, False),
            ("前向后同步", True, False),
            ("反向传播后同步", False, True),
            ("每一步都同步", True, True)
        ]
        
        strategy_results = []
        
        for name, sync_after_forward, sync_after_backward in strategies:
            print(f"\n策略: {name}")
            
            start_time = time.time()
            for _ in range(20):
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(inputs)
                if sync_after_forward:
                    torch.cuda.synchronize()
                
                # 反向传播
                loss = criterion(outputs, targets)
                loss.backward()
                if sync_after_backward:
                    torch.cuda.synchronize()
                
                optimizer.step()
            
            # 等待所有操作完成
            torch.cuda.synchronize()
            training_time = time.time() - start_time
            avg_step_time = training_time / 20 * 1000  # 毫秒
            
            strategy_results.append({
                'name': name,
                'total_time': training_time,
                'avg_step_time': avg_step_time
            })
            
            print(f"  总训练时间: {training_time:.4f}秒")
            print(f"  平均每步时间: {avg_step_time:.2f}ms")
        
        return strategy_results
    
    training_results = test_nn_training_sync()
    
    # 6. 可视化结果
    print("\n6. 可视化同步对性能的影响")
    print("-"*40)
    
    # 准备可视化数据
    sync_strategies = ['无同步', '前向后同步', '反向后同步', '每一步同步']
    sync_times = [r['avg_step_time'] for r in training_results]
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 子图1: 同步策略对训练时间的影响
    bars = axes[0].bar(sync_strategies, sync_times, color=['skyblue', 'lightgreen', 'lightcoral', 'orange'])
    axes[0].set_xlabel('同步策略')
    axes[0].set_ylabel('平均每步时间 (ms)')
    axes[0].set_title('不同同步策略的训练时间对比')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, time_val in zip(bars, sync_times):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{time_val:.1f}', ha='center', va='bottom')
    
    # 子图2: 有无同步的时间测量差异
    measurement_scenarios = ['无同步测量', '最后同步', '每次同步']
    measurement_times = [
        time_results['without_sync'] * 1000,
        time_results['with_sync'] * 1000,
        time_results['with_each_sync'] * 1000
    ]
    
    bars2 = axes[1].bar(measurement_scenarios, measurement_times, color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[1].set_xlabel('测量策略')
    axes[1].set_ylabel('总时间 (ms)')
    axes[1].set_title('同步对时间测量的影响')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, time_val in zip(bars2, measurement_times):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{time_val:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('cuda_synchronize_impact.png', dpi=150, bbox_inches='tight')
    print("同步影响图已保存为 'cuda_synchronize_impact.png'")
    plt.show()
    
    return {
        'time_results': time_results,
        'copy_results': copy_results,
        'training_results': training_results
    }

def main():
    """主函数"""
    print("开始演示torch.cuda.synchronize()的作用...")
    results = demonstrate_cuda_synchronize()
    
    if results is not None:
        print("\n" + "="*60)
        print("演示完成!")
        print("="*60)
        
        print("\n关键结论:")
        print("1. torch.cuda.synchronize() 用于同步CPU和GPU")
        print("2. 确保所有CUDA操作完成后再继续执行CPU代码")
        print("3. 在以下情况特别重要:")
        print("   - 精确测量CUDA操作时间")
        print("   - 多流并行编程")
        print("   - 内存拷贝操作")
        print("   - 调试CUDA程序")
        print("4. 过度使用会降低性能，适当使用可确保正确性")
    else:
        print("演示未完成，需要CUDA设备支持")

if __name__ == "__main__":
    main()
