"""
MPS设备性能深度分析
准确测试MPS设备性能，找出瓶颈并优化
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def setup_test_environment():
    """设置测试环境"""
    print("MPS设备性能深度分析")
    print("="*60)
    
    # 检查MPS可用性
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"MPS可用: {mps_available}")
    print(f"CUDA可用: {cuda_available}")
    
    if mps_available:
        # 设置MPS设备
        device = torch.device("mps")
        
        # 获取MPS设备信息
        print("\nMPS设备信息:")
        if hasattr(torch.mps, 'current_allocated_memory'):
            allocated = torch.mps.current_allocated_memory() / 1024**2
            print(f"  已分配内存: {allocated:.2f} MB")
        
        # 设置MPS优化参数
        if hasattr(torch.mps, 'set_per_process_memory_fraction'):
            torch.mps.set_per_process_memory_fraction(0.9)
            print("  设置内存限制为90%")
        
        torch.set_default_dtype(torch.float32)
        print("  设置默认数据类型为float32")
        
        return device, "mps"
    
    # 回退到CPU
    device = torch.device("cpu")
    print(f"\n使用CPU设备: {device}")
    
    # 设置CPU优化
    torch.set_num_threads(torch.get_num_threads())
    print(f"  CPU线程数: {torch.get_num_threads()}")
    
    return device, "cpu"

def test_matrix_multiplication(device, device_type, sizes: List[Tuple[int, int, int]]):
    """测试矩阵乘法性能"""
    print("\n" + "="*60)
    print("矩阵乘法性能测试")
    print("="*60)
    
    results = []
    
    for m, n, k in sizes:
        print(f"\n测试矩阵大小: A({m}x{k}) x B({k}x{n}) = C({m}x{n})")
        
        # 创建随机矩阵
        A_cpu = torch.randn(m, k)
        B_cpu = torch.randn(k, n)
        
        # 测试CPU性能
        start_time = time.time()
        C_cpu = torch.matmul(A_cpu, B_cpu)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        cpu_time = time.time() - start_time
        cpu_flops = 2 * m * n * k / cpu_time / 1e9  # GFLOPS
        
        print(f"  CPU时间: {cpu_time*1000:.2f} ms, 性能: {cpu_flops:.2f} GFLOPS")
        
        # 如果设备不是CPU，测试设备性能
        if device_type != "cpu":
            A_device = A_cpu.to(device)
            B_device = B_cpu.to(device)
            
            # 预热
            for _ in range(3):
                _ = torch.matmul(A_device, B_device)
            
            # 同步
            if device_type == "mps":
                torch.mps.synchronize()
            elif device_type == "cuda":
                torch.cuda.synchronize()
            
            # 正式测试
            start_time = time.time()
            for _ in range(10):
                C_device = torch.matmul(A_device, B_device)
            
            if device_type == "mps":
                torch.mps.synchronize()
            elif device_type == "cuda":
                torch.cuda.synchronize()
            
            device_time = (time.time() - start_time) / 10
            device_flops = 2 * m * n * k / device_time / 1e9
            
            print(f"  {device_type.upper()}时间: {device_time*1000:.2f} ms, 性能: {device_flops:.2f} GFLOPS")
            print(f"  加速比: {cpu_time/device_time:.2f}x")
            
            results.append({
                'size': (m, n, k),
                'cpu_time': cpu_time,
                'device_time': device_time,
                'cpu_flops': cpu_flops,
                'device_flops': device_flops,
                'speedup': cpu_time / device_time
            })
        else:
            results.append({
                'size': (m, n, k),
                'cpu_time': cpu_time,
                'cpu_flops': cpu_flops
            })
    
    return results

def test_neural_network_forward(device, device_type, batch_sizes: List[int]):
    """测试神经网络前向传播性能"""
    print("\n" + "="*60)
    print("神经网络前向传播性能测试")
    print("="*60)
    
    class TestNet(nn.Module):
        def __init__(self, input_size=256, hidden_size=512, output_size=10):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_size, hidden_size)
            self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.relu3(self.fc3(x))
            return self.fc4(x)
    
    model = TestNet()
    model.to(device)
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n测试批次大小: {batch_size}")
        
        # 创建输入
        input_cpu = torch.randn(batch_size, 256)
        
        # 测试CPU性能
        model_cpu = TestNet()
        start_time = time.time()
        with torch.no_grad():
            output_cpu = model_cpu(input_cpu)
        cpu_time = time.time() - start_time
        
        print(f"  CPU前向传播时间: {cpu_time*1000:.2f} ms")
        
        # 如果设备不是CPU，测试设备性能
        if device_type != "cpu":
            input_device = input_cpu.to(device)
            
            # 预热
            for _ in range(3):
                with torch.no_grad():
                    _ = model(input_device)
            
            # 同步
            if device_type == "mps":
                torch.mps.synchronize()
            
            # 正式测试
            start_time = time.time()
            for _ in range(10):
                with torch.no_grad():
                    output_device = model(input_device)
            
            if device_type == "mps":
                torch.mps.synchronize()
            
            device_time = (time.time() - start_time) / 10
            
            print(f"  {device_type.upper()}前向传播时间: {device_time*1000:.2f} ms")
            print(f"  加速比: {cpu_time/device_time:.2f}x")
            
            results.append({
                'batch_size': batch_size,
                'cpu_time': cpu_time,
                'device_time': device_time,
                'speedup': cpu_time / device_time
            })
        else:
            results.append({
                'batch_size': batch_size,
                'cpu_time': cpu_time
            })
    
    return results

def test_data_transfer(device, device_type, data_sizes: List[Tuple[int, ...]]):
    """测试数据传输性能"""
    print("\n" + "="*60)
    print("CPU到设备数据传输性能测试")
    print("="*60)
    
    results = []
    
    for size in data_sizes:
        print(f"\n测试数据大小: {size}")
        
        # 创建CPU数据
        data_cpu = torch.randn(*size)
        
        # 测试CPU到设备传输时间
        if device_type != "cpu":
            start_time = time.time()
            data_device = data_cpu.to(device)
            
            if device_type == "mps":
                torch.mps.synchronize()
            elif device_type == "cuda":
                torch.cuda.synchronize()
            
            transfer_time = time.time() - start_time
            data_size_mb = data_cpu.element_size() * data_cpu.numel() / 1024**2
            
            print(f"  数据大小: {data_size_mb:.2f} MB")
            print(f"  CPU -> {device_type.upper()} 传输时间: {transfer_time*1000:.2f} ms")
            print(f"  传输带宽: {data_size_mb/transfer_time:.2f} MB/s")
            
            # 测试设备到CPU传输时间
            start_time = time.time()
            data_back = data_device.cpu()
            transfer_back_time = time.time() - start_time
            
            print(f"  {device_type.upper()} -> CPU 传输时间: {transfer_back_time*1000:.2f} ms")
            print(f"  回传带宽: {data_size_mb/transfer_back_time:.2f} MB/s")
            
            results.append({
                'size': size,
                'data_size_mb': data_size_mb,
                'to_device_time': transfer_time,
                'to_cpu_time': transfer_back_time,
                'to_device_bandwidth': data_size_mb / transfer_time,
                'to_cpu_bandwidth': data_size_mb / transfer_back_time
            })
    
    return results

def test_mixed_precision(device, device_type, batch_size=32):
    """测试混合精度性能"""
    print("\n" + "="*60)
    print("混合精度性能测试")
    print("="*60)
    
    if device_type == "cpu":
        print("CPU不支持混合精度，跳过测试")
        return None
    
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(256, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)
    
    model = TestModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 创建测试数据
    inputs = torch.randn(batch_size, 256).to(device)
    targets = torch.randn(batch_size, 10).to(device)
    
    results = []
    
    # 测试float32精度
    print("\n测试float32精度:")
    model.float()
    
    # 预热
    for _ in range(3):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    if device_type == "mps":
        torch.mps.synchronize()
    
    # 正式测试
    start_time = time.time()
    for _ in range(20):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    if device_type == "mps":
        torch.mps.synchronize()
    
    fp32_time = (time.time() - start_time) / 20
    print(f"  float32训练步平均时间: {fp32_time*1000:.2f} ms")
    
    # 测试混合精度（如果支持）
    if device_type == "mps" and hasattr(torch, 'autocast'):
        print("\n测试混合精度:")
        
        # 使用混合精度
        scaler = torch.amp.GradScaler('mps') if hasattr(torch.amp, 'GradScaler') else None
        
        start_time = time.time()
        for _ in range(20):
            optimizer.zero_grad()
            
            with torch.autocast(device_type='mps', dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        
        if device_type == "mps":
            torch.mps.synchronize()
        
        mixed_time = (time.time() - start_time) / 20
        speedup = fp32_time / mixed_time
        
        print(f"  混合精度训练步平均时间: {mixed_time*1000:.2f} ms")
        print(f"  加速比: {speedup:.2f}x")
        
        results = {
            'fp32_time': fp32_time,
            'mixed_time': mixed_time,
            'speedup': speedup
        }
    
    return results

def analyze_performance_bottlenecks(device_type, test_results):
    """分析性能瓶颈"""
    print("\n" + "="*60)
    print("性能瓶颈分析")
    print("="*60)
    
    bottlenecks = []
    
    if device_type == "mps":
        print("MPS设备常见性能瓶颈:")
        print("1. 小矩阵运算: MPS适合大规模并行计算，小矩阵可能表现不佳")
        print("2. 数据传输: CPU和GPU之间的数据传输开销")
        print("3. 内核启动开销: 每个操作都有固定开销")
        print("4. 内存带宽: 内存访问模式影响性能")
        print("5. 软件优化: PyTorch MPS后端可能不如CUDA优化充分")
        
        # 分析矩阵乘法结果
        if 'matrix_mult' in test_results:
            mat_results = test_results['matrix_mult']
            for result in mat_results:
                if 'speedup' in result and result['speedup'] < 1:
                    bottlenecks.append(f"矩阵大小{result['size']}: MPS比CPU慢{1/result['speedup']:.2f}倍")
        
        # 分析神经网络结果
        if 'nn_forward' in test_results:
            nn_results = test_results['nn_forward']
            for result in nn_results:
                if 'speedup' in result and result['speedup'] < 1:
                    bottlenecks.append(f"批次大小{result['batch_size']}: MPS前向传播比CPU慢{1/result['speedup']:.2f}倍")
    
    return bottlenecks

def visualize_results(test_results, device_type):
    """可视化测试结果"""
    if not test_results:
        return
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 矩阵乘法性能
    if 'matrix_mult' in test_results:
        mat_results = test_results['matrix_mult']
        if mat_results and 'speedup' in mat_results[0]:
            sizes = [f"{m}×{k}×{n}" for m, n, k in [r['size'] for r in mat_results]]
            speedups = [r['speedup'] for r in mat_results]
            
            axes[0, 0].bar(sizes, speedups, color=['red' if s < 1 else 'green' for s in speedups])
            axes[0, 0].axhline(y=1, color='black', linestyle='--', linewidth=1)
            axes[0, 0].set_xlabel('矩阵大小 (M×K×N)')
            axes[0, 0].set_ylabel('加速比 (CPU时间/设备时间)')
            axes[0, 0].set_title('矩阵乘法加速比')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 神经网络前向传播性能
    if 'nn_forward' in test_results:
        nn_results = test_results['nn_forward']
        if nn_results and 'speedup' in nn_results[0]:
            batch_sizes = [r['batch_size'] for r in nn_results]
            speedups = [r['speedup'] for r in nn_results]
            
            axes[0, 1].plot(batch_sizes, speedups, 'o-', linewidth=2, markersize=8)
            axes[0, 1].axhline(y=1, color='black', linestyle='--', linewidth=1)
            axes[0, 1].set_xlabel('批次大小')
            axes[0, 1].set_ylabel('加速比 (CPU时间/设备时间)')
            axes[0, 1].set_title('神经网络前向传播加速比')
            axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 数据传输带宽
    if 'data_transfer' in test_results:
        data_results = test_results['data_transfer']
        if data_results:
            sizes = [f"{r['size']}" for r in data_results]
            to_device_bw = [r['to_device_bandwidth'] for r in data_results]
            to_cpu_bw = [r['to_cpu_bandwidth'] for r in data_results]
            
            x = np.arange(len(sizes))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, to_device_bw, width, label='CPU->设备', color='skyblue')
            axes[1, 0].bar(x + width/2, to_cpu_bw, width, label='设备->CPU', color='lightcoral')
            axes[1, 0].set_xlabel('数据大小')
            axes[1, 0].set_ylabel('带宽 (MB/s)')
            axes[1, 0].set_title('数据传输带宽')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(sizes, rotation=45)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 混合精度性能
    if 'mixed_precision' in test_results:
        mp_results = test_results['mixed_precision']
        if mp_results:
            labels = ['float32', '混合精度']
            times = [mp_results['fp32_time']*1000, mp_results['mixed_time']*1000]
            speedup = mp_results['speedup']
            
            bars = axes[1, 1].bar(labels, times, color=['blue', 'orange'])
            axes[1, 1].set_xlabel('精度模式')
            axes[1, 1].set_ylabel('训练步时间 (ms)')
            axes[1, 1].set_title(f'混合精度性能 (加速比: {speedup:.2f}x)')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{device_type}_performance_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n性能分析图已保存为: {device_type}_performance_analysis.png")
    plt.show()

def main():
    """主函数"""
    # 设置测试环境
    device, device_type = setup_test_environment()
    
    test_results = {}
    
    # 1. 测试矩阵乘法性能
    matrix_sizes = [
        (32, 32, 32),      # 小矩阵
        (128, 128, 128),   # 中等矩阵
        (512, 512, 512),   # 大矩阵
        (1024, 1024, 1024) # 超大矩阵
    ]
    
    mat_results = test_matrix_multiplication(device, device_type, matrix_sizes)
    test_results['matrix_mult'] = mat_results
    
    # 2. 测试神经网络前向传播性能
    batch_sizes = [1, 8, 16, 32, 64, 128]
    nn_results = test_neural_network_forward(device, device_type, batch_sizes)
    test_results['nn_forward'] = nn_results
    
    # 3. 测试数据传输性能
    if device_type != "cpu":
        data_sizes = [
            (1000, 1000),    # 1M元素
            (2000, 2000),    # 4M元素
            (3000, 3000)     # 9M元素
        ]
        transfer_results = test_data_transfer(device, device_type, data_sizes)
        test_results['data_transfer'] = transfer_results
    
    # 4. 测试混合精度性能
    if device_type != "cpu":
        mp_results = test_mixed_precision(device, device_type, batch_size=32)
        test_results['mixed_precision'] = mp_results
    
    # 5. 分析性能瓶颈
    bottlenecks = analyze_performance_bottlenecks(device_type, test_results)
    
    # 6. 可视化结果
    visualize_results(test_results, device_type)
    
    # 7. 打印总结
    print("\n" + "="*60)
    print("性能测试总结")
    print("="*60)
    
    if device_type == "mps":
        print("MPS设备性能分析:")
        
        # 检查是否有MPS比CPU慢的情况
        mps_slower_cases = []
        
        for test_name, results in test_results.items():
            if test_name in ['matrix_mult', 'nn_forward'] and results:
                for result in results:
                    if 'speedup' in result and result['speedup'] < 1:
                        mps_slower_cases.append(f"{test_name}: 加速比 {result['speedup']:.2f}x")
        
        if mps_slower_cases:
            print("⚠ 发现MPS比CPU慢的情况:")
            for case in mps_slower_cases:
                print(f"  - {case}")
            
            print("\n可能的原因:")
            print("1. 计算规模太小: MPS适合大规模并行计算")
            print("2. 数据传输开销: CPU-GPU数据传输时间超过计算时间")
            print("3. 内核启动开销: 每个MPS操作都有固定开销")
            print("4. 软件优化不足: PyTorch MPS后端可能不如CUDA优化充分")
            print("5. 硬件限制: 某些Apple Silicon芯片的GPU性能有限")
            
            print("\n优化建议:")
            print("1. 增加批次大小: 使用更大的批量进行计算")
            print("2. 减少数据传输: 在设备上创建数据，避免频繁传输")
            print("3. 使用更大模型: 大规模计算更能发挥MPS优势")
            print("4. 合并操作: 将多个小操作合并为大操作")
            print("5. 使用float32: 避免混合精度的转换开销")
        else:
            print("✓ MPS在所有测试中都比CPU快")
    
    return test_results, bottlenecks

if __name__ == "__main__":
    test_results, bottlenecks = main()
