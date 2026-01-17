import numpy as np
import time
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
import sobol_seq  # 用于准蒙特卡洛
import warnings
warnings.filterwarnings("ignore")

# 尝试导入cupy（如果可用）
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✓ CuPy可用，将使用GPU加速")
except ImportError:
    CUPY_AVAILABLE = False
    print("⚠️ CuPy不可用，将使用CPU计算")

class ImprovedMonteCarlo:
    """改进版的蒙特卡洛方法实现，包含多种优化技术"""
    
    def __init__(self, x0, T, r, sigma):
        """
        初始化改进版蒙特卡洛模拟器
        
        参数:
        x0: 初始条件 [1.0, 0.5, 1.0, 0.5, ...] (100维)
        T: 到期时间
        r: 无风险利率
        sigma: 波动率
        """
        self.x0 = np.array(x0)
        self.D = len(x0)  # 维度
        self.T = T
        self.r = r
        self.sigma = sigma
        
        # 预计算常量
        self.drift = (r - 0.5 * sigma**2) * T
        self.diffusion = sigma * np.sqrt(T)
        
        # 解析解
        self.analytical_solution = np.exp((r + sigma**2) * T) * np.sum(self.x0**2)
    
    def standard_mc(self, n_simulations=1000000):
        """
        标准蒙特卡洛实现（原始版本）
        
        参数:
        n_simulations: 模拟路径数量
        
        返回:
        mc_price: 蒙特卡洛估计值
        mc_time: 计算时间
        mc_std: 标准误差
        mc_error: 相对误差（相对于解析解）
        """
        print("运行标准蒙特卡洛方法...")
        start_time = time.time()
        
        n_simulations = min(n_simulations, 1000000)
        
        try:
            # 生成随机数
            z = np.random.randn(n_simulations, self.D)
            
            # 计算终端资产价格
            X_T = self.x0 * np.exp(self.drift + self.diffusion * z)
            
            # 计算终端条件
            terminal_values = np.sum(X_T**2, axis=1)
            
            # 计算结果
            mc_price = np.mean(terminal_values)
            mc_std = np.std(terminal_values) / np.sqrt(n_simulations)
            mc_error = abs(mc_price - self.analytical_solution) / self.analytical_solution
            
            mc_time = time.time() - start_time
            
            return mc_price, mc_time, mc_std, mc_error
            
        except MemoryError:
            # 分批处理
            print("内存不足，使用分批处理...")
            batch_size = 10000
            n_batches = n_simulations // batch_size
            
            terminal_sum = 0
            terminal_sq_sum = 0
            
            for i in range(n_batches):
                z_batch = np.random.randn(batch_size, self.D)
                X_T_batch = self.x0 * np.exp(self.drift + self.diffusion * z_batch)
                terminal_batch = np.sum(X_T_batch**2, axis=1)
                
                terminal_sum += np.sum(terminal_batch)
                terminal_sq_sum += np.sum(terminal_batch**2)
            
            mc_price = terminal_sum / (n_batches * batch_size)
            variance = (terminal_sq_sum - terminal_sum**2 / (n_batches * batch_size)) / (n_batches * batch_size - 1)
            mc_std = np.sqrt(variance / (n_batches * batch_size))
            mc_error = abs(mc_price - self.analytical_solution) / self.analytical_solution
            
            mc_time = time.time() - start_time
            
            return mc_price, mc_time, mc_std, mc_error
    
    def quasi_mc(self, n_simulations=1000000):
        """
        准蒙特卡洛实现（使用Sobol序列减少方差）
        
        参数:
        n_simulations: 模拟路径数量
        
        返回:
        mc_price: 蒙特卡洛估计值
        mc_time: 计算时间
        mc_std: 标准误差
        mc_error: 相对误差（相对于解析解）
        """
        print("运行准蒙特卡洛方法...")
        start_time = time.time()
        
        n_simulations = min(n_simulations, 1000000)
        
        try:
            # 生成Sobol序列并转换为正态分布
            sobol_points = sobol_seq.i4_sobol_generate(self.D, n_simulations)
            # 使用逆变换采样将均匀分布转换为正态分布
            z = norminv(sobol_points)
            
            # 计算终端资产价格
            X_T = self.x0 * np.exp(self.drift + self.diffusion * z)
            
            # 计算终端条件
            terminal_values = np.sum(X_T**2, axis=1)
            
            # 计算结果
            mc_price = np.mean(terminal_values)
            mc_std = np.std(terminal_values) / np.sqrt(n_simulations)
            mc_error = abs(mc_price - self.analytical_solution) / self.analytical_solution
            
            mc_time = time.time() - start_time
            
            return mc_price, mc_time, mc_std, mc_error
            
        except MemoryError:
            # 分批处理
            print("内存不足，使用分批处理...")
            batch_size = 10000
            n_batches = n_simulations // batch_size
            
            terminal_sum = 0
            terminal_sq_sum = 0
            
            for i in range(n_batches):
                # 生成批次的Sobol点
                sobol_batch = sobol_seq.i4_sobol_generate(self.D, batch_size)
                z_batch = norminv(sobol_batch)
                
                X_T_batch = self.x0 * np.exp(self.drift + self.diffusion * z_batch)
                terminal_batch = np.sum(X_T_batch**2, axis=1)
                
                terminal_sum += np.sum(terminal_batch)
                terminal_sq_sum += np.sum(terminal_batch**2)
            
            mc_price = terminal_sum / (n_batches * batch_size)
            variance = (terminal_sq_sum - terminal_sum**2 / (n_batches * batch_size)) / (n_batches * batch_size - 1)
            mc_std = np.sqrt(variance / (n_batches * batch_size))
            mc_error = abs(mc_price - self.analytical_solution) / self.analytical_solution
            
            mc_time = time.time() - start_time
            
            return mc_price, mc_time, mc_std, mc_error
    
    def parallel_mc(self, n_simulations=1000000, n_jobs=-1):
        """
        并行蒙特卡洛实现（使用joblib）
        
        参数:
        n_simulations: 模拟路径数量
        n_jobs: 并行工作数，-1表示使用所有CPU核心
        
        返回:
        mc_price: 蒙特卡洛估计值
        mc_time: 计算时间
        mc_std: 标准误差
        mc_error: 相对误差（相对于解析解）
        """
        print(f"运行并行蒙特卡洛方法（{n_jobs}个并行工作）...")
        start_time = time.time()
        
        n_simulations = min(n_simulations, 1000000)
        
        # 计算每个job的模拟数量
        n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        simulations_per_job = n_simulations // n_jobs
        
        # 并行计算函数
        def simulate_batch(batch_id):
            """单个批次的模拟函数"""
            # 为每个批次设置不同的随机种子
            np.random.seed(batch_id)
            
            # 生成随机数
            z = np.random.randn(simulations_per_job, self.D)
            
            # 计算终端资产价格
            X_T = self.x0 * np.exp(self.drift + self.diffusion * z)
            
            # 计算终端条件
            terminal_values = np.sum(X_T**2, axis=1)
            
            return terminal_values
        
        # 执行并行计算
        results = Parallel(n_jobs=n_jobs)(
            delayed(simulate_batch)(i) for i in range(n_jobs)
        )
        
        # 合并结果
        terminal_values = np.concatenate(results)
        
        # 计算结果
        mc_price = np.mean(terminal_values)
        mc_std = np.std(terminal_values) / np.sqrt(n_simulations)
        mc_error = abs(mc_price - self.analytical_solution) / self.analytical_solution
        
        mc_time = time.time() - start_time
        
        return mc_price, mc_time, mc_std, mc_error
    
    def gpu_mc(self, n_simulations=1000000):
        """
        GPU加速的蒙特卡洛实现（使用CuPy）
        
        参数:
        n_simulations: 模拟路径数量
        
        返回:
        mc_price: 蒙特卡洛估计值
        mc_time: 计算时间
        mc_std: 标准误差
        mc_error: 相对误差（相对于解析解）
        """
        if not CUPY_AVAILABLE:
            print("⚠️ CuPy不可用，将使用标准蒙特卡洛方法")
            return self.standard_mc(n_simulations)
        
        print("运行GPU加速的蒙特卡洛方法...")
        start_time = time.time()
        
        n_simulations = min(n_simulations, 1000000)
        
        try:
            # 使用CuPy在GPU上生成随机数
            with cp.cuda.Device(0):  # 使用第一个GPU
                # 转换参数到CuPy数组
                x0_cp = cp.array(self.x0)
                drift_cp = cp.array(self.drift)
                diffusion_cp = cp.array(self.diffusion)
                
                # 生成随机数
                z = cp.random.randn(n_simulations, self.D)
                
                # 计算终端资产价格
                X_T = x0_cp * cp.exp(drift_cp + diffusion_cp * z)
                
                # 计算终端条件
                terminal_values = cp.sum(X_T**2, axis=1)
                
                # 计算结果
                mc_price = float(terminal_values.mean())
                mc_std = float(terminal_values.std()) / np.sqrt(n_simulations)
                
            mc_error = abs(mc_price - self.analytical_solution) / self.analytical_solution
            mc_time = time.time() - start_time
            
            return mc_price, mc_time, mc_std, mc_error
            
        except (MemoryError, cp.cuda.memory.OutOfMemoryError):
            print("GPU内存不足，使用CPU分批处理...")
            batch_size = 10000
            n_batches = n_simulations // batch_size
            
            terminal_sum = 0
            terminal_sq_sum = 0
            
            with cp.cuda.Device(0):
                x0_cp = cp.array(self.x0)
                drift_cp = cp.array(self.drift)
                diffusion_cp = cp.array(self.diffusion)
                
                for i in range(n_batches):
                    # 生成批次随机数
                    z_batch = cp.random.randn(batch_size, self.D)
                    
                    # 计算终端资产价格
                    X_T_batch = x0_cp * cp.exp(drift_cp + diffusion_cp * z_batch)
                    
                    # 计算终端条件
                    terminal_batch = cp.sum(X_T_batch**2, axis=1)
                    
                    # 转换到CPU并累加
                    terminal_batch_np = cp.asnumpy(terminal_batch)
                    terminal_sum += np.sum(terminal_batch_np)
                    terminal_sq_sum += np.sum(terminal_batch_np**2)
            
            mc_price = terminal_sum / (n_batches * batch_size)
            variance = (terminal_sq_sum - terminal_sum**2 / (n_batches * batch_size)) / (n_batches * batch_size - 1)
            mc_std = np.sqrt(variance / (n_batches * batch_size))
            mc_error = abs(mc_price - self.analytical_solution) / self.analytical_solution
            
            mc_time = time.time() - start_time
            
            return mc_price, mc_time, mc_std, mc_error
    
    def optimized_mc(self, n_simulations=1000000):
        """
        综合优化的蒙特卡洛实现（结合多种优化技术）
        
        参数:
        n_simulations: 模拟路径数量
        
        返回:
        mc_price: 蒙特卡洛估计值
        mc_time: 计算时间
        mc_std: 标准误差
        mc_error: 相对误差（相对于解析解）
        """
        print("运行综合优化的蒙特卡洛方法...")
        start_time = time.time()
        
        n_simulations = min(n_simulations, 1000000)
        
        # 选择最佳优化组合
        if CUPY_AVAILABLE:
            # 如果GPU可用，优先使用GPU加速
            result = self.gpu_mc(n_simulations)
        else:
            # 否则使用并行计算
            result = self.parallel_mc(n_simulations)
        
        mc_time = time.time() - start_time
        
        return result[0], mc_time, result[2], result[3]

def norminv(u):
    """
    使用Box-Muller变换将均匀分布转换为正态分布
    
    参数:
    u: 均匀分布的随机数（[0,1]区间）
    
    返回:
    z: 标准正态分布的随机数
    """
    # 处理边界情况
    u = np.clip(u, 1e-10, 1 - 1e-10)
    
    # Box-Muller变换
    z = np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * u)
    
    return z

def compare_methods():
    """比较不同蒙特卡洛方法的性能"""
    print("=" * 80)
    print("             蒙特卡洛方法性能比较")
    print("=" * 80)
    
    # 参数设置
    D = 100  # 维度
    x0 = [1.0, 0.5] * int(D / 2)  # 初始条件
    T = 1.0  # 时间范围
    r = 0.05  # 无风险利率
    sigma = 0.4  # 波动率
    n_simulations = 1000000  # 模拟数量
    
    # 创建改进版蒙特卡洛实例
    mc = ImprovedMonteCarlo(x0, T, r, sigma)
    
    print(f"\n参数设置:")
    print(f"- 维度: {D}维")
    print(f"- 初始条件: {x0[:4]}...")
    print(f"- 模拟路径数: {n_simulations:,}")
    print(f"- 解析解: {mc.analytical_solution:.6f}")
    
    # 存储结果
    methods = []
    prices = []
    times = []
    stds = []
    errors = []
    
    # 1. 标准蒙特卡洛
    methods.append("标准蒙特卡洛")
    price, time_taken, std, error = mc.standard_mc(n_simulations)
    prices.append(price)
    times.append(time_taken)
    stds.append(std)
    errors.append(error)
    print(f"\n✓ 标准蒙特卡洛完成: 价格={price:.6f}, 时间={time_taken:.2f}s, 标准误差={std:.6f}, 相对误差={error:.6f}")
    
    # 2. 准蒙特卡洛
    methods.append("准蒙特卡洛 (Sobol)")
    price, time_taken, std, error = mc.quasi_mc(n_simulations)
    prices.append(price)
    times.append(time_taken)
    stds.append(std)
    errors.append(error)
    print(f"✓ 准蒙特卡洛完成: 价格={price:.6f}, 时间={time_taken:.2f}s, 标准误差={std:.6f}, 相对误差={error:.6f}")
    
    # 3. 并行蒙特卡洛
    methods.append("并行蒙特卡洛")
    price, time_taken, std, error = mc.parallel_mc(n_simulations)
    prices.append(price)
    times.append(time_taken)
    stds.append(std)
    errors.append(error)
    print(f"✓ 并行蒙特卡洛完成: 价格={price:.6f}, 时间={time_taken:.2f}s, 标准误差={std:.6f}, 相对误差={error:.6f}")
    
    # 4. GPU加速蒙特卡洛（如果可用）
    if CUPY_AVAILABLE:
        methods.append("GPU加速蒙特卡洛")
        price, time_taken, std, error = mc.gpu_mc(n_simulations)
        prices.append(price)
        times.append(time_taken)
        stds.append(std)
        errors.append(error)
        print(f"✓ GPU加速蒙特卡洛完成: 价格={price:.6f}, 时间={time_taken:.2f}s, 标准误差={std:.6f}, 相对误差={error:.6f}")
    
    # 5. 综合优化蒙特卡洛
    methods.append("综合优化蒙特卡洛")
    price, time_taken, std, error = mc.optimized_mc(n_simulations)
    prices.append(price)
    times.append(time_taken)
    stds.append(std)
    errors.append(error)
    print(f"✓ 综合优化蒙特卡洛完成: 价格={price:.6f}, 时间={time_taken:.2f}s, 标准误差={std:.6f}, 相对误差={error:.6f}")
    
    # 性能比较报告
    print("\n" + "=" * 80)
    print("                 性能比较报告")
    print("=" * 80)
    
    print(f"{'方法':<25} {'价格':<12} {'时间(s)':<10} {'标准误差':<12} {'相对误差':<12} {'加速比':<10}")
    print("-" * 80)
    
    base_time = times[0]  # 以标准蒙特卡洛为基准
    for i, method in enumerate(methods):
        speedup = base_time / times[i]
        print(f"{method:<25} {prices[i]:<12.6f} {times[i]:<10.2f} {stds[i]:<12.6f} {errors[i]:<12.6f} {speedup:<10.2f}x")
    
    # 可视化结果
    visualize_results(methods, prices, times, stds, errors, mc.analytical_solution)
    
    return {
        'methods': methods,
        'prices': prices,
        'times': times,
        'stds': stds,
        'errors': errors,
        'analytical_solution': mc.analytical_solution
    }

def visualize_results(methods, prices, times, stds, errors, analytical_solution):
    """可视化不同方法的性能比较"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 价格估计对比
    ax = axes[0, 0]
    ax.bar(range(len(methods)), prices, alpha=0.7, color=['blue', 'green', 'orange', 'red', 'purple'])
    ax.axhline(y=analytical_solution, color='black', linestyle='--', label=f'解析解: {analytical_solution:.2f}')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('价格估计')
    ax.set_title('价格估计对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 在柱子上添加数值
    for i, v in enumerate(prices):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    # 计算时间对比
    ax = axes[0, 1]
    bars = ax.bar(range(len(methods)), times, alpha=0.7, color=['blue', 'green', 'orange', 'red', 'purple'])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('计算时间 (秒)')
    ax.set_title('计算效率对比')
    ax.grid(True, alpha=0.3)
    
    # 在柱子上添加数值
    for i, v in enumerate(times):
        ax.text(i, v, f'{v:.1f}s', ha='center', va='bottom')
    
    # 相对误差对比
    ax = axes[1, 0]
    ax.bar(range(len(methods)), errors, alpha=0.7, color=['blue', 'green', 'orange', 'red', 'purple'])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('相对误差')
    ax.set_title('准确性对比')
    ax.grid(True, alpha=0.3)
    
    # 在柱子上添加数值
    for i, v in enumerate(errors):
        ax.text(i, v, f'{v:.6f}', ha='center', va='bottom')
    
    # 标准误差对比
    ax = axes[1, 1]
    ax.bar(range(len(methods)), stds, alpha=0.7, color=['blue', 'green', 'orange', 'red', 'purple'])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('标准误差')
    ax.set_title('稳定性对比')
    ax.grid(True, alpha=0.3)
    
    # 在柱子上添加数值
    for i, v in enumerate(stds):
        ax.text(i, v, f'{v:.6f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.suptitle('蒙特卡洛方法性能综合比较', fontsize=16, y=1.02)
    plt.savefig('mc_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✓ 可视化结果已保存到 mc_performance_comparison.png")

if __name__ == "__main__":
    # 运行方法比较
    results = compare_methods()
    
    print("\n" + "=" * 80)
    print("             比较分析完成")
    print("=" * 80)
    print("建议使用综合优化的蒙特卡洛方法，")
    print("它会根据硬件自动选择最佳的优化策略。")
