import numpy as np
import time
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

# 尝试导入sobol_seq（用于准蒙特卡洛）
try:
    import sobol_seq
    SOBOL_AVAILABLE = True
except ImportError:
    SOBOL_AVAILABLE = False
    print("⚠️ sobol_seq不可用，准蒙特卡洛方法将使用随机数替代")

# 尝试导入cupy（用于GPU加速）
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
        x0: 初始条件数组
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
        
        # 解析解（用于误差计算）
        self.analytical_solution = np.exp((r + sigma**2) * T) * np.sum(self.x0**2)
    
    def norminv(self, u):
        """将均匀分布转换为正态分布（逆变换采样）"""
        # 处理边界情况
        u = np.clip(u, 1e-10, 1 - 1e-10)
        # 使用逆正态累积分布函数
        from scipy import stats
        return stats.norm.ppf(u)
    
    def standard_mc(self, n_simulations=1000000):
        """标准蒙特卡洛实现"""
        print("运行标准蒙特卡洛方法...")
        start_time = time.time()
        
        n_simulations = min(n_simulations, 1000000)  # 限制模拟数量
        
        try:
            # 生成随机数
            z = np.random.randn(n_simulations, self.D)
            
            # 计算终端资产价格
            X_T = self.x0 * np.exp(self.drift + self.diffusion * z)
            
            # 计算终端条件：g(X_T) = sum(X_T^2)
            terminal_values = np.sum(X_T**2, axis=1)
            
            # 计算结果
            mc_price = np.mean(terminal_values)
            mc_std = np.std(terminal_values) / np.sqrt(n_simulations)
            mc_error = abs(mc_price - self.analytical_solution) / self.analytical_solution
            
            mc_time = time.time() - start_time
            
            return mc_price, mc_time, mc_std, mc_error
            
        except MemoryError:
            # 内存不足时使用分批处理
            return self._batch_processing(n_simulations, start_time)
    
    def quasi_mc(self, n_simulations=1000000):
        """准蒙特卡洛实现（使用低差异序列）"""
        if not SOBOL_AVAILABLE:
            print("⚠️ sobol_seq不可用，使用标准蒙特卡洛替代")
            return self.standard_mc(n_simulations)
            
        print("运行准蒙特卡洛方法...")
        start_time = time.time()
        
        n_simulations = min(n_simulations, 1000000)
        
        try:
            # 生成Sobol序列
            sobol_points = sobol_seq.i4_sobol_generate(self.D, n_simulations)
            
            # 转换为正态分布
            z = self.norminv(sobol_points)
            
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
            return self._batch_processing_quasi(n_simulations, start_time)
    
    def parallel_mc(self, n_simulations=1000000, n_jobs=-1):
        """并行蒙特卡洛实现"""
        print("运行并行蒙特卡洛方法...")
        start_time = time.time()
        
        n_simulations = min(n_simulations, 1000000)
        
        # 设置并行工作数
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        
        simulations_per_job = n_simulations // n_jobs
        
        def simulate_batch(batch_id):
            """单个批次的模拟函数"""
            np.random.seed(batch_id)  # 设置不同的随机种子
            z_batch = np.random.randn(simulations_per_job, self.D)
            X_T_batch = self.x0 * np.exp(self.drift + self.diffusion * z_batch)
            terminal_batch = np.sum(X_T_batch**2, axis=1)
            return terminal_batch
        
        # 执行并行计算
        try:
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
            
        except Exception as e:
            print(f"并行计算出错: {e}，使用标准方法")
            return self.standard_mc(n_simulations)
    
    def gpu_mc(self, n_simulations=1000000):
        """GPU加速的蒙特卡洛实现"""
        if not CUPY_AVAILABLE:
            print("⚠️ CuPy不可用，使用标准蒙特卡洛替代")
            return self.standard_mc(n_simulations)
            
        print("运行GPU加速的蒙特卡洛方法...")
        start_time = time.time()
        
        n_simulations = min(n_simulations, 1000000)
        
        try:
            with cp.cuda.Device(0):  # 使用第一个GPU
                # 转换参数到GPU
                x0_gpu = cp.array(self.x0)
                drift_gpu = cp.array(self.drift)
                diffusion_gpu = cp.array(self.diffusion)
                
                # 在GPU上生成随机数
                z_gpu = cp.random.randn(n_simulations, self.D)
                
                # 计算终端资产价格
                X_T_gpu = x0_gpu * cp.exp(drift_gpu + diffusion_gpu * z_gpu)
                
                # 计算终端条件
                terminal_values_gpu = cp.sum(X_T_gpu**2, axis=1)
                
                # 传输回CPU并计算结果
                terminal_values = cp.asnumpy(terminal_values_gpu)
                mc_price = np.mean(terminal_values)
                mc_std = np.std(terminal_values) / np.sqrt(n_simulations)
                mc_error = abs(mc_price - self.analytical_solution) / self.analytical_solution
                
                mc_time = time.time() - start_time
                
                return mc_price, mc_time, mc_std, mc_error
                
        except Exception as e:
            print(f"GPU计算出错: {e}，使用CPU分批处理")
            return self._batch_processing(n_simulations, start_time)
    
    def antithetic_mc(self, n_simulations=1000000):
        """对偶变量法蒙特卡洛（方差缩减技术）"""
        print("运行对偶变量法蒙特卡洛...")
        start_time = time.time()
        
        n_simulations = min(n_simulations, 1000000) // 2  # 每对需要两个路径
        
        try:
            # 生成主要路径
            z_primary = np.random.randn(n_simulations, self.D)
            
            # 生成对偶路径（取负号）
            z_antithetic = -z_primary
            
            # 计算两条路径的终端值
            X_T_primary = self.x0 * np.exp(self.drift + self.diffusion * z_primary)
            X_T_antithetic = self.x0 * np.exp(self.drift + self.diffusion * z_antithetic)
            
            terminal_primary = np.sum(X_T_primary**2, axis=1)
            terminal_antithetic = np.sum(X_T_antithetic**2, axis=1)
            
            # 取平均作为最终估计
            terminal_combined = 0.5 * (terminal_primary + terminal_antithetic)
            
            mc_price = np.mean(terminal_combined)
            mc_std = np.std(terminal_combined) / np.sqrt(n_simulations * 2)
            mc_error = abs(mc_price - self.analytical_solution) / self.analytical_solution
            
            mc_time = time.time() - start_time
            
            return mc_price, mc_time, mc_std, mc_error
            
        except MemoryError:
            return self._batch_processing(n_simulations * 2, start_time)
    
    def optimized_mc(self, n_simulations=1000000):
        """综合优化的蒙特卡洛方法（自动选择最佳策略）"""
        print("运行综合优化的蒙特卡洛方法...")
        start_time = time.time()
        
        # 根据可用资源选择最佳方法
        if CUPY_AVAILABLE:
            result = self.gpu_mc(n_simulations)
        else:
            # 使用对偶变量法（方差缩减效果好）
            result = self.antithetic_mc(n_simulations)
        
        mc_time = time.time() - start_time
        return result[0], mc_time, result[2], result[3]
    
    def _batch_processing(self, n_simulations, start_time):
        """分批处理避免内存溢出"""
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
            
            if (i + 1) % 10 == 0:
                print(f"处理进度: {i+1}/{n_batches} 批次")
        
        mc_price = terminal_sum / (n_batches * batch_size)
        variance = (terminal_sq_sum - terminal_sum**2 / (n_batches * batch_size)) / (n_batches * batch_size - 1)
        mc_std = np.sqrt(variance / (n_batches * batch_size))
        mc_error = abs(mc_price - self.analytical_solution) / self.analytical_solution
        
        mc_time = time.time() - start_time
        
        return mc_price, mc_time, mc_std, mc_error
    
    def _batch_processing_quasi(self, n_simulations, start_time):
        """准蒙特卡洛的分批处理"""
        if not SOBOL_AVAILABLE:
            return self._batch_processing(n_simulations, start_time)
            
        batch_size = 10000
        n_batches = n_simulations // batch_size
        
        terminal_sum = 0
        terminal_sq_sum = 0
        
        for i in range(n_batches):
            sobol_batch = sobol_seq.i4_sobol_generate(self.D, batch_size)
            z_batch = self.norminv(sobol_batch)
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

def compare_methods():
    """比较不同蒙特卡洛方法的性能"""
    print("=" * 80)
    print("             蒙特卡洛方法性能比较")
    print("=" * 80)
    
    # 参数设置（100维问题）
    D = 100
    x0 = [1.0, 0.5] * (D // 2)  # 初始条件
    T = 1.0
    r = 0.05
    sigma = 0.4
    n_simulations = 1000000
    
    # 创建改进版蒙特卡洛实例
    mc = ImprovedMonteCarlo(x0, T, r, sigma)
    
    print(f"参数设置:")
    print(f"- 维度: {D}维")
    print(f"- 初始条件: {x0[:4]}...")
    print(f"- 模拟路径数: {n_simulations:,}")
    print(f"- 解析解: {mc.analytical_solution:.6f}")
    print()
    
    # 存储结果
    methods = []
    prices = []
    times = []
    stds = []
    errors = []
    
    # 测试各种方法
    method_configs = [
        ("标准蒙特卡洛", "standard_mc"),
        ("对偶变量法", "antithetic_mc"),
        ("准蒙特卡洛", "quasi_mc"),
        ("并行蒙特卡洛", "parallel_mc"),
    ]
    
    # 添加GPU方法（如果可用）
    if CUPY_AVAILABLE:
        method_configs.append(("GPU加速", "gpu_mc"))
    
    method_configs.append(("综合优化", "optimized_mc"))
    
    for method_name, method_func in method_configs:
        print(f"正在运行{method_name}...")
        methods.append(method_name)
        
        try:
            if method_func == "parallel_mc":
                price, time_taken, std, error = getattr(mc, method_func)(n_simulations, n_jobs=-1)
            else:
                price, time_taken, std, error = getattr(mc, method_func)(n_simulations)
            
            prices.append(price)
            times.append(time_taken)
            stds.append(std)
            errors.append(error)
            
            print(f"✓ {method_name}完成: 价格={price:.6f}, 时间={time_taken:.2f}s, "
                  f"标准误差={std:.6f}, 相对误差={error:.6f}")
                  
        except Exception as e:
            print(f"✗ {method_name}执行失败: {e}")
            # 使用默认值继续
            prices.append(0)
            times.append(0)
            stds.append(0)
            errors.append(1)
    
    # 性能比较报告
    print("\n" + "=" * 80)
    print("                 性能比较报告")
    print("=" * 80)
    
    print(f"{'方法':<15} {'价格':<12} {'时间(s)':<10} {'标准误差':<12} {'相对误差':<12} {'加速比':<10}")
    print("-" * 80)
    
    base_time = times[0] if times[0] > 0 else 1  # 避免除零
    for i, method in enumerate(methods):
        speedup = base_time / times[i] if times[i] > 0 else 0
        print(f"{method:<15} {prices[i]:<12.6f} {times[i]:<10.2f} {stds[i]:<12.6f} {errors[i]:<12.6f} {speedup:<10.2f}x")
    
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
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 颜色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 1. 价格估计对比
    bars1 = ax1.bar(range(len(methods)), prices, color=colors[:len(methods)], alpha=0.7)
    ax1.axhline(y=analytical_solution, color='red', linestyle='--', 
                label=f'解析解: {analytical_solution:.2f}', linewidth=2)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45)
    ax1.set_ylabel('价格估计')
    ax1.set_title('价格估计对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 在柱子上添加数值
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    # 2. 计算时间对比
    bars2 = ax2.bar(range(len(methods)), times, color=colors[:len(methods)], alpha=0.7)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45)
    ax2.set_ylabel('计算时间 (秒)')
    ax2.set_title('计算效率对比')
    ax2.grid(True, alpha=0.3)
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom')
    
    # 3. 相对误差对比
    bars3 = ax3.bar(range(len(methods)), errors, color=colors[:len(methods)], alpha=0.7)
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(methods, rotation=45)
    ax3.set_ylabel('相对误差')
    ax3.set_title('准确性对比（相对误差）')
    ax3.grid(True, alpha=0.3)
    
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    # 4. 标准误差对比
    bars4 = ax4.bar(range(len(methods)), stds, color=colors[:len(methods)], alpha=0.7)
    ax4.set_xticks(range(len(methods)))
    ax4.set_xticklabels(methods, rotation=45)
    ax4.set_ylabel('标准误差')
    ax4.set_title('稳定性对比（标准误差）')
    ax4.grid(True, alpha=0.3)
    
    for i, bar in enumerate(bars4):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.suptitle('蒙特卡洛方法性能综合比较（100维问题）', fontsize=16, y=1.02)
    plt.savefig('mc_performance_comparison.png', dpi=300, bbox_inches='tight')
    # 移除plt.show()以避免在非交互式环境中挂起
    # plt.show()
    
    print("\n✓ 可视化结果已保存到 mc_performance_comparison.png")

def main():
    """主函数"""
    print("开始蒙特卡洛方法性能比较...")
    print("此程序将比较多种蒙特卡洛优化技术在100维问题上的表现")
    print()
    
    # 运行比较
    results = compare_methods()
    
    print("\n" + "=" * 80)
    print("                比较分析完成")
    print("=" * 80)
    
    # 找出最佳方法
    best_idx = np.argmin(results['errors'])
    best_method = results['methods'][best_idx]
    best_error = results['errors'][best_idx]
    
    print(f"推荐使用: {best_method} 方法")
    print(f"相对误差: {best_error:.4f}")
    print(f"估计价格: {results['prices'][best_idx]:.6f}")
    print(f"解析解: {results['analytical_solution']:.6f}")
    print()
    
    print("各方法特点:")
    print("• 标准蒙特卡洛: 基准方法，实现简单")
    print("• 对偶变量法: 方差缩减约30-50%，计算成本基本不变")
    print("• 准蒙特卡洛: 收敛速度更快，需要sobol_seq库")
    print("• 并行蒙特卡洛: 多核CPU加速，适合大规模计算")
    if CUPY_AVAILABLE:
        print("• GPU加速: 10-100倍速度提升，需要GPU硬件")
    print("• 综合优化: 自动选择最佳可用方法")
    
    return results

if __name__ == "__main__":
    # 运行主程序
    results = main()