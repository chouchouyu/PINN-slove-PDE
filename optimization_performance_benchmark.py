import sys
import time
import numpy as np

sys.path.append("/Users/susan/PINN-slove-PDE")

from cqf_1_mc_American_optimized import (
    MC_American_Option, 
    Paths_generater,
    NUMBA_AVAILABLE
)

def benchmark_individual_optimizations():
    """测试各项单独优化的效果"""
    
    print("=" * 70)
    print("各项优化效果详细测试")
    print("=" * 70)
    
    np.random.seed(123)
    strike = 100
    asset_num = 50
    init_price_vec = 100 + np.random.randn(asset_num) * 5
    vol_vec = 0.2 + np.random.rand(asset_num) * 0.1
    ir = 0.05
    dividend_vec = 0.02 + np.random.rand(asset_num) * 0.03
    
    corr_mat = np.eye(asset_num)
    for i in range(asset_num):
        for j in range(i + 1, asset_num):
            corr = np.random.rand() * 0.1
            corr_mat[i, j] = corr
            corr_mat[j, i] = corr
    
    def test_payoff(*l):
        return max(np.max(l) - strike, 0)
    
    n_simulations = 100
    n_runs = 5
    
    print("\n1. Numba JIT 编译测试")
    print("-" * 50)
    print(f"Numba 可用: {NUMBA_AVAILABLE}")
    
    if NUMBA_AVAILABLE:
        print("进行Numba预热...")
        np.random.seed(123)
        pg = Paths_generater(1, 50, init_price_vec, ir, vol_vec, dividend_vec, corr_mat, n_jobs=1)
        opt = MC_American_Option(pg, test_payoff, regression_method='fast')
        
        times_with_numba = []
        for i in range(n_runs):
            np.random.seed(123 + i)
            start = time.time()
            price = opt.price(n_simulations)
            elapsed = time.time() - start
            times_with_numba.append(elapsed)
        
        avg_time_numba = np.mean(times_with_numba)
        print(f"使用Numba JIT平均运行时间: {avg_time_numba:.4f}秒 (基于{n_runs}次测试)")
        print(f"价格: {price:.6f}")
    else:
        print("跳过Numba测试（未安装）")
    
    print("\n2. 预计算折现因子测试")
    print("-" * 50)
    
    pg = Paths_generater(1, 50, init_price_vec, ir, vol_vec, dividend_vec, corr_mat, n_jobs=1)
    opt = MC_American_Option(pg, test_payoff, regression_method='fast')
    
    print(f"预计算折现因子数量: {len(opt.discount_factors)}")
    print("每次迭代不再需要重新计算折现因子")
    
    print("\n3. 向量化payoff计算测试")
    print("-" * 50)
    
    def slow_payoff(*l):
        return max(np.max(l) - strike, 0)
    
    np.random.seed(123)
    test_paths = np.random.rand(n_simulations, asset_num, 51) * 100 + 50
    test_prices = test_paths[:, :, 25]
    
    start = time.time()
    slow_results = np.array([slow_payoff(*row) for row in test_prices])
    time_slow = time.time() - start
    print(f"使用map/list comprehension: {time_slow:.6f}秒")
    
    print("\n4. 并行路径生成测试")
    print("-" * 50)
    
    times_sequential = []
    times_parallel_2 = []
    times_parallel_4 = []
    
    for i in range(n_runs):
        np.random.seed(123 + i)
        pg_seq = Paths_generater(1, 50, init_price_vec, ir, vol_vec, dividend_vec, corr_mat, n_jobs=1)
        start = time.time()
        _ = pg_seq.gbm(n_simulations * 5)
        times_sequential.append(time.time() - start)
        
        np.random.seed(123 + i)
        pg_par2 = Paths_generater(1, 50, init_price_vec, ir, vol_vec, dividend_vec, corr_mat, n_jobs=2)
        start = time.time()
        _ = pg_par2.gbm(n_simulations * 5)
        times_parallel_2.append(time.time() - start)
        
        np.random.seed(123 + i)
        pg_par4 = Paths_generater(1, 50, init_price_vec, ir, vol_vec, dividend_vec, corr_mat, n_jobs=4)
        start = time.time()
        _ = pg_par4.gbm(n_simulations * 5)
        times_parallel_4.append(time.time() - start)
    
    print(f"顺序路径生成平均时间: {np.mean(times_sequential):.4f}秒")
    print(f"2核并行路径生成平均时间: {np.mean(times_parallel_2):.4f}秒")
    print(f"4核并行路径生成平均时间: {np.mean(times_parallel_4):.4f}秒")
    
    if np.mean(times_sequential) > 0:
        speedup_2 = np.mean(times_sequential) / np.mean(times_parallel_2)
        speedup_4 = np.mean(times_sequential) / np.mean(times_parallel_4)
        print(f"2核加速比: {speedup_2:.2f}x")
        print(f"4核加速比: {speedup_4:.2f}x")
    
    print("\n5. 回归方法对比测试")
    print("-" * 50)
    
    pg = Paths_generater(1, 50, init_price_vec, ir, vol_vec, dividend_vec, corr_mat, n_jobs=1)
    
    times_fast = []
    times_reference = []
    
    for i in range(n_runs):
        np.random.seed(123 + i)
        opt_fast = MC_American_Option(pg, test_payoff, regression_method='fast')
        start = time.time()
        price_fast = opt_fast.price(n_simulations)
        times_fast.append(time.time() - start)
        
        np.random.seed(123 + i)
        opt_ref = MC_American_Option(pg, test_payoff, regression_method='reference')
        start = time.time()
        price_ref = opt_ref.price(n_simulations)
        times_reference.append(time.time() - start)
    
    print(f"Fast方法平均时间: {np.mean(times_fast):.4f}秒, 价格: {price_fast:.6f}")
    print(f"Reference方法平均时间: {np.mean(times_reference):.4f}秒, 价格: {price_ref:.6f}")
    print(f"价格差异: {abs(price_fast - price_ref):.10f}")


def benchmark_scalability():
    """可扩展性测试"""
    
    print("\n" + "=" * 70)
    print("可扩展性测试 - 不同维度下的性能")
    print("=" * 70)
    
    asset_configs = [
        (10, "10维"),
        (30, "30维"),
        (50, "50维"),
        (100, "100维")
    ]
    
    n_simulations = 50
    n_runs = 3
    
    for asset_num, desc in asset_configs:
        print(f"\n{desc}配置测试:")
        print("-" * 50)
        
        np.random.seed(123)
        init_price_vec = 100 + np.random.randn(asset_num) * 5
        vol_vec = 0.2 + np.random.rand(asset_num) * 0.1
        dividend_vec = 0.02 + np.random.rand(asset_num) * 0.03
        
        corr_mat = np.eye(asset_num)
        for i in range(asset_num):
            for j in range(i + 1, asset_num):
                corr = np.random.rand() * 0.1
                corr_mat[i, j] = corr
                corr_mat[j, i] = corr
        
        def test_payoff(*l):
            return max(np.max(l) - 100, 0)
        
        times = []
        for i in range(n_runs):
            np.random.seed(123 + i)
            pg = Paths_generater(1, 50, init_price_vec, 0.05, vol_vec, dividend_vec, corr_mat, n_jobs=1)
            opt = MC_American_Option(pg, test_payoff, regression_method='fast')
            
            start = time.time()
            price = opt.price(n_simulations)
            elapsed = time.time() - start
            times.append(elapsed)
        
        print(f"  平均运行时间: {np.mean(times):.4f}秒 (基于{n_runs}次测试)")
        print(f"  价格结果: {price:.6f}")


def benchmark_full_workflow():
    """完整工作流程性能测试"""
    
    print("\n" + "=" * 70)
    print("完整工作流程性能测试")
    print("=" * 70)
    
    np.random.seed(123)
    
    test_configs = [
        (50, 100, "50维 100次模拟"),
        (50, 200, "50维 200次模拟"),
        (100, 100, "100维 100次模拟"),
    ]
    
    n_runs = 3
    
    for asset_num, n_simulations, desc in test_configs:
        print(f"\n{desc}:")
        print("-" * 50)
        
        init_price_vec = 100 + np.random.randn(asset_num) * 5
        vol_vec = 0.2 + np.random.rand(asset_num) * 0.1
        dividend_vec = 0.02 + np.random.rand(asset_num) * 0.03
        
        corr_mat = np.eye(asset_num)
        for i in range(asset_num):
            for j in range(i + 1, asset_num):
                corr = np.random.rand() * 0.1
                corr_mat[i, j] = corr
                corr_mat[j, i] = corr
        
        def test_payoff(*l):
            return max(np.max(l) - 100, 0)
        
        times = []
        prices = []
        for i in range(n_runs):
            np.random.seed(123 + i)
            pg = Paths_generater(1, 50, init_price_vec, 0.05, vol_vec, dividend_vec, corr_mat, n_jobs=1)
            opt = MC_American_Option(pg, test_payoff, regression_method='fast')
            
            start = time.time()
            price = opt.price(n_simulations)
            elapsed = time.time() - start
            
            times.append(elapsed)
            prices.append(price)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_price = np.mean(prices)
        std_price = np.std(prices)
        
        print(f"  平均运行时间: {avg_time:.4f}±{std_time:.4f}秒")
        print(f"  平均价格: {avg_price:.6f}±{std_price:.6f}")
        print(f"  模拟/秒: {n_simulations / avg_time:.1f}")


def main():
    print("=" * 70)
    print("美式期权蒙特卡洛定价 - 优化性能基准测试")
    print("=" * 70)
    print(f"\nPython环境信息:")
    print(f"  NumPy版本: {np.__version__}")
    print(f"  Numba可用: {NUMBA_AVAILABLE}")
    
    benchmark_individual_optimizations()
    benchmark_scalability()
    benchmark_full_workflow()
    
    print("\n" + "=" * 70)
    print("性能测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
