import sys, os
import numpy as np
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.mc.American import American

# 模拟一个简单的随机游走类用于测试
class MockRandomWalk:
    def __init__(self, N=50, asset_num=1):
        self.N = N  # 时间步
        self.asset_num = asset_num
        self.ir = 0.05  # 利率
        self.dt = 1.0  # 时间间隔
        self.simulation_result = None
    
    def simulateV2(self, path_num):
        # 生成简单的模拟结果，用于测试
        self.simulation_result = np.random.randn(path_num, self.asset_num, self.N+1)
        return self.simulation_result

# 简单的支付函数 (看涨期权)
def simple_payoff(x):
    return max(x[0] - 100, 0) if len(x) == 1 else max(np.mean(x) - 100, 0)

# 原始的_get_discounted_cashflow方法（用于比较）
def original_get_discounted_cashflow(self, t, cashflow_matrix, path_num):
    N = self.random_walk.N
    ir = self.random_walk.ir
    dt = self.random_walk.dt
    
    # 计算折扣因子
    time_indices = np.arange(N+1)
    discount_factors = np.exp((t - time_indices) * dt * ir)
    
    # 创建掩码，只考虑 t+1 到 N 的时间点
    mask = np.zeros_like(cashflow_matrix, dtype=bool)
    mask[:, t+1:N+1] = cashflow_matrix[:, t+1:N+1] != 0
    
    # 倒序查找每行第一个非零值的位置
    reversed_mask = np.fliplr(mask)
    reversed_indices = np.argmax(reversed_mask, axis=1)
    
    # 转换为原始索引
    first_nonzero_indices = mask.shape[1] - reversed_indices - 1
    
    # 检查是否有非零值
    has_nonzero = np.any(mask, axis=1)
    
    # 计算结果
    result = np.zeros(path_num)
    result[has_nonzero] = cashflow_matrix[has_nonzero, first_nonzero_indices[has_nonzero]] * discount_factors[first_nonzero_indices[has_nonzero]]
    
    return result

# 原始的_get_discounted_cashflow_at_t0方法（用于比较）
def original_get_discounted_cashflow_at_t0(self, cashflow_matrix):
    summation = 0
    for cashflow in cashflow_matrix:
        for i in range(1, len(cashflow)):
            if cashflow[i] != 0:
                summation += cashflow[i]*np.exp(-self.random_walk.ir*i*self.random_walk.dt)
                break
    return summation / len(cashflow_matrix)

# 性能测试函数
def benchmark_methods():
    # 设置测试参数
    path_num = 10000  # 路径数量
    N = 50  # 时间步
    
    # 创建测试对象
    rw = MockRandomWalk(N=N, asset_num=1)
    american = American(simple_payoff, rw)
    
    # 生成测试用的现金流矩阵
    cashflow_matrix = np.zeros([path_num, rw.N+1])
    
    # 随机设置一些现金流
    np.random.seed(42)  # 设置随机种子以确保可重复性
    for i in range(path_num):
        # 随机选择一个时间点（1到N）设置现金流
        t = np.random.randint(1, rw.N+1)
        cashflow_matrix[i, t] = np.random.uniform(1, 100)
    
    print("性能测试：_get_discounted_cashflow方法")
    print("="*50)
    
    # 测试原始方法
    original_times = []
    for t in range(rw.N):
        start_time = time.time()
        original_result = original_get_discounted_cashflow(american, t, cashflow_matrix, path_num)
        end_time = time.time()
        original_times.append(end_time - start_time)
    
    original_total_time = sum(original_times)
    original_avg_time = original_total_time / rw.N
    
    print(f"原始方法：")
    print(f"  总时间: {original_total_time:.6f} 秒")
    print(f"  平均时间 (每t): {original_avg_time:.6f} 秒")
    
    # 测试优化后的方法
    optimized_times = []
    for t in range(rw.N):
        start_time = time.time()
        optimized_result = american._get_discounted_cashflow(t, cashflow_matrix, path_num)
        end_time = time.time()
        optimized_times.append(end_time - start_time)
    
    optimized_total_time = sum(optimized_times)
    optimized_avg_time = optimized_total_time / rw.N
    
    print(f"优化后方法：")
    print(f"  总时间: {optimized_total_time:.6f} 秒")
    print(f"  平均时间 (每t): {optimized_avg_time:.6f} 秒")
    
    # 计算性能提升
    speedup = original_total_time / optimized_total_time
    print(f"性能提升: {speedup:.2f}x ({(speedup-1)*100:.2f}% faster)")
    print()
    
    print("性能测试：_get_discounted_cashflow_at_t0方法")
    print("="*50)
    
    # 测试原始方法
    start_time = time.time()
    original_result_t0 = original_get_discounted_cashflow_at_t0(american, cashflow_matrix)
    end_time = time.time()
    original_t0_time = end_time - start_time
    
    print(f"原始方法：")
    print(f"  执行时间: {original_t0_time:.6f} 秒")
    
    # 测试优化后的方法
    start_time = time.time()
    optimized_result_t0 = american._get_discounted_cashflow_at_t0(cashflow_matrix)
    end_time = time.time()
    optimized_t0_time = end_time - start_time
    
    print(f"优化后方法：")
    print(f"  执行时间: {optimized_t0_time:.6f} 秒")
    
    # 计算性能提升
    speedup_t0 = original_t0_time / optimized_t0_time
    print(f"性能提升: {speedup_t0:.2f}x ({(speedup_t0-1)*100:.2f}% faster)")

if __name__ == "__main__":
    benchmark_methods()
