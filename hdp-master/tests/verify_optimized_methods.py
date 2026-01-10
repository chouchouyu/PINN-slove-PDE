import sys, os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.mc.American import American

# 模拟一个简单的随机游走类用于测试
class MockRandomWalk:
    def __init__(self, N=5, asset_num=1):
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

# 测试函数
def test_optimized_methods():
    # 创建测试对象
    rw = MockRandomWalk(N=7, asset_num=1)
    american = American(simple_payoff, rw)
    
    # 生成测试用的现金流矩阵
    path_num = 100
    cashflow_matrix = np.zeros([path_num, rw.N+1])
    
    # 随机设置一些现金流
    np.random.seed(42)  # 设置随机种子以确保可重复性
    for i in range(path_num):
        # 随机选择一个时间点（1到N）设置现金流
        t = np.random.randint(1, rw.N+1)
        cashflow_matrix[i, t] = np.random.uniform(1, 100)
    
    print("测试_get_discounted_cashflow方法...")
    # 测试_get_discounted_cashflow方法
    for t in range(rw.N):
        # 使用原始方法计算
        original_result = original_get_discounted_cashflow(american, t, cashflow_matrix, path_num)
        # 使用优化后的方法计算
        optimized_result = american._get_discounted_cashflow(t, cashflow_matrix, path_num)
        
        # 计算差异
        diff = np.abs(original_result - optimized_result)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"  t={t}: 最大差异 = {max_diff:.10f}, 平均差异 = {mean_diff:.10f}")
        
        # 验证结果是否一致
        assert np.allclose(original_result, optimized_result), f"_get_discounted_cashflow在t={t}时结果不一致！"
    
    print("\n测试_get_discounted_cashflow_at_t0方法...")
    # 测试_get_discounted_cashflow_at_t0方法
    original_result_t0 = original_get_discounted_cashflow_at_t0(american, cashflow_matrix)
    optimized_result_t0 = american._get_discounted_cashflow_at_t0(cashflow_matrix)
    
    print(f"  原始方法结果: {original_result_t0:.10f}")
    print(f"  优化方法结果: {optimized_result_t0:.10f}")
    print(f"  差异: {np.abs(original_result_t0 - optimized_result_t0):.10f}")
    
    # 验证结果是否一致
    assert np.isclose(original_result_t0, optimized_result_t0), "_get_discounted_cashflow_at_t0结果不一致！"
    
    print("\n所有测试通过！优化后的方法与原始方法结果一致。")

if __name__ == "__main__":
    test_optimized_methods()
