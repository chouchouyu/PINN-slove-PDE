import sys, os
import numpy as np
import timeit

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

# 导入必要的类
from src.blackscholes.mc.American import American
from src.blackscholes.utils.GBM import GBM

# 定义测试用的支付函数
def put_payoff(X):
    """看跌期权支付函数"""
    return max(100 - X[0], 0)

# 创建GBM和American实例
gbm = GBM(T=1.0, N=10, init_price_vec=[100], ir=0.05, vol_vec=[0.2], dividend_vec=[0], corr_mat=np.eye(1))
american = American(payoff_func=put_payoff, random_walk=gbm)

# 生成模拟数据和现金流矩阵
def generate_test_data(american_instance, path_num=1000):
    american_instance.simulation_result = american_instance.random_walk.simulateV2(path_num)
    cashflow_matrix = np.zeros([path_num, american_instance.random_walk.N+1])
    
    # 生成一些测试用的现金流数据
    for i in range(path_num):
        # 随机选择一个时间点作为现金流产生点
        t = np.random.randint(1, american_instance.random_walk.N+1)
        # 随机生成现金流值
        cashflow_matrix[i, t] = np.random.uniform(0, 50)
    
    return cashflow_matrix

# 原始方法
def original_discounted_cashflow(american_instance, cashflow_matrix):
    return american_instance._get_discounted_cashflow_at_t0(cashflow_matrix)

# 向量化方法
def vectorized_discounted_cashflow(american_instance, cashflow_matrix):
    ir = american_instance.random_walk.ir
    dt = american_instance.random_walk.dt
    
    # 从t=1开始查找第一个非零现金流
    cashflow_matrix_from_1 = cashflow_matrix[:, 1:]
    mask = cashflow_matrix_from_1 != 0
    
    # 找到每行第一个非零元素的索引
    first_nonzero_indices_in_subset = np.argmax(mask, axis=1)
    
    # 检查哪些行有非零元素
    has_nonzero = np.any(mask, axis=1)
    
    # 调整索引到原始矩阵的时间点（+1是因为我们从t=1开始）
    first_cashflow_times = first_nonzero_indices_in_subset[has_nonzero] + 1
    
    # 计算折扣因子
    time_vector = np.arange(cashflow_matrix.shape[1])
    discount_factors = np.exp(-ir * time_vector * dt)
    
    # 获取第一个非零现金流值
    first_cashflow_values = cashflow_matrix[has_nonzero, first_cashflow_times]
    
    # 计算折扣后的值并求平均
    discounted_values = first_cashflow_values * discount_factors[first_cashflow_times]
    
    return discounted_values.mean()

# 运行测试
if __name__ == "__main__":
    # 生成测试数据
    cashflow_matrix = generate_test_data(american)
    
    # 计算原始方法结果
    original_result = original_discounted_cashflow(american, cashflow_matrix)
    
    # 计算向量化方法结果
    vectorized_result = vectorized_discounted_cashflow(american, cashflow_matrix)
    
    # 比较结果
    print(f"原始方法结果: {original_result}")
    print(f"向量化方法结果: {vectorized_result}")
    print(f"结果差异: {abs(original_result - vectorized_result)}")
    print(f"结果是否一致: {np.isclose(original_result, vectorized_result)}")
    
    # 性能比较
    print("\n性能比较:")
    original_time = timeit.timeit(lambda: original_discounted_cashflow(american, cashflow_matrix), number=1000)
    vectorized_time = timeit.timeit(lambda: vectorized_discounted_cashflow(american, cashflow_matrix), number=1000)
    
    print(f"原始方法平均时间: {original_time/1000:.6f} 秒")
    print(f"向量化方法平均时间: {vectorized_time/1000:.6f} 秒")
    print(f"性能提升: {((original_time - vectorized_time)/original_time * 100):.2f}%")
