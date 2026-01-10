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

# 生成测试用的现金流矩阵
def generate_test_cashflow_matrix(path_num=1000, N=10):
    cashflow_matrix = np.zeros([path_num, N+1])
    
    for i in range(path_num):
        # 随机选择一个时间点作为现金流产生点
        t = np.random.randint(0, N+1)
        # 随机生成现金流值
        cashflow_matrix[i, t] = np.random.uniform(0, 50)
    
    return cashflow_matrix

# 原始方法
def original_get_discounted_cashflow(american_instance, t, cashflow_matrix, path_num):
    discounted_cashflow = np.zeros(path_num)
    for i in range(len(cashflow_matrix)):
        cashflow = cashflow_matrix[i]
        for j in range(american_instance.random_walk.N, t, -1):
            if cashflow[j] != 0:
                discounted_cashflow[i] = cashflow[j] * np.exp((t-j)*american_instance.random_walk.dt*american_instance.random_walk.ir)
                break
    return discounted_cashflow

# 向量化方法
def vectorized_get_discounted_cashflow(american_instance, t, cashflow_matrix, path_num):
    N = american_instance.random_walk.N
    ir = american_instance.random_walk.ir
    dt = american_instance.random_walk.dt
    
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

# 运行测试
if __name__ == "__main__":
    # 生成测试数据
    path_num = 1000
    N = 10
    cashflow_matrix = generate_test_cashflow_matrix(path_num, N)
    t = 3
    
    # 计算原始方法结果
    original_result = original_get_discounted_cashflow(american, t, cashflow_matrix, path_num)
    
    # 计算向量化方法结果
    vectorized_result = vectorized_get_discounted_cashflow(american, t, cashflow_matrix, path_num)
    
    # 比较结果
    print(f"原始方法结果前5个值: {original_result[:5]}")
    print(f"向量化方法结果前5个值: {vectorized_result[:5]}")
    print(f"结果是否一致: {np.allclose(original_result, vectorized_result)}")
    print(f"平均绝对误差: {np.mean(np.abs(original_result - vectorized_result))}")
    
    # 性能比较
    print("\n性能比较:")
    original_time = timeit.timeit(lambda: original_get_discounted_cashflow(american, t, cashflow_matrix, path_num), number=1000)
    vectorized_time = timeit.timeit(lambda: vectorized_get_discounted_cashflow(american, t, cashflow_matrix, path_num), number=1000)
    
    print(f"原始方法平均时间: {original_time/1000:.6f} 秒")
    print(f"向量化方法平均时间: {vectorized_time/1000:.6f} 秒")
    print(f"性能提升: {((original_time - vectorized_time)/original_time * 100):.2f}%")
