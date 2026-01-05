import numpy as np
import time

class DiscountedCashflowCalculator:
    """现金流折现计算器，提供循环和向量化两种实现"""
    
    def __init__(self, ir=0.05, dt=0.1):
        """
        初始化计算器
        
        参数:
        - ir: 无风险利率
        - dt: 时间步长
        """
        self.ir = ir
        self.dt = dt
        
    def _get_discounted_cashflow_at_t0_original(self, cashflow_matrix):
        """
        原始循环实现
        计算每条路径的现金流折现到t0时刻的平均值
        
        参数:
        - cashflow_matrix: 现金流矩阵，形状 (n_paths, n_time_steps+1)
        
        返回: 平均折现价值
        """
        summation = 0
        for cashflow in cashflow_matrix:
            for i in range(1, len(cashflow)):
                if cashflow[i] != 0:
                    summation += cashflow[i] * np.exp(-self.ir * i * self.dt)
                    break
        return summation / len(cashflow_matrix)
    
    def _get_discounted_cashflow_at_t0_vectorized(self, cashflow_matrix):
        """
        向量化实现
        使用NumPy向量化操作提高计算效率
        
        参数:
        - cashflow_matrix: 现金流矩阵，形状 (n_paths, n_time_steps+1)
        
        返回: 平均折现价值
        """
        # 1. 创建时间索引数组
        n_paths, n_times = cashflow_matrix.shape
        time_indices = np.arange(n_times)
        
        # 2. 创建折现因子矩阵
        # 注意：我们需要排除时间0，因为原代码从i=1开始
        discount_factors = np.exp(-self.ir * time_indices * self.dt)
        
        # 3. 找到每条路径第一个非零现金流的位置（从索引1开始）
        # 方法：将零设为np.inf，然后取最小值的位置
        masked_matrix = cashflow_matrix.copy()
        
        # 将0替换为inf，这样最小非零值就是第一个非零位置
        masked_matrix[masked_matrix == 0] = np.inf
        
        # 找到第一个非零位置（如果全为inf则返回0）
        first_nonzero_idx = np.argmin(masked_matrix, axis=1)
        
        # 标记哪些路径有非零现金流
        # 修正：将first_non_zero_idx改为first_nonzero_idx
        has_nonzero_cashflow = (first_nonzero_idx < n_times - 1) & (first_nonzero_idx > 0)
        
        # 4. 提取第一个非零现金流的值
        # 使用高级索引获取每条路径的第一个非零现金流
        row_indices = np.arange(n_paths)
        first_nonzero_values = cashflow_matrix[row_indices, first_nonzero_idx]
        
        # 5. 对于没有非零现金流的路径，将值设为0
        first_nonzero_values[~has_nonzero_cashflow] = 0
        
        # 6. 计算折现值
        discounted_values = first_nonzero_values * discount_factors[first_nonzero_idx]
        
        # 7. 计算平均值
        return np.mean(discounted_values)
    
    def _get_discounted_cashflow_at_t0_optimized(self, cashflow_matrix):
        """
        优化后的向量化实现（推荐）
        更简洁高效的实现方式
        
        参数:
        - cashflow_matrix: 现金流矩阵
        
        返回: 平均折现价值
        """
        n_paths, n_times = cashflow_matrix.shape
        
        # 创建时间索引和折现因子
        time_indices = np.arange(n_times)
        discount_factors = np.exp(-self.ir * time_indices * self.dt)
        
        # 更高效的实现：使用argmax找到第一个非零位置
        # 注意：这个方法假设现金流非负
        # 对于每一行，找到第一个非零元素的位置
        # 使用argmax在布尔数组上，True > False
        non_zero_mask = cashflow_matrix[:, 1:] != 0  # 从第1列开始
        
        # 找到第一个非零的位置
        first_nonzero_relative = non_zero_mask.argmax(axis=1)
        
        # 创建行索引
        row_indices = np.arange(n_paths)
        
        # 将相对位置转换为绝对位置（加1是因为我们从第1列开始）
        first_nonzero_absolute = first_nonzero_relative + 1
        
        # 检查哪些行有非零现金流（如果全为0，argmax返回0）
        has_nonzero = non_zero_mask.any(axis=1)
        
        # 初始化结果数组
        discounted_values = np.zeros(n_paths)
        
        # 只处理有非零现金流的路径
        if has_nonzero.any():
            # 获取有非零现金流的行索引
            nonzero_rows = row_indices[has_nonzero]
            
            # 获取对应的第一个非零位置
            nonzero_cols = first_nonzero_absolute[has_nonzero]
            
            # 获取现金流值
            cashflow_values = cashflow_matrix[nonzero_rows, nonzero_cols]
            
            # 计算折现值
            discount_factors_for_nonzero = discount_factors[nonzero_cols]
            discounted_values[nonzero_rows] = cashflow_values * discount_factors_for_nonzero
        
        # 返回平均值
        return np.mean(discounted_values)
    
    def _get_discounted_cashflow_at_t0_elegant(self, cashflow_matrix):
        """
        优雅的向量化实现
        使用np.where和高级索引
        
        参数:
        - cashflow_matrix: 现金流矩阵
        
        返回: 平均折现价值
        """
        n_paths, n_times = cashflow_matrix.shape
        
        # 创建时间索引和折现因子
        time_indices = np.arange(n_times)
        discount_factors = np.exp(-self.ir * time_indices * self.dt)
        
        # 找到每条路径的第一个非零现金流位置
        # 创建一个布尔矩阵，标识非零元素
        non_zero_mask = cashflow_matrix != 0
        
        # 对于每条路径，找到第一个非零的列索引
        # 如果路径全为0，则返回n_times
        first_nonzero_idx = np.where(non_zero_mask.any(axis=1),
                                    non_zero_mask.argmax(axis=1),
                                    n_times)
        
        # 创建掩码，排除全0路径和索引为0的情况（原代码从i=1开始）
        valid_mask = (first_nonzero_idx > 0) & (first_nonzero_idx < n_times)
        
        # 提取有效路径的现金流值
        valid_rows = np.arange(n_paths)[valid_mask]
        valid_cols = first_nonzero_idx[valid_mask]
        
        # 计算折现值
        discounted_values = np.zeros(n_paths)
        if len(valid_rows) > 0:
            cashflow_values = cashflow_matrix[valid_rows, valid_cols]
            discount_vals = discount_factors[valid_cols]
            discounted_values[valid_rows] = cashflow_values * discount_vals
        
        return np.mean(discounted_values)

def generate_test_data(n_paths=1000, n_time_steps=10):
    """生成测试用的现金流矩阵"""
    np.random.seed(42)
    
    # 创建基础现金流矩阵
    cashflow_matrix = np.zeros((n_paths, n_time_steps + 1))
    
    # 为每条路径随机生成现金流
    for i in range(n_paths):
        # 随机决定是否产生现金流
        if np.random.random() > 0.3:  # 70%的路径有现金流
            # 随机选择现金流发生的时间（从1开始）
            cashflow_time = np.random.randint(1, n_time_steps + 1)
            # 随机生成现金流值（1-10之间）
            cashflow_value = np.random.uniform(1, 10)
            cashflow_matrix[i, cashflow_time] = cashflow_value
    
    return cashflow_matrix

def test_performance():
    """测试不同实现的性能"""
    print("现金流折现计算性能测试")
    print("=" * 60)
    
    # 生成测试数据
    n_paths = 5000
    n_time_steps = 20
    cashflow_matrix = generate_test_data(n_paths, n_time_steps)
    
    print(f"测试数据:")
    print(f"  路径数量: {n_paths}")
    print(f"  时间步数: {n_time_steps}")
    print(f"  矩阵形状: {cashflow_matrix.shape}")
    print(f"  非零元素比例: {np.sum(cashflow_matrix != 0) / cashflow_matrix.size:.2%}")
    
    # 创建计算器
    calculator = DiscountedCashflowCalculator(ir=0.05, dt=0.1)
    
    # 测试原始实现
    print("\n1. 测试原始循环实现:")
    start_time = time.perf_counter()
    result_original = calculator._get_discounted_cashflow_at_t0_original(cashflow_matrix)
    end_time = time.perf_counter()
    time_original = end_time - start_time
    print(f"   结果: {result_original:.6f}")
    print(f"   耗时: {time_original:.4f} 秒")
    
    # 测试向量化实现
    print("\n2. 测试向量化实现（修正后）:")
    start_time = time.perf_counter()
    result_vectorized = calculator._get_discounted_cashflow_at_t0_vectorized(cashflow_matrix)
    end_time = time.perf_counter()
    time_vectorized = end_time - start_time
    print(f"   结果: {result_vectorized:.6f}")
    print(f"   耗时: {time_vectorized:.4f} 秒")
    
    # 测试优化实现
    print("\n3. 测试优化实现:")
    start_time = time.perf_counter()
    result_optimized = calculator._get_discounted_cashflow_at_t0_optimized(cashflow_matrix)
    end_time = time.perf_counter()
    time_optimized = end_time - start_time
    print(f"   结果: {result_optimized:.6f}")
    print(f"   耗时: {time_optimized:.4f} 秒")
    
    # 测试优雅实现
    print("\n4. 测试优雅实现:")
    start_time = time.perf_counter()
    result_elegant = calculator._get_discounted_cashflow_at_t0_elegant(cashflow_matrix)
    end_time = time.perf_counter()
    time_elegant = end_time - start_time
    print(f"   结果: {result_elegant:.6f}")
    print(f"   耗时: {time_elegant:.4f} 秒")
    
    # 验证结果一致性
    print("\n" + "=" * 60)
    print("结果验证:")
    
    # 计算相对误差
    def relative_error(a, b):
        if a == 0 and b == 0:
            return 0
        return abs(a - b) / max(abs(a), abs(b))
    
    error_1 = relative_error(result_original, result_vectorized)
    error_2 = relative_error(result_original, result_optimized)
    error_3 = relative_error(result_original, result_elegant)
    
    print(f"  原始 vs 向量化: {error_1:.10f} {'✓' if error_1 < 1e-10 else '✗'}")
    print(f"  原始 vs 优化: {error_2:.10f} {'✓' if error_2 < 1e-10 else '✗'}")
    print(f"  原始 vs 优雅: {error_3:.10f} {'✓' if error_3 < 1e-10 else '✗'}")
    
    # 性能对比
    print("\n" + "=" * 60)
    print("性能对比:")
    speedup_vectorized = time_original / time_vectorized
    speedup_optimized = time_original / time_optimized
    speedup_elegant = time_original / time_elegant
    
    print(f"  向量化加速比: {speedup_vectorized:.2f}x")
    print(f"  优化加速比: {speedup_optimized:.2f}x")
    print(f"  优雅加速比: {speedup_elegant:.2f}x")
    
    # 推荐使用的方法
    print("\n" + "=" * 60)
    print("使用建议:")
    print("""
    1. 对于大型数据集（n_paths > 1000），推荐使用 _get_discounted_cashflow_at_t0_optimized
        - 性能最好
        - 代码清晰
        - 只处理有效路径
    
    2. 对于小型数据集或调试，可以使用 _get_discounted_cashflow_at_t0_original
        - 逻辑直观
        - 易于理解
    
    3. 关键优化点:
        - 使用向量化操作替代Python循环
        - 使用高级索引避免循环
        - 使用布尔掩码只处理有效数据
        - 一次性计算所有折现因子
    """)

def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 60)
    print("边界情况测试")
    print("=" * 60)
    
    calculator = DiscountedCashflowCalculator(ir=0.05, dt=0.1)
    
    # 测试1: 全零矩阵
    print("\n1. 全零矩阵:")
    zero_matrix = np.zeros((100, 11))
    result = calculator._get_discounted_cashflow_at_t0_optimized(zero_matrix)
    print(f"   结果: {result:.6f} (应为0)")
    
    # 测试2: 所有现金流都在时间1
    print("\n2. 所有现金流都在时间1:")
    early_matrix = np.zeros((100, 11))
    early_matrix[:, 1] = 5.0
    result = calculator._get_discounted_cashflow_at_t0_optimized(early_matrix)
    expected = 5.0 * np.exp(-0.05 * 1 * 0.1)
    print(f"   结果: {result:.6f}, 期望: {expected:.6f}")
    
    # 测试3: 所有现金流都在最后时间
    print("\n3. 所有现金流都在最后时间:")
    late_matrix = np.zeros((100, 11))
    late_matrix[:, 10] = 8.0
    result = calculator._get_discounted_cashflow_at_t0_optimized(late_matrix)
    expected = 8.0 * np.exp(-0.05 * 10 * 0.1)
    print(f"   结果: {result:.6f}, 期望: {expected:.6f}")
    
    return True

def main():
    """主函数"""
    print("现金流折现向量化计算（修正版）")
    print("=" * 60)
    
    # 测试性能
    test_performance()
    
    # 测试边界情况
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("代码示例使用:")
    print("=" * 60)
    
    # 示例使用
    calculator = DiscountedCashflowCalculator(ir=0.05, dt=0.1)
    
    # 创建一个小型测试矩阵
    test_matrix = np.array([
        [0, 0, 5, 0, 0],  # 路径1: 在时间2有现金流5
        [0, 3, 0, 0, 0],  # 路径2: 在时间1有现金流3
        [0, 0, 0, 0, 0],  # 路径3: 没有现金流
        [0, 0, 0, 4, 0]   # 路径4: 在时间3有现金流4
    ])
    
    print("\n测试矩阵:")
    print(test_matrix)
    
    # 计算折现值
    result = calculator._get_discounted_cashflow_at_t0_optimized(test_matrix)
    
    print(f"\n平均折现值: {result:.6f}")
    
    # 手动验证
    print("\n手动验证:")
    cashflows = [5, 3, 0, 4]
    times = [2, 1, 0, 3]  # 对应的时间点
    discounted_sum = 0
    for i, (cf, t) in enumerate(zip(cashflows, times)):
        if cf > 0 and t > 0:
            discounted = cf * np.exp(-0.05 * t * 0.1)
            print(f"  路径{i+1}: {cf} * exp(-0.05*{t}*0.1) = {discounted:.6f}")
            discounted_sum += discounted
    avg = discounted_sum / 4
    print(f"  平均值: {avg:.6f}")

if __name__ == "__main__":
    main()
