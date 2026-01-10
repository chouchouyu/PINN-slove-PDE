import numpy as np
import time

from cqf_1_Regression_optimized import Regression


class Paths_generater:
    def __init__(self, T, days, S0_vec, r: float, vol_vec, dividend_vec, corr_mat):
        assert len(S0_vec) == len(vol_vec) == len(dividend_vec) == corr_mat.shape[0] == corr_mat.shape[1], "Vectors' lengths are different"
        self.dt = T / days
        self.T = T
        self.days = days
        self.S0_vec = S0_vec
        self.r = r
        self.vol_vec = vol_vec
        self.dividend_vec = dividend_vec
        self.corr_mat = corr_mat
        self.asset_num = len(S0_vec)
        
        self.n = len(S0_vec)
        self.sqrt_dt = np.sqrt(self.dt)
        
        if np.allclose(corr_mat, np.eye(self.n)):
            self.L = np.eye(self.n)
        else:
            try:
                self.L = np.linalg.cholesky(corr_mat)
            except np.linalg.LinAlgError:
                print("警告: 相关矩阵不是正定的，使用单位矩阵")
                self.L = np.eye(self.n)
    
    def gbm(self, n_simulations: int)-> np.ndarray:
        price_paths =[]    
        L = np.linalg.cholesky(self.corr_mat)
        drift_vec = self.r - self.dividend_vec  
        for _ in range(n_simulations):
            price_path = np.zeros((self.asset_num, self.days + 1))
            price_path[:, 0] = self.S0_vec
            for t in range(1, self.days + 1):
                dW = L.dot(np.random.normal(size=self.asset_num))*np.sqrt(self.dt)
                rand_term = np.multiply(self.vol_vec, dW)
                price_path[:, t] = np.multiply(price_path[:, t-1], np.exp((drift_vec-self.vol_vec**2/2)*self.dt + rand_term))
            price_paths.append(price_path)

        return np.array(price_paths)


class MC_American_Option:
    def __init__(self, paths_generater: Paths_generater, payoff_func, regression_method='fast'):
        self.paths_generater = paths_generater
        self.payoff_func = payoff_func
        self.regression_method = regression_method
        
        self.discount_factors = self._precompute_discount_factors()
    
    def _precompute_discount_factors(self):
        days = self.paths_generater.days
        r = self.paths_generater.r
        dt = self.paths_generater.dt
        
        discount_factors = np.zeros(days + 1)
        for t in range(days + 1):
            discount_factors[t] = np.exp(-r * dt * (days - t))
        
        return discount_factors
    
    def _get_discounted_cashflow(self, t: int, cashflow_matrix: np.ndarray, path_num) -> np.ndarray:
        """
        与原始版本保持一致的实现
        """
        # 计算折扣因子
        time_indices = np.arange(path_num+1)
        discount_factors = np.exp((t - time_indices) * self.paths_generater.dt * self.paths_generater.r)
        
        # 创建掩码，只考虑 t+1 到 N 的时间点
        mask = np.zeros_like(cashflow_matrix, dtype=bool)
        mask[:, t+1:path_num+1] = cashflow_matrix[:, t+1:path_num+1] != 0
        
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
    
    def _get_discounted_cashflow_optimized(self, t, cash_flow, n_simulations):
        # Match the original method's logic
        r = self.paths_generater.r
        dt = self.paths_generater.dt
        
        # Calculate discount factors
        time_indices = np.arange(n_simulations+1)
        discount_factors = np.exp((t - time_indices) * dt * r)
        
        # Create mask, only consider t+1 to n_simulations+1 time points
        mask = np.zeros_like(cash_flow, dtype=bool)
        mask[:, t+1:n_simulations+1] = cash_flow[:, t+1:n_simulations+1] != 0
        
        # Reverse search for first non-zero value in each row
        reversed_mask = np.fliplr(mask)
        reversed_indices = np.argmax(reversed_mask, axis=1)
        
        # Convert to original indices
        first_nonzero_indices = mask.shape[1] - reversed_indices - 1
        
        # Check if there are non-zero values
        has_nonzero = np.any(mask, axis=1)
        
        # Calculate result
        result = np.zeros(n_simulations)
        result[has_nonzero] = cash_flow[has_nonzero, first_nonzero_indices[has_nonzero]] * discount_factors[first_nonzero_indices[has_nonzero]]
        
        return result
    
    def price(self, n_simulations=1000):
        self.simulations_paths = self.paths_generater.gbm(n_simulations=n_simulations)
        
        cash_flow = np.zeros([n_simulations, self.paths_generater.days + 1])
        
        for t in range(self.paths_generater.days, 0, -1):
            prices_at_t = self.simulations_paths[:, :, t]
            
            if t == self.paths_generater.days:
                maturity_price = prices_at_t
                maturity_payoff = np.array(list(map(self.payoff_func, maturity_price)))
                cash_flow[:, -1] = maturity_payoff
            else:
                discounted_cashflow = self._get_discounted_cashflow_optimized(t, cash_flow, n_simulations)
                
                r = Regression(prices_at_t, discounted_cashflow, payoff_func=self.payoff_func, method=self.regression_method)
                
                if r.has_intrinsic_value:
                    all_cur_payoff = np.array(list(map(self.payoff_func, prices_at_t)))
                    continuation_value = np.array([r.evaluate(X) for X in prices_at_t[r.index]])
                    
                    exercise_index = r.index[all_cur_payoff[r.index] >= continuation_value]
        
                    if len(exercise_index) > 0:
                        cash_flow[exercise_index] = np.zeros(cash_flow[exercise_index].shape)
                        cash_flow[exercise_index, t] = all_cur_payoff[exercise_index]
                
                else:
                    continue
        
        return self._get_discounted_cashflow_at_t0(cash_flow)
    
    def _get_discounted_cashflow_at_t0(self, cashflow_matrix):
        """
        与原始版本保持一致的实现
        """
        # 提取第1列到最后一列的数据
        future_cashflows = cashflow_matrix[:, 1:]
        
        # 查找每行第一个非零值的位置
        first_nonzero_positions = np.argmax(future_cashflows != 0, axis=1)
        
        # 检查是否存在非零值
        has_cashflow = np.any(future_cashflows != 0, axis=1)
        
        # 计算对应的时间索引 (从1开始)
        time_indices = first_nonzero_positions + 1
        
        # 计算折扣因子
        discount_factors = np.exp(-self.paths_generater.r * time_indices * self.paths_generater.dt)
        
        # 获取对应的现金流值并折扣
        discounted_values = future_cashflows[np.arange(len(cashflow_matrix)), first_nonzero_positions] * discount_factors
        
        # 只考虑有现金流的路径
        return discounted_values[has_cashflow].mean()


def test_get_discounted_cashflow():
    r = 0.03
    dt = 3
    random_walk = Paths_generater(dt, 3, np.ones(1), r, np.ones(1), np.zeros(1), np.eye(1))
    def test_payoff(*l):
        return max(3 - np.sum(l), 0)
    opt = MC_American_Option(random_walk, test_payoff)
    
    print("\n测试1: cashflow_matrix = [[0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 2]], t=2")
    cashflow_matrix = np.array([[0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 2]])
    t = 2
    discounted = opt._get_discounted_cashflow_optimized(t, cashflow_matrix, 3)
    print(f"结果: {discounted}")
    
    discount_factor_t2 = np.exp(-r * dt)
    expected_t2_path0 = 3 * discount_factor_t2
    expected_t2_path2 = 2 * discount_factor_t2
    expected_t2 = np.array([expected_t2_path0, 0, expected_t2_path2])
    print(f"期望: {expected_t2}")
    assert np.allclose(discounted, expected_t2, atol=1e-8), f"折现现金流计算错误: {discounted} != {expected_t2}"
    print("✓ 折现现金流计算测试通过")
    
    print("\n测试2: cashflow_matrix = [[0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 2]], t=0")
    cashflow_matrix2 = np.array([[0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 2]])
    t = 0
    discounted2 = opt._get_discounted_cashflow_optimized(t, cashflow_matrix2, 3)
    print(f"结果: {discounted2}")
    
    discount_factor_t0_path0 = np.exp(-r * dt * 2)
    discount_factor_t0_path2 = np.exp(-r * dt * 3)
    expected_t0_path0 = 3 * discount_factor_t0_path0
    expected_t0_path2 = 2 * discount_factor_t0_path2
    expected_t0 = np.array([expected_t0_path0, 0, expected_t0_path2])
    print(f"期望: {expected_t0}")
    assert np.allclose(discounted2, expected_t0, atol=1e-8), f"折现现金流计算错误: {discounted2} != {expected_t0}"
    print("✓ 折现现金流计算测试通过")


def test_regression():
    np.random.seed(456)
    X = np.random.rand(100, 5) * 100
    Y = np.random.rand(100) * 50
    
    def payoff(*l):
        return max(np.max(l) - 100, 0)
    
    r_fast = Regression(X.reshape(1, -1), Y, payoff_func=payoff, method='fast')
    r_ref = Regression(X.reshape(1, -1), Y, payoff_func=payoff, method='reference')
    
    print(f"Fast 回归 - has_intrinsic_value: {r_fast.has_intrinsic_value}, index长度: {len(r_fast.index) if r_fast.index is not None else 0}")
    print(f"Reference 回归 - has_intrinsic_value: {r_ref.has_intrinsic_value}, index长度: {len(r_ref.index) if r_ref.index is not None else 0}")
    
    if r_fast.has_intrinsic_value and r_ref.has_intrinsic_value:
        test_point = X[0].reshape(1, -1)
        val_fast = r_fast.evaluate(test_point)
        val_ref = r_ref.evaluate(test_point)
        print(f"评估点: {test_point[0]}")
        print(f"Fast 评估结果: {val_fast:.6f}")
        print(f"Reference 评估结果: {val_ref:.6f}")
    
    print("✓ 回归测试完成")


def test_100d_pricing():
    np.random.seed(123)
    strike = 100
    asset_num = 100
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
    
    random_walk = Paths_generater(1, 50, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
    
    def test_payoff(*l):
        return max(np.max(l) - strike, 0)
    
    print("\n" + "=" * 60)
    print("测试不同回归方法的性能")
    print("=" * 60)
    
    print("\n使用 'fast' 方法:")
    np.random.seed(123)
    opt_fast = MC_American_Option(random_walk, test_payoff, regression_method='fast')
    start_time = time.time()
    price_fast = opt_fast.price(100)
    time_fast = time.time() - start_time
    print(f"价格: {price_fast:.8f}")
    print(f"运行时间: {time_fast:.4f} 秒")
    
    print("\n使用 'reference' 方法:")
    np.random.seed(123)
    opt_ref = MC_American_Option(random_walk, test_payoff, regression_method='reference')
    start_time = time.time()
    price_ref = opt_ref.price(100)
    time_ref = time.time() - start_time
    print(f"价格: {price_ref:.8f}")
    print(f"运行时间: {time_ref:.4f} 秒")
    
    print("\n" + "=" * 60)
    print("性能对比")
    print("=" * 60)
    print(f"Fast 方法价格: {price_fast:.8f}")
    print(f"Reference 方法价格: {price_ref:.8f}")
    print(f"价格差异: {abs(price_fast - price_ref):.8f}")
    print(f"Fast 方法速度提升: {(time_ref - time_fast) / time_ref * 100:.2f}%")


def benchmark_performance():
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
    
    random_walk = Paths_generater(1, 50, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
    
    def test_payoff(*l):
        return max(np.max(l) - strike, 0)
    
    print("\n" + "=" * 60)
    print("预计算折现因子性能测试")
    print("=" * 60)
    
    np.random.seed(123)
    opt = MC_American_Option(random_walk, test_payoff, regression_method='fast')
    
    start_time = time.time()
    price = opt.price(50)
    elapsed = time.time() - start_time
    
    print(f"期权价格: {price:.8f}")
    print(f"运行时间: {elapsed:.4f} 秒")
    print(f"折现因子数量: {len(opt.discount_factors)}")


if __name__ == "__main__":
    print("=" * 60)
    print("测试优化后的美式期权定价代码")
    print("=" * 60)
    
    print("\n1. 测试折现现金流计算:")
    test_get_discounted_cashflow()
    
    print("\n2. 测试回归算法:")
    test_regression()
    
    print("\n3. 性能基准测试:")
    benchmark_performance()
    
    print("\n4. 100维美式期权定价测试:")
    test_100d_pricing()
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
