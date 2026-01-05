
# "这三张图片分布实现随机过程的 逻辑有什么..."点击查看元宝的回答
# https://yb.tencent.com/s/vQ6dF6xx0qE0
# https://github.com/ZewenShen/hdp
import numpy as np
import pandas

import cqf_Regression
from cqf_utils import set_seed
 
def test_price_american_option():
    # Test parameters
    asset_num = 1
    S0_vec = 0.99*np.ones(asset_num)  # Initial stock price
    K = 1   # Strike price
    T = 1     # Time to maturity in years
    r = 0.05  # Risk-free interest rate
    dividend_vec = np.zeros(asset_num)
    sigma_vec = 0.2*np.ones(asset_num)  # Volatility
    corr_mat = np.eye(asset_num)
    # n_simulations = 300  # Number of Monte Carlo simulations

    # Expected price (this value should be pre-calculated or known from a reliable source)
    # expected_price = 10.45  # Example expected price for the given parameters
    # 1D 看跌期权（单一资产）
    def payoff(*l):
        return max(np.sum(l) - K, 0)  # strike - 资产价格，最大值为0
    # def payoff(prices):
    #     return np.maximum(prices - 100, 0)
    # 2D 看涨期权（多资产，取最大值）
    # def test_payoff(*l):
    # return max(np.max(l) - strike, 0)  # 最高资产价格 - strike，最大值为0
 
    paths_generater = Paths_generater(T=T, days=50, S0_vec=S0_vec, r=r, vol_vec=sigma_vec, dividend_vec=dividend_vec, corr_mat=corr_mat) 
    
    test_price_american_option = MC_American_Option(paths_generater,payoff) 
    V = test_price_american_option.price(n_simulations=3000)
    print(f"计算得到的美式期权价格: {V:.8f}")
    # assert abs(V - 0.07187167189125372) < 1e-10
    # Assert that the calculated price is close to the expected price
    # assert isclose(calculated_price, expected_price, rel_tol=0.1),

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

    def gbm(self, n_simulations: int)-> np.ndarray:
        price_paths =[]    # np.zeros((n_simulations, self.asset_num, self.days + 1))
        L = np.linalg.cholesky(self.corr_mat)
        drift_vec =  self.r - self.dividend_vec  
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
    def __init__(self, paths_generater: Paths_generater, payoff_func):
        self.paths_generater = paths_generater
        self.payoff_func = payoff_func

 
    def price(self, n_simulations=1000):
        self.simulations_paths = self.paths_generater.gbm(n_simulations=n_simulations)
        # print("simulations_paths shape:", self.simulations_paths)
        cash_flow = np.zeros([n_simulations, self.paths_generater.days+1])

        for t in range(self.paths_generater.days, 0, -1):
            # print("Current time step t:-------", t)
            # print('cash_flow:', cash_flow)

            prices_at_t = self.simulations_paths[:, :, t]

            if t == self.paths_generater.days:
                maturity_price = prices_at_t
                maturity_payoff = np.array(list(map(self.payoff_func, maturity_price)))
                cash_flow[:, -1] = maturity_payoff
            else:
                # showlog = False
                # if t==298 or t==297:
                #     showlog = True
                # 使用正确的折现方法，找到未来最后一个非零现金流并折现到当前时刻
                discounted_cashflow = self._get_discounted_cashflow(t, cash_flow, n_simulations)
                # if showlog:
                #     print("discounted_cashflow:", discounted_cashflow)
                r = cqf_Regression.Regression(prices_at_t, discounted_cashflow, payoff_func=self.payoff_func)
                # if showlog:
                #     print("Regression object:", r.has_intrinsic_value, r.index)
                if r.has_intrinsic_value:
                    all_cur_payoff = np.array(list(map(self.payoff_func, prices_at_t)))
                    continuation_value = np.array([r.evaluate(X) for X in prices_at_t[r.index]])

                    exercise_index = r.index[all_cur_payoff[r.index] >= continuation_value]
                    # if showlog:
                    #     print("exercise_index:", exercise_index)
        
                    if len(exercise_index) > 0:
                        cash_flow[exercise_index] = np.zeros(cash_flow[exercise_index].shape)
                        cash_flow[exercise_index, t] = all_cur_payoff[exercise_index]
                
                else:
                    continue
        
        return self._get_discounted_cashflow_at_t0(cash_flow)


    def _get_discounted_cashflow(self, t: int, cashflow_matrix: np.ndarray,path_num) -> np.ndarray:
        """
        最终优化版本：推荐使用
        
        算法思路：
        1. 使用高效的方法从后向前搜索第一个非零现金流
        2. 最小化内存使用
        3. 代码简洁易懂
        
        参数:
        - t: 当前时间点
        - cashflow_matrix: 现金流矩阵
        
        返回: 折现现金流数组
        """
        # N = self.paths_generater.N
        # ir = american_instance.random_walk.ir
        # dt = american_instance.random_walk.dt
        
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


    def  _get_discounted_cashflow_at_t0(self, cashflow_matrix):
        """
        与 American.py 保持一致的实现
        
        参数:
        - cashflow_matrix: 现金流矩阵
        
        返回: 平均折现价值
        """
        # ir = 
        # dt = 
        
        # 提取第1列到最后一列的数据 (与 American.py 一致)
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

if __name__ == "__main__":
    set_seed(444)
    r = 0.03
    random_walk = Paths_generater(3, 3, np.ones(1), r, np.ones(1), np.zeros(1), np.eye(1))
    def test_payoff(*l):
        return max(3 - np.sum(l), 0)
    opt = MC_American_Option( random_walk,test_payoff)
    
    cashflow_matrix = np.array([[0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 2]])
    t=2
    print("cashflow_matrix:", cashflow_matrix[:, t + 1])
    discounted = cashflow_matrix[:, t + 1] * np.exp(-random_walk.r * random_walk.dt)

    print(discounted)  #[2.9113366 0.        1.94089107]
    assert sum(abs(discounted - np.array([2.9113366, 0, 1.94089107]))) < 0.00000001
    print("-----", sum(abs(discounted - np.array([2.9113366, 0, 1.94089107])))) #3.548508153983221e-09

    cashflow_matrix2 = np.array([[0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 2]])
    t=0
    discounted2 = cashflow_matrix2[:, t + 1] * np.exp(-random_walk.r * random_walk.dt)
    print(discounted2)  #[2.8252936 0.        1.82786237]
    print("-----", sum(abs(discounted2 - np.array([2.8252936, 0, 1.82786237])))) #1.2952021677392622e-09
    assert sum(abs(discounted2 - np.array([2.8252936, 0, 1.82786237]))) < 0.00000001


    # test_price_american_option()

#     "这个文件涉及的知识点 和数学原理说一下"点击查看元宝的回答
# https://yb.tencent.com/s/bBg8i1iF9hh7

# https://yb.tencent.com/s/8z69z9n8CSWM