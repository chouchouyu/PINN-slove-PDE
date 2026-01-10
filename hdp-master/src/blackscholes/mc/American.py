import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../..")
from blackscholes.utils.Regression import Regression
import numpy as np

class American:
    """
    Multi-Dimensional American Option. Priced by the Least Square Monte Carlo method.
    """
    def __init__(self, payoff_func, random_walk):
        """
        payoff: A function that takes ${asset_num} variables as input, returns the a scalar payoff
        random_walk: A random walk generator, e.g. GBM (geometric brownian motion)
        """
        self.payoff_func = payoff_func
        self.random_walk = random_walk
    
    def price(self, path_num=1000):
        """Least Square Monte Carlo method"""
        self.simulation_result = self.random_walk.simulateV2(path_num)
        print("simulation_result shape:", self.simulation_result)
        cashflow_matrix = np.zeros([path_num, self.random_walk.N+1])
        cur_price = np.array([x[:, -1] for x in self.simulation_result])
        cur_payoff = np.array(list(map(self.payoff_func, cur_price)))
        # print("cur_payoff:",cur_payoff)
        cashflow_matrix[:, self.random_walk.N] = cur_payoff

        for t in range(self.random_walk.N-1, 0, -1):
            print("Current time step t:-------", t)
            print("cash_flow:", cashflow_matrix)
            showlog = False
            if t==298 or t==297:
                showlog = True
            discounted_cashflow = self._get_discounted_cashflow(t, cashflow_matrix, path_num)
            if showlog:
                print("discounted_cashflow:", discounted_cashflow)
            # Compute the discounted payoff
            r = Regression(self.simulation_result[:, :, t], discounted_cashflow, payoff_func=self.payoff_func)
            if showlog:
                print("Regression object:", r.has_intrinsic_value, r.index)
            if not r.has_intrinsic_value: continue # Intrinsic value = 0
            cur_price = np.array([x[:, t] for x in self.simulation_result])
            cur_payoff = np.array(list(map(self.payoff_func, cur_price[r.index])))
            continuation = np.array([r.evaluate(X) for X in cur_price[r.index]])

            exercise_index = r.index[cur_payoff >= continuation]
            if showlog:
                print("exercise_index:", exercise_index)    
            
            cashflow_matrix[exercise_index] = np.zeros(cashflow_matrix[exercise_index].shape)
            cashflow_matrix[exercise_index, t] = np.array(list(map(self.payoff_func, cur_price)))[exercise_index]
            # print("cash_flow after exercise:", cashflow_matrix)

            # print(self._get_discounted_cashflow_at_t0(cashflow_matrix))
            # return
        return self._get_discounted_cashflow_at_t0(cashflow_matrix)

    def _get_discounted_cashflow(self, t, cashflow_matrix, path_num):
        N = self.random_walk.N
        ir = self.random_walk.ir
        dt = self.random_walk.dt
        
        # 提取 t+1 到 N 列的数据
        future_cashflows = cashflow_matrix[:, t+1:N+1]
        
        # 查找每行最后一个非零值的位置 (从右向左)
        # np.argmax 找到第一个 True 的位置，即从右向左的第一个非零值
        reversed_mask = future_cashflows[:, ::-1] != 0
        last_nonzero_positions = reversed_mask.shape[1] - np.argmax(reversed_mask, axis=1) - 1
        
        # 计算对应的原始时间索引
        time_indices = t + 1 + last_nonzero_positions
        
        # 计算折扣因子并应用
        discount_factors = np.exp(-ir * (time_indices - t) * dt)
        
        # 获取对应的现金流值
        result = future_cashflows[np.arange(path_num), last_nonzero_positions] * discount_factors
        
        return result

    def _get_discounted_cashflow_at_t0(self, cashflow_matrix):
        ir = self.random_walk.ir
        dt = self.random_walk.dt
        
        # 提取第1列到最后一列的数据
        future_cashflows = cashflow_matrix[:, 1:]
        
        # 查找每行第一个非零值的位置
        first_nonzero_positions = np.argmax(future_cashflows != 0, axis=1)
        
        # 检查是否存在非零值
        has_cashflow = np.any(future_cashflows != 0, axis=1)
        
        # 计算对应的时间索引 (从1开始)
        time_indices = first_nonzero_positions + 1
        
        # 计算折扣因子
        discount_factors = np.exp(-ir * time_indices * dt)
        
        # 获取对应的现金流值并折扣
        discounted_values = future_cashflows[np.arange(len(cashflow_matrix)), first_nonzero_positions] * discount_factors
        
        # 只考虑有现金流的路径
        return discounted_values[has_cashflow].mean()