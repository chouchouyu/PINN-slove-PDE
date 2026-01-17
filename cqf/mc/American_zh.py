import numpy as np

from Paths_generater import Paths_generater
from Regression import Regression


class American_Option:
    def __init__(self, paths_generater: Paths_generater, payoff_func):
        self.paths_generater = paths_generater
        self.payoff_func = payoff_func

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
                # 使用修正后的_get_discounted_cashflow方法，传入n_simulations（路径数）
                discounted_cashflow = self._get_discounted_cashflow(
                    t, cash_flow, n_simulations
                )
                r = Regression(
                    prices_at_t, discounted_cashflow, payoff_func=self.payoff_func
                )
                if r.has_intrinsic_value:
                    all_cur_payoff = np.array(list(map(self.payoff_func, prices_at_t)))
                    continuation_value = np.array(
                        [r.evaluate(X) for X in prices_at_t[r.index]]
                    )
                    exercise_index = r.index[
                        all_cur_payoff[r.index] >= continuation_value
                    ]
                    if len(exercise_index) > 0:
                        cash_flow[exercise_index] = np.zeros(
                            cash_flow[exercise_index].shape
                        )
                        cash_flow[exercise_index, t] = all_cur_payoff[exercise_index]
        return self._get_discounted_cashflow_at_t0(cash_flow)

    def _get_discounted_cashflow(
        self, t: int, cashflow_matrix: np.ndarray, n_paths
    ) -> np.ndarray:
        """
        修正版本：使用正确的时间步数N（self.paths_generater.days）
        """
        N = self.paths_generater.days  # 时间步数
        time_indices = np.arange(N + 1)  # 时间点从0到N
        discount_factors = np.exp(
            (t - time_indices) * self.paths_generater.dt * self.paths_generater.r
        )

        # 创建掩码，只考虑t+1到N的时间点
        mask = np.zeros_like(cashflow_matrix, dtype=bool)
        mask[:, t + 1 : N + 1] = cashflow_matrix[:, t + 1 : N + 1] != 0

        # 倒序查找每行第一个非零现金流的位置
        reversed_mask = np.fliplr(mask)
        reversed_indices = np.argmax(reversed_mask, axis=1)
        first_nonzero_indices = mask.shape[1] - reversed_indices - 1

        # 检查是否有非零值
        has_nonzero = np.any(mask, axis=1)

        # 计算结果
        result = np.zeros(n_paths)
        result[has_nonzero] = (
            cashflow_matrix[has_nonzero, first_nonzero_indices[has_nonzero]]
            * discount_factors[first_nonzero_indices[has_nonzero]]
        )
        return result

    def _get_discounted_cashflow_at_t0(self, cashflow_matrix):
        """
        与 American.py 保持一致的实现

        参数:
        - cashflow_matrix: 现金流矩阵

        返回: 平均折现价值
        """
        # 提取第1列到最后一列的数据 (与 American.py 一致)
        future_cashflows = cashflow_matrix[:, 1:]

        # 查找每行第一个非零值的位置
        first_nonzero_positions = np.argmax(future_cashflows != 0, axis=1)

        # 计算对应的时间索引 (从1开始)
        time_indices = first_nonzero_positions + 1

        # 计算折扣因子
        discount_factors = np.exp(
            -self.paths_generater.r * time_indices * self.paths_generater.dt
        )

        # 获取对应的现金流值并折扣
        discounted_values = (
            future_cashflows[np.arange(len(cashflow_matrix)), first_nonzero_positions]
            * discount_factors
        )

        # 对所有路径取平均，包括没有现金流的路径 (与 American.py 一致)
        return discounted_values.mean()
