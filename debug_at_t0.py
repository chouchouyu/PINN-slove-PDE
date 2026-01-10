import numpy as np
import sys
sys.path.append("/Users/susan/Downloads/hdp-master/src")

from blackscholes.mc.American import American as AmericanRef

class MockRandomWalkRef:
    def __init__(self):
        self.T = 3
        self.days = 3
        self.N = 3
        self.dt = 1
        self.s0 = np.ones(1)
        self.ir = 0.03
        self.sigma = np.ones(1)
        self.dividend = np.zeros(1)
        self.corr_mat = np.eye(1)
    
    def simulateV2(self, n_simulations=1000):
        return np.zeros((n_simulations, 1, 4))

random_walk_ref = MockRandomWalkRef()

def test_payoff(*l):
    return max(3 - np.sum(l), 0)

opt_ref = AmericanRef(test_payoff, random_walk_ref)

cashflow_matrix = np.array([[0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 2]])

print("测试 _get_discounted_cashflow_at_t0")
print(f"cashflow_matrix:\n{cashflow_matrix}")

future_cashflows = cashflow_matrix[:, 1:]
print(f"\nfuture_cashflows (第1列到最后):\n{future_cashflows}")

first_nonzero_positions = np.argmax(future_cashflows != 0, axis=1)
print(f"first_nonzero_positions: {first_nonzero_positions}")

has_cashflow = np.any(future_cashflows != 0, axis=1)
print(f"has_cashflow: {has_cashflow}")

time_indices_correct = first_nonzero_positions + 1
print(f"time_indices = first_nonzero_positions + 1 = {time_indices_correct}")

discount_factors = np.exp(-0.03 * time_indices_correct * 1)
print(f"discount_factors: {discount_factors}")

discounted_values = future_cashflows[np.arange(len(cashflow_matrix)), first_nonzero_positions] * discount_factors
print(f"discounted_values: {discounted_values}")

result = discounted_values[has_cashflow].mean()
print(f"result (平均): {result}")

print("\n" + "="*50)
print("参考实现结果:")
result_ref = opt_ref._get_discounted_cashflow_at_t0(cashflow_matrix)
print(f"result_ref: {result_ref}")
