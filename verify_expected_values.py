import numpy as np
import sys
sys.path.append("/Users/susan/Downloads/hdp-master/src")

from blackscholes.mc.American import American as AmericanRef

class MockRandomWalk:
    def __init__(self, T=3, days=3, s0=None, r=0.03, sigma=None, dividend=None, corr_mat=None):
        self.T = T
        self.days = days
        self.dt = T / days
        self.N = days
        self.s0 = s0 if s0 is not None else np.ones(1)
        self.ir = r
        self.sigma = sigma if sigma is not None else np.ones(1)
        self.dividend = dividend if dividend is not None else np.zeros(1)
        self.corr_mat = corr_mat if corr_mat is not None else np.eye(1)
    
    def simulateV2(self, n_simulations=1000):
        return np.zeros((n_simulations, 1, self.N + 1))

def test_payoff(*l):
    return max(3 - np.sum(l), 0)

print("=" * 60)
print("Testing _get_discounted_cashflow from reference implementation")
print("=" * 60)

random_walk = MockRandomWalk(T=3, days=3, s0=np.ones(1), r=0.03, sigma=np.ones(1), dividend=np.zeros(1), corr_mat=np.eye(1))
opt_ref = AmericanRef(test_payoff, random_walk)

print(f"\nRandom walk parameters:")
print(f"  T = {random_walk.T}")
print(f"  days = {random_walk.days}")
print(f"  dt = {random_walk.dt}")
print(f"  ir = {random_walk.ir}")
print(f"  N = {random_walk.N}")

print(f"\nDiscount factor at t=2 (dt=1, r=0.03):")
print(f"  exp(-0.03 * 1) = {np.exp(-0.03 * 1):.10f}")

cashflow_matrix = np.array([[0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 2]])
print(f"\nTest case 1: cashflow_matrix = {cashflow_matrix.tolist()}")
print(f"  t = 2")

discounted = opt_ref._get_discounted_cashflow(2, cashflow_matrix, 3)
print(f"  Discounted cashflow (reference): {discounted}")
print(f"  Expected discounted value for path 0: 3 * exp(-0.03 * 1) = {3 * np.exp(-0.03 * 1):.10f}")
print(f"  Expected discounted value for path 2: 2 * exp(-0.03 * 1) = {2 * np.exp(-0.03 * 1):.10f}")

cashflow_matrix2 = np.array([[0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 2]])
print(f"\nTest case 2: cashflow_matrix = {cashflow_matrix2.tolist()}")
print(f"  t = 0")

discounted2 = opt_ref._get_discounted_cashflow(0, cashflow_matrix2, 3)
print(f"  Discounted cashflow (reference): {discounted2}")
print(f"  Expected discounted value for path 0: 3 * exp(-0.03 * 3) = {3 * np.exp(-0.03 * 3):.10f}")
print(f"  Expected discounted value for path 2: 2 * exp(-0.03 * 3) = {2 * np.exp(-0.03 * 3):.10f}")

print("\n" + "=" * 60)
print("Verifying _get_discounted_cashflow_at_t0")
print("=" * 60)

cashflow_matrix_t0 = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 2, 0, 0]])
print(f"\nTest case 3: cashflow_matrix = {cashflow_matrix_t0.tolist()}")

discounted_t0 = opt_ref._get_discounted_cashflow_at_t0(cashflow_matrix_t0)
print(f"  Discounted at t0 (reference): {discounted_t0}")

cashflow_matrix_t0_2 = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 2, 0, 0]])
print(f"\nTest case 4 (same as expected in test):")
discounted_t0_2 = opt_ref._get_discounted_cashflow_at_t0(cashflow_matrix_t0_2)
print(f"  Discounted at t0 (reference): {discounted_t0_2}")
