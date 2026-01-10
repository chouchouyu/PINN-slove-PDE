import numpy as np
import sys
sys.path.append("/Users/susan/PINN-slove-PDE")

from cqf_1_mc_American_optimized import MC_American_Option, Paths_generater

print("=" * 60)
print("Testing _get_discounted_cashflow from optimized implementation")
print("=" * 60)

random_walk = Paths_generater(T=3, days=3, s0=np.ones(1), r=0.03, sigma=np.ones(1), dividend=np.zeros(1), corr_mat=np.eye(1))

def test_payoff(*l):
    return max(3 - np.sum(l), 0)

opt = MC_American_Option(random_walk, test_payoff)

print(f"\nRandom walk parameters:")
print(f"  T = {random_walk.T}")
print(f"  days = {random_walk.days}")
print(f"  dt = {random_walk.dt}")
print(f"  r = {random_walk.r}")

print(f"\nDiscount factor at t=2 (dt=1, r=0.03):")
print(f"  exp(-0.03 * 1) = {np.exp(-0.03 * 1):.10f}")

cashflow_matrix = np.array([[0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 2]])
print(f"\nTest case 1: cashflow_matrix = {cashflow_matrix.tolist()}")
print(f"  t = 2")

discounted = opt._get_discounted_cashflow(2, cashflow_matrix, 3)
print(f"  Discounted cashflow (optimized): {discounted}")
print(f"  Expected (from reference): [2.9113366, 0, 1.94089107]")

cashflow_matrix2 = np.array([[0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 2]])
print(f"\nTest case 2: cashflow_matrix = {cashflow_matrix2.tolist()}")
print(f"  t = 0")

discounted2 = opt._get_discounted_cashflow(0, cashflow_matrix2, 3)
print(f"  Discounted cashflow (optimized): {discounted2}")
print(f"  Expected (from reference): [2.8252936, 0, 1.82786237]")

print("\n" + "=" * 60)
print("Verifying _get_discounted_cashflow_at_t0")
print("=" * 60)

cashflow_matrix_t0 = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 2, 0, 0]])
print(f"\nTest case 3: cashflow_matrix = {cashflow_matrix_t0.tolist()}")

discounted_t0 = opt._get_discounted_cashflow_at_t0(cashflow_matrix_t0)
print(f"  Discounted at t0 (optimized): {discounted_t0}")
print(f"  Expected (from reference): 1.4413278003406325")

print("\n" + "=" * 60)
print("Debugging the _get_discounted_cashflow method")
print("=" * 60)

print("\nDebugging test case 1 (t=2, cashflow_matrix = [[0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 2]])")

cashflow = cashflow_matrix
t = 2
n_simulations = 3
r = random_walk.r
dt = random_walk.dt
days = random_walk.days

print(f"  t={t}, days={days}")
print(f"  cashflow shape: {cashflow.shape}")
print(f"  cashflow[:, t+1:days+1] = cashflow[:, {t+1}:{days+1}]")

future_cashflows = cashflow[:, t+1:days+1]
print(f"  future_cashflows shape: {future_cashflows.shape}")
print(f"  future_cashflows:\n{future_cashflows}")

reversed_mask = future_cashflows[:, ::-1] != 0
print(f"  reversed_mask:\n{reversed_mask}")

last_nonzero_positions = reversed_mask.shape[1] - np.argmax(reversed_mask, axis=1) - 1
print(f"  last_nonzero_positions: {last_nonzero_positions}")

time_indices = t + 1 + last_nonzero_positions
print(f"  time_indices: {time_indices}")

discount_factors = np.exp(-r * (time_indices - t) * dt)
print(f"  discount_factors: {discount_factors}")

result = future_cashflows[np.arange(len(cashflow)), last_nonzero_positions] * discount_factors
print(f"  result: {result}")
