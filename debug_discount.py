import numpy as np

cashflow_matrix = np.array([[0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 2]])
t = 2

future_cashflows = cashflow_matrix[:, t+1:]

print(f"t = {t}")
print(f"future_cashflows (t+1到末尾):\n{future_cashflows}")

reversed_mask = future_cashflows[:, ::-1] != 0
print(f"\nreversed_mask:\n{reversed_mask}")

last_nonzero_positions = reversed_mask.shape[1] - np.argmax(reversed_mask, axis=1) - 1
print(f"last_nonzero_positions: {last_nonzero_positions}")

time_indices = t + 1 + last_nonzero_positions
print(f"time_indices = t + 1 + last_nonzero_positions = {t} + 1 + {last_nonzero_positions} = {time_indices}")

# 正确的公式
time_indices_correct = last_nonzero_positions + 1
print(f"time_indices_correct = last_nonzero_positions + 1 = {last_nonzero_positions} + 1 = {time_indices_correct}")

r = 0.03
dt = 1

discount_factors_wrong = np.exp(-r * (time_indices - t) * dt)
discount_factors_correct = np.exp(-r * time_indices_correct * dt)

print(f"\n折现因子 (错误): {discount_factors_wrong}")
print(f"折现因子 (正确): {discount_factors_correct}")

result_wrong = future_cashflows[np.arange(len(cashflow_matrix)), last_nonzero_positions] * discount_factors_wrong
result_correct = future_cashflows[np.arange(len(cashflow_matrix)), last_nonzero_positions] * discount_factors_correct

print(f"\n结果 (错误): {result_wrong}")
print(f"结果 (正确): {result_correct}")
print(f"期望值: [2.9113366, 0, 1.94089107]")
