import numpy as np
import pandas as pd

cashflow_matrix = pd.read_csv('/Users/susan/PINN-slove-PDE/cash_flow.csv', header=None).values

t = 298
N = 300
ir = 0.05
dt = 1/300

print(f"cashflow_matrix shape: {cashflow_matrix.shape}")
print(f"t = {t}, N = {N}")

future_cashflows = cashflow_matrix[:, t+1:N+1]
print(f"future_cashflows shape (t+1 to N): {future_cashflows.shape}")
print(f"Column indices: {t+1} to {N}")

print(f"\nfuture_cashflows[:, 0] (column {t+1}): {future_cashflows[:, 0]}")
print(f"future_cashflows[:, 1] (column {t+2}): {future_cashflows[:, 1]}")

print(f"\nNon-zero count in each column of future_cashflows:")
for i in range(future_cashflows.shape[1]):
    non_zero_count = np.sum(future_cashflows[:, i] != 0)
    if non_zero_count > 0:
        print(f"  Column {t+1+i}: {non_zero_count} non-zero values")
        print(f"    Values: {future_cashflows[:, i][future_cashflows[:, i] != 0]}")

def hdp_discounted_cashflow(t, cashflow_matrix, N, ir, dt):
    future_cashflows = cashflow_matrix[:, t+1:N+1]
    
    reversed_mask = future_cashflows[:, ::-1] != 0
    last_nonzero_positions = reversed_mask.shape[1] - np.argmax(reversed_mask, axis=1) - 1
    
    time_indices = t + 1 + last_nonzero_positions
    
    discount_factors = np.exp(-ir * (time_indices - t) * dt)
    
    result = future_cashflows[np.arange(len(cashflow_matrix)), last_nonzero_positions] * discount_factors
    
    return result

def cqf_discounted_cashflow(t, cashflow_matrix, N, ir, dt):
    time_indices = np.arange(N+1)
    discount_factors = np.exp((t - time_indices) * dt * ir)
    
    mask = np.zeros_like(cashflow_matrix, dtype=bool)
    mask[:, t+1:N+1] = cashflow_matrix[:, t+1:N+1] != 0
    
    reversed_mask = np.fliplr(mask)
    reversed_indices = np.argmax(reversed_mask, axis=1)
    
    first_nonzero_indices = mask.shape[1] - reversed_indices - 1
    
    has_nonzero = np.any(mask, axis=1)
    
    num_paths = cashflow_matrix.shape[0]
    result = np.zeros(num_paths)
    result[has_nonzero] = cashflow_matrix[has_nonzero, first_nonzero_indices[has_nonzero]] * discount_factors[first_nonzero_indices[has_nonzero]]
    
    return result

hdp_result = hdp_discounted_cashflow(t, cashflow_matrix, N, ir, dt)
cqf_result = cqf_discounted_cashflow(t, cashflow_matrix, N, ir, dt)

print(f"\n{'='*60}")
print(f"Results at t={t}:")
print(f"{'='*60}")
print(f"hdp-master result (non-zero count): {np.sum(hdp_result != 0)}")
print(f"cqf result (non-zero count): {np.sum(cqf_result != 0)}")

non_zero_hdp = hdp_result[hdp_result != 0]
non_zero_cqf = cqf_result[cqf_result != 0]

print(f"\nhdp-master non-zero values (first 10): {non_zero_hdp[:10]}")
print(f"cqf non-zero values (first 10): {non_zero_cqf[:10]}")

print(f"\nAre they equal? {np.allclose(hdp_result, cqf_result)}")
print(f"Max absolute difference: {np.max(np.abs(hdp_result - cqf_result))}")

if not np.allclose(hdp_result, cqf_result):
    print(f"\nDifferences (non-zero):")
    diff = hdp_result - cqf_result
    diff_nonzero = diff[diff != 0]
    print(f"Number of differences: {len(diff_nonzero)}")
    print(f"Max diff: {np.max(diff_nonzero)}")
    print(f"Min diff: {np.min(diff_nonzero)}")
    print(f"Sample differences: {diff_nonzero[:10]}")
