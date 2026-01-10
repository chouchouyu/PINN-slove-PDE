import numpy as np
import sys
sys.path.append("/Users/susan/Downloads/hdp-master/src")

np.random.seed(555)

print("=" * 70)
print("逐行对比GBM实现")
print("=" * 70)

init_price_vec = np.array([100.0, 100.0])
vol_vec = np.array([0.2, 0.2])
ir = 0.05
dividend_vec = np.array([0.1, 0.1])
corr_mat = np.eye(2)
corr_mat[0, 1] = 0.3
corr_mat[1, 0] = 0.3
T = 1
days = 300

dt = T / days
sqrt_dt = np.sqrt(dt)

drift_vec = np.array([ir] * len(dividend_vec)) - np.array(dividend_vec)
vol_vec_np = np.array(vol_vec)
L = np.linalg.cholesky(corr_mat)

print(f"\n参数:")
print(f"  dt = {dt}")
print(f"  drift_vec = {drift_vec}")
print(f"  vol_vec = {vol_vec_np}")
print(f"  drift_vec - vol^2/2 = {drift_vec - vol_vec_np**2/2}")

print("\n" + "=" * 70)
print("GBM simulateV2 手动实现")
print("=" * 70)

n_simulations = 1
n = len(init_price_vec)

np.random.seed(555)

sim = np.zeros((n, days + 1))
sim[:, 0] = init_price_vec

print(f"\n初始状态 sim[:, 0] = {sim[:, 0]}")

for i in range(1, days + 1):
    dW = L @ np.random.randn(n) * sqrt_dt
    rand_term = np.multiply(vol_vec_np, dW)
    sim[:, i] = np.multiply(sim[:, i-1], np.exp((drift_vec - vol_vec_np**2/2) * dt + rand_term))

print(f"GBM路径 (资产0, t=0..5): {sim[0, :6]}")
print(f"GBM路径 (资产1, t=0..5): {sim[1, :6]}")

print("\n" + "=" * 70)
print("优化代码的手动实现")
print("=" * 70)

np.random.seed(555)

sim2 = np.zeros((n, days + 1))
sim2[:, 0] = init_price_vec

r = ir
sigma = vol_vec_np
dividend = dividend_vec

drift = (r - dividend - 0.5 * sigma ** 2) * dt

print(f"\n参数:")
print(f"  drift = (r - dividend - 0.5*sigma^2)*dt = {drift}")

for i in range(1, days + 1):
    dW = L @ np.random.randn(n) * sqrt_dt
    log_returns = drift + dW
    sim2[:, i] = sim2[:, i - 1] * np.exp(log_returns)

print(f"优化路径 (资产0, t=0..5): {sim2[0, :6]}")
print(f"优化路径 (资产1, t=0..5): {sim2[1, :6]}")

print("\n" + "=" * 70)
print("对比")
print("=" * 70)
print(f"资产0差异 (t=0..5): {sim[0, :6] - sim2[0, :6]}")
print(f"资产1差异 (t=0..5): {sim[1, :6] - sim2[1, :6]}")
