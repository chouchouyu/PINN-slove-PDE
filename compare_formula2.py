import numpy as np
import sys
sys.path.append("/Users/susan/Downloads/hdp-master/src")

np.random.seed(555)

print("=" * 70)
print("逐个随机数对比")
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

n = len(init_price_vec)

print("\n" + "=" * 70)
print("GBM的随机数生成方式")
print("=" * 70)

np.random.seed(555)
dW_gbm = L @ np.random.randn(n) * sqrt_dt
rand_term_gbm = np.multiply(vol_vec_np, dW_gbm)

print(f"dW (GBM) = {dW_gbm}")
print(f"rand_term (GBM) = vol_vec * dW = {rand_term_gbm}")
print(f"漂移项 = (drift_vec - vol_vec^2/2) * dt = {(drift_vec - vol_vec_np**2/2) * dt}")

print("\n" + "=" * 70)
print("优化代码的随机数生成方式")
print("=" * 70)

np.random.seed(555)
dW_opt = L @ np.random.randn(n) * sqrt_dt
drift = (ir - dividend_vec - 0.5 * vol_vec_np ** 2) * dt
log_returns_opt = drift + dW_opt

print(f"dW (优化) = {dW_opt}")
print(f"log_returns (优化) = drift + dW = {log_returns_opt}")

print("\n对比:")
print(f"GBM rand_term = {rand_term_gbm}")
print(f"优化 log_returns = {log_returns_opt}")
print(f"差异 = {rand_term_gbm - log_returns_opt}")

print("\n" + "=" * 70)
print("关键差异")
print("=" * 70)
print("GBM的dW已经是cholesky变换后的结果，不需要再乘vol_vec")
print("GBM的rand_term = vol_vec * dW = vol_vec * L @ randn * sqrt_dt")
print("优化的log_returns = (r - q - 0.5*sigma^2)*dt + dW")
print("                        = (r - q - 0.5*sigma^2)*dt + L @ randn * sqrt_dt")
print("\n所以GBM公式等价于:")
print("  drift + sigma * L @ randn * sqrt_dt")
print("  = (r - q - sigma^2/2)*dt + sigma * L @ randn * sqrt_dt")
print("\n而优化公式是:")
print("  drift + L @ randn * sqrt_dt")
print("  = (r - q - sigma^2/2)*dt + L @ randn * sqrt_dt")
print("\n差异在于: GBM乘了sigma，优化没有乘sigma")
