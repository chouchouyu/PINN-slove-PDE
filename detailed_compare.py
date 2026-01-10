import numpy as np
import sys
sys.path.append("/Users/susan/Downloads/hdp-master/src")

np.random.seed(555)

print("=" * 70)
print("详细路径比较")
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

from blackscholes.utils.GBM import GBM
gbm = GBM(T=T, N=days, init_price_vec=init_price_vec, ir=ir, vol_vec=vol_vec, dividend_vec=dividend_vec, corr_mat=corr_mat)

from cqf_1_mc_American_optimized import Paths_generater as Paths_generater_opt
random_walk_opt = Paths_generater_opt(T=T, days=days, s0=init_price_vec, r=ir, sigma=vol_vec, dividend=dividend_vec, corr_mat=corr_mat)

n_simulations = 5

np.random.seed(555)
gbm_paths = gbm.simulateV2(n_simulations)

np.random.seed(555)
opt_paths = random_walk_opt.gbm(n_simulations=n_simulations)

print(f"\nGBM路径形状: {gbm_paths.shape}")
print(f"优化路径形状: {opt_paths.shape}")

print(f"\nGBM路径数据 (路径0, 资产0, t=0..5): {gbm_paths[0, 0, :6]}")
print(f"优化路径数据 (路径0, 资产0, t=0..5): {opt_paths[0, 0, :6]}")
print(f"\n差异: {gbm_paths[0, 0, :6] - opt_paths[0, 0, :6]}")

print(f"\nGBM路径数据 (路径0, 资产1, t=0..5): {gbm_paths[0, 1, :6]}")
print(f"优化路径数据 (路径0, 资产1, t=0..5): {opt_paths[0, 1, :6]}")
print(f"\n差异: {gbm_paths[0, 1, :6] - opt_paths[0, 1, :6]}")

print(f"\n路径1 (资产0, t=0..5):")
print(f"  GBM: {gbm_paths[1, 0, :6]}")
print(f"  优化: {opt_paths[1, 0, :6]}")

print(f"\n路径2 (资产0, t=0..5):")
print(f"  GBM: {gbm_paths[2, 0, :6]}")
print(f"  优化: {opt_paths[2, 0, :6]}")

print("\n" + "=" * 70)
print("使用完全相同的方式生成随机数 (不使用seed)")
print("=" * 70)

np.random.seed(555)
gbm_paths2 = gbm.simulateV2(n_simulations)

np.random.seed(555)
opt_paths2 = random_walk_opt.gbm(n_simulations=n_simulations)

print(f"\n路径0 (资产0, t=0..5):")
print(f"  GBM: {gbm_paths2[0, 0, :6]}")
print(f"  优化: {opt_paths2[0, 0, :6]}")
print(f"  差异: {np.abs(gbm_paths2[0, 0, :6] - opt_paths2[0, 0, :6])}")
print(f"  最大差异: {np.max(np.abs(gbm_paths2[0, 0, :] - opt_paths2[0, 0, :]))}")

print("\n" + "=" * 70)
print("不设置seed，直接比较")
print("=" * 70)

gbm_paths3 = gbm.simulateV2(n_simulations)
opt_paths3 = random_walk_opt.gbm(n_simulations=n_simulations)

print(f"\n路径0 (资产0, t=0..5):")
print(f"  GBM: {gbm_paths3[0, 0, :6]}")
print(f"  优化: {opt_paths3[0, 0, :6]}")

print("\n" + "=" * 70)
print("手动检查GBM的实现细节")
print("=" * 70)

print("\nGBM中的参数:")
print(f"  T: {gbm.T}")
print(f"  N: {gbm.N}")
print(f"  dt: {gbm.dt}")
print(f"  drift_vec: {[ir]*len(dividend_vec) - dividend_vec}")
print(f"  vol_vec: {vol_vec}")

print("\n优化中的参数:")
print(f"  T: {random_walk_opt.T}")
print(f"  dt: {random_walk_opt.dt}")
print(f"  drift: (r - dividend - 0.5 * sigma^2) * dt = ({ir} - {dividend_vec[0]} - 0.5 * {vol_vec[0]}^2) * {random_walk_opt.dt}")
print(f"  drift = {(ir - dividend_vec[0] - 0.5 * vol_vec[0]**2) * random_walk_opt.dt}")
