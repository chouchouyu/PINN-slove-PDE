import numpy as np
import sys
sys.path.append("/Users/susan/Downloads/hdp-master/src")
sys.path.append("/Users/susan/PINN-slove-PDE")

np.random.seed(123)
strike = 100
asset_num = 100
init_price_vec = 100 + np.random.randn(asset_num) * 5
vol_vec = 0.2 + np.random.rand(asset_num) * 0.1
ir = 0.05
dividend_vec = 0.02 + np.random.rand(asset_num) * 0.03

corr_mat = np.eye(asset_num)
for i in range(asset_num):
    for j in range(i + 1, asset_num):
        corr = np.random.rand() * 0.1
        corr_mat[i, j] = corr
        corr_mat[j, i] = corr

print("=" * 70)
print("比较 100D 路径生成")
print("=" * 70)

from blackscholes.utils.GBM import GBM
from cqf_1_mc_American_optimized import MC_American_Option, Paths_generater

np.random.seed(123)
gbm = GBM(T=1, N=50, init_price_vec=init_price_vec, ir=ir, vol_vec=vol_vec, dividend_vec=dividend_vec, corr_mat=corr_mat)
simulation_result_ref = gbm.simulateV2(100)

random_walk_opt = Paths_generater(T=1, days=50, s0=init_price_vec, r=ir, sigma=vol_vec, dividend=dividend_vec, corr_mat=corr_mat)
np.random.seed(123)
opt_paths = random_walk_opt.gbm(n_simulations=100)

print(f"\nsimulation_result_ref shape: {simulation_result_ref.shape}")
print(f"opt_paths shape: {opt_paths.shape}")

print(f"\n路径差异 (最大绝对值): {np.max(np.abs(simulation_result_ref - opt_paths))}")

print("\n检查几个时间点的差异:")
for t in [0, 10, 25, 50]:
    diff = np.max(np.abs(simulation_result_ref[:, :, t] - opt_paths[:, :, t]))
    print(f"  t={t}: 最大差异 = {diff}")

print("\n检查初始价格和最终价格:")
print(f"  参考初始价格 (前5): {simulation_result_ref[:, :, 0][:5, :3]}")
print(f"  优化初始价格 (前5): {opt_paths[:, :, 0][:5, :3]}")
print(f"  参考最终价格 (前5): {simulation_result_ref[:, :, -1][:5, :3]}")
print(f"  优化最终价格 (前5): {opt_paths[:, :, -1][:5, :3]}")
