import numpy as np
import sys
sys.path.append("/Users/susan/Downloads/hdp-master/src")

np.random.seed(555)

print("=" * 70)
print("使用相同随机数直接比较")
print("=" * 70)

strike = 100
asset_num = 2
init_price_vec = np.array([100.0, 100.0])
vol_vec = np.array([0.2, 0.2])
ir = 0.05
dividend_vec = np.array([0.1, 0.1])
corr_mat = np.eye(asset_num)
corr_mat[0, 1] = 0.3
corr_mat[1, 0] = 0.3
T = 1
days = 300

def test_payoff(*l):
    return max(np.max(l) - strike, 0)

from cqf_1_mc_American_optimized import MC_American_Option, Paths_generater as Paths_generater_opt

random_walk_opt = Paths_generater_opt(T=T, days=days, s0=init_price_vec, r=ir, sigma=vol_vec, dividend=dividend_vec, corr_mat=corr_mat)

n_simulations = 3000

print("\n生成路径数据 (使用相同的随机数种子)...")
opt_paths = random_walk_opt.gbm(n_simulations=n_simulations)

print(f"优化路径形状: {opt_paths.shape}")
print(f"路径数据前几个值 (路径0, 资产0, t=0,1,2): {opt_paths[0, 0, :3]}")
print(f"路径数据前几个值 (路径1, 资产0, t=0,1,2): {opt_paths[1, 0, :3]}")

print("\n计算优化价格...")
opt = MC_American_Option(random_walk_opt, test_payoff)
opt.simulations_paths = opt_paths
price_opt = opt.price(n_simulations)
print(f"优化价格: {price_opt}")

print("\n" + "=" * 70)
print("使用 GBM 类的路径生成方式")
print("=" * 70)

from blackscholes.utils.GBM import GBM
from blackscholes.mc.American import American

gbm = GBM(T=T, N=days, init_price_vec=init_price_vec, ir=ir, vol_vec=vol_vec, dividend_vec=dividend_vec, corr_mat=corr_mat)

print("\n生成路径数据...")
np.random.seed(555)
gbm_paths = gbm.simulateV2(n_simulations)

print(f"GBM路径形状: {gbm_paths.shape}")
print(f"GBM路径数据前几个值 (路径0, 资产0, t=0,1,2): {gbm_paths[0, 0, :3]}")
print(f"GBM路径数据前几个值 (路径1, 资产0, t=0,1,2): {gbm_paths[1, 0, :3]}")

print("\n计算GBM价格...")
american = American(test_payoff, gbm)
price_gbm = american.price(n_simulations)
print(f"GBM价格: {price_gbm}")

print("\n" + "=" * 70)
print("结果比较")
print("=" * 70)
print(f"优化价格: {price_opt:.8f}")
print(f"GBM价格: {price_gbm:.8f}")
print(f"差异: {abs(price_opt - price_gbm):.8f}")
print(f"相对差异: {abs(price_opt - price_gbm) / price_gbm * 100:.4f}%")
