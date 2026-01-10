import numpy as np
import sys
sys.path.append("/Users/susan/Downloads/hdp-master/src")
sys.path.append("/Users/susan/PINN-slove-PDE")

np.random.seed(555)

from blackscholes.mc.American import American as AmericanRef
from blackscholes.utils.GBM import GBM
from cqf_1_mc_American_optimized import MC_American_Option, Paths_generater

init_price_vec = np.array([100.0, 100.0])
vol_vec = np.array([0.2, 0.2])
ir = 0.05
dividend_vec = np.array([0.1, 0.1])
corr_mat = np.eye(2)
corr_mat[0, 1] = 0.3
corr_mat[1, 0] = 0.3
T = 1
days = 300

def test_payoff(*l):
    return max(np.max(l) - 100, 0)

print("=" * 70)
print("详细比较行权逻辑")
print("=" * 70)

np.random.seed(555)
gbm = GBM(T=T, N=days, init_price_vec=init_price_vec, ir=ir, vol_vec=vol_vec, dividend_vec=dividend_vec, corr_mat=corr_mat)
simulation_result = gbm.simulateV2(100)

random_walk_opt = Paths_generater(T=T, days=days, s0=init_price_vec, r=ir, sigma=vol_vec, dividend=dividend_vec, corr_mat=corr_mat)
np.random.seed(555)
opt_paths = random_walk_opt.gbm(n_simulations=100)

print(f"simulation_result shape: {simulation_result.shape}")
print(f"opt_paths shape: {opt_paths.shape}")
print(f"路径差异: {np.max(np.abs(simulation_result - opt_paths))}")

t = 200
print(f"\n在时间点 t = {t} 比较:")

prices_ref = np.array([x[:, t] for x in simulation_result])
prices_opt = opt_paths[:, :, t]

print(f"prices_ref shape: {prices_ref.shape}")
print(f"prices_opt shape: {prices_opt.shape}")
print(f"prices差异: {np.max(np.abs(prices_ref - prices_opt))}")

cur_payoff_ref = np.array(list(map(test_payoff, prices_ref)))
cur_payoff_opt = np.array([test_payoff(*prices_opt[i]) for i in range(len(prices_opt))])

print(f"\ncur_payoff_ref (前10): {cur_payoff_ref[:10]}")
print(f"cur_payoff_opt (前10): {cur_payoff_opt[:10]}")
print(f"payoff差异: {np.max(np.abs(cur_payoff_ref - cur_payoff_opt))}")

itm_ref = np.where(cur_payoff_ref > 0)[0]
itm_opt = np.where(cur_payoff_opt > 0)[0]

print(f"\nITM 路径数量: 参考={len(itm_ref)}, 优化={len(itm_opt)}")
print(f"ITM 路径差异: {set(itm_ref) - set(itm_opt)}")

if len(itm_ref) > 0 and len(itm_opt) > 0:
    print(f"\n参考 ITM 索引 (前10): {itm_ref[:10]}")
    print(f"优化 ITM 索引 (前10): {itm_opt[:10]}")
