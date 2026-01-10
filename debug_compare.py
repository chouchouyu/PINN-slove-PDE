import numpy as np
import sys
sys.path.append("/Users/susan/Downloads/hdp-master/src")

from blackscholes.mc.American import American
from blackscholes.utils.GBM import GBM

from cqf_1_mc_American_optimized import MC_American_Option, Paths_generater

np.random.seed(555)

print("=" * 70)
print("调试：比较原始实现和优化实现")
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

print("\n1. 路径生成比较:")
print("-" * 50)

random_walk_opt = Paths_generater(T=T, days=days, s0=init_price_vec, r=ir, sigma=vol_vec, dividend=dividend_vec, corr_mat=corr_mat)
gbm_orig = GBM(T=T, N=days, init_price_vec=init_price_vec, ir=ir, vol_vec=vol_vec, dividend_vec=dividend_vec, corr_mat=corr_mat)

paths_opt = random_walk_opt.gbm(n_simulations=10)
paths_orig = gbm_orig.simulateV2(10)

print(f"优化路径形状: {paths_opt.shape}")
print(f"原始路径形状: {paths_orig.shape}")

print(f"\n优化路径 - 第0个路径的资产0价格 (t=0,1,2): {paths_opt[0, 0, :3]}")
print(f"优化路径 - 第0个路径的资产1价格 (t=0,1,2): {paths_opt[0, 1, :3]}")

print(f"原始路径 - 第0个路径的资产0价格 (t=0,1,2): {paths_orig[0, 0, :3]}")
print(f"原始路径 - 第0个路径的资产1价格 (t=0,1,2): {paths_orig[0, 1, :3]}")

print("\n2. Regression 输入格式比较:")
print("-" * 50)

t = 200
prices_at_t_orig = np.array([x[:, t] for x in paths_orig])
prices_at_t_opt = paths_opt[:, :, t]

print(f"t={t}")
print(f"原始 Regression 输入 X 形状: {prices_at_t_orig.shape}")
print(f"优化 Regression 输入 X 形状: {prices_at_t_opt.shape}")

print(f"\n原始 Regression X[0]: {prices_at_t_orig[0]}")
print(f"优化 Regression X[0]: {prices_at_t_opt[0]}")

print("\n3. 现金流计算逻辑比较:")
print("-" * 50)

cash_flow_orig = np.zeros([10, days + 1])
cash_flow_opt = np.zeros([10, days + 1])

cash_flow_orig[:, -1] = np.array(list(map(test_payoff, [paths_orig[i, :, -1] for i in range(10)])))
cash_flow_opt[:, -1] = np.array(list(map(test_payoff, [paths_opt[i, :, -1] for i in range(10)])))

print(f"到期日现金流 (原始): {cash_flow_orig[0, -1]:.6f}")
print(f"到期日现金流 (优化): {cash_flow_opt[0, -1]:.6f}")

print("\n4. 折现现金流函数比较:")
print("-" * 50)

from cqf_1_Regression_optimized import Regression

cash_flow_test = np.zeros([3, 4])
cash_flow_test[0, 3] = 3
cash_flow_test[2, 3] = 2

random_walk_test = Paths_generater(T=3, days=3, s0=np.ones(2), r=0.03, sigma=np.ones(2), dividend=np.zeros(2), corr_mat=np.eye(2))
opt_test = MC_American_Option(random_walk_test, test_payoff)

discounted = opt_test._get_discounted_cashflow_optimized(2, cash_flow_test, 3)
print(f"折现现金流 (优化, t=2): {discounted}")

print("\n5. 直接对比两个实现的定价结果:")
print("-" * 50)

print("\n运行原始实现...")
random_walk_orig = GBM(T=T, N=days, init_price_vec=init_price_vec, ir=ir, vol_vec=vol_vec, dividend_vec=dividend_vec, corr_mat=corr_mat)
american_orig = American(test_payoff, random_walk_orig)
price_orig = american_orig.price(100)
print(f"原始实现价格 (100 simulations): {price_orig:.6f}")

print("\n运行优化实现...")
random_walk_opt = Paths_generater(T=T, days=days, s0=init_price_vec, r=ir, sigma=vol_vec, dividend=dividend_vec, corr_mat=corr_mat)
opt = MC_American_Option(random_walk_opt, test_payoff)
price_opt = opt.price(100)
print(f"优化实现价格 (100 simulations): {price_opt:.6f}")

print("\n" + "=" * 70)
