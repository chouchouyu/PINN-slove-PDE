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

print("=" * 70)
print("比较 Regression 类的实现")
print("=" * 70)

from blackscholes.utils.Regression import Regression as RegressionRef
from cqf_1_Regression_optimized import Regression as RegressionOpt

print("\n参考 Regression 类:")
print(RegressionRef.__doc__)

print("\n优化 Regression 类:")
print(RegressionOpt.__doc__)

print("\n" + "=" * 70)
print("使用相同的输入测试 Regression 类")
print("=" * 70)

np.random.seed(555)
n_simulations = 1000
n_assets = 2

gbm = GBM(T=T, N=days, init_price_vec=init_price_vec, ir=ir, vol_vec=vol_vec, dividend_vec=dividend_vec, corr_mat=corr_mat)
simulation_result = gbm.simulateV2(n_simulations)

prices_at_t = simulation_result[:, :, 200]
discounted_cashflow = np.random.rand(n_simulations) * 10

def test_payoff(*l):
    return max(np.max(l) - 100, 0)

print(f"\nprices_at_t shape: {prices_at_t.shape}")

X_ref = prices_at_t.reshape(n_simulations, n_assets, 1)
X_opt = prices_at_t.reshape(n_simulations, n_assets, 1)

r_ref = RegressionRef(X_ref, discounted_cashflow, payoff_func=test_payoff)
r_opt = RegressionOpt(X_opt, discounted_cashflow, payoff_func=test_payoff)

print(f"\n参考 Regression:")
print(f"  has_intrinsic_value: {r_ref.has_intrinsic_value}")
print(f"  index length: {len(r_ref.index) if r_ref.index is not None else 0}")

print(f"\n优化 Regression:")
print(f"  has_intrinsic_value: {r_opt.has_intrinsic_value}")
print(f"  index length: {len(r_opt.index) if r_opt.index is not None else 0}")

if r_ref.has_intrinsic_value and r_opt.has_intrinsic_value:
    print(f"\nindex 差异:")
    print(f"  参考 index[:10]: {r_ref.index[:10]}")
    print(f"  优化 index[:10]: {r_opt.index[:10]}")
    
    if len(r_ref.index) == len(r_opt.index):
        index_match = np.array_equal(r_ref.index, r_opt.index)
        print(f"  Index 相同: {index_match}")
        
        if not index_match:
            print(f"  不同索引数量: {len(set(r_ref.index) - set(r_opt.index))}")

print("\n" + "=" * 70)
print("比较完整的 American Option pricing")
print("=" * 70)

def test_payoff2(*l):
    return max(np.max(l) - 100, 0)

gbm_for_ref = GBM(T=T, N=days, init_price_vec=init_price_vec, ir=ir, vol_vec=vol_vec, dividend_vec=dividend_vec, corr_mat=corr_mat)

print("\n参考实现:")
np.random.seed(555)
opt_ref = AmericanRef(test_payoff2, gbm_for_ref)
price_ref = opt_ref.price(1000)
print(f"  价格: {price_ref}")

print("\n优化实现:")
random_walk_opt = Paths_generater(T=T, days=days, s0=init_price_vec, r=ir, sigma=vol_vec, dividend=dividend_vec, corr_mat=corr_mat)
np.random.seed(555)
opt_opt = MC_American_Option(random_walk_opt, test_payoff2)
price_opt = opt_opt.price(1000)
print(f"  价格: {price_opt}")

print(f"\n差异: {abs(price_ref - price_opt)}")
print(f"相对差异: {abs(price_ref - price_opt) / price_ref * 100:.4f}%")
