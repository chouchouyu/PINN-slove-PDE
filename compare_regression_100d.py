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
print("比较 100D Regression 类")
print("=" * 70)

from blackscholes.utils.Regression import Regression as RegressionRef
from cqf_1_Regression_optimized import Regression as RegressionOpt
from cqf_1_mc_American_optimized import Paths_generater

np.random.seed(123)
random_walk_opt = Paths_generater(T=1, days=50, s0=init_price_vec, r=ir, sigma=vol_vec, dividend=dividend_vec, corr_mat=corr_mat)
opt_paths = random_walk_opt.gbm(n_simulations=100)

t = 25
prices_2d = opt_paths[:, :, t]
prices_3d = prices_2d.reshape(100, asset_num, 1)

discounted_cashflow = np.random.rand(100) * 10

def test_payoff(*l):
    return max(np.max(l) - strike, 0)

print(f"\nprices_2d shape: {prices_2d.shape}")
print(f"prices_3d shape: {prices_3d.shape}")

print("\n测试 2D 输入:")
r_ref_2d = RegressionRef(prices_2d, discounted_cashflow, payoff_func=test_payoff)
r_opt_2d = RegressionOpt(prices_2d, discounted_cashflow, payoff_func=test_payoff)

print(f"  参考 has_intrinsic_value: {r_ref_2d.has_intrinsic_value}")
print(f"  优化 has_intrinsic_value: {r_opt_2d.has_intrinsic_value}")
print(f"  参考 index length: {len(r_ref_2d.index)}")
print(f"  优化 index length: {len(r_opt_2d.index)}")

if r_ref_2d.has_intrinsic_value and r_opt_2d.has_intrinsic_value:
    print(f"  参考 basis 数量: {len(r_ref_2d.basis.monomials)}")
    print(f"  优化 basis 数量: {len(r_opt_2d.basis.exponent_vectors)}")
    
    test_point = prices_2d[0]
    print(f"\n  测试点 (前5): {test_point[:5]}")
    
    val_ref = r_ref_2d.evaluate(test_point)
    val_opt = r_opt_2d.evaluate(test_point)
    print(f"  参考 evaluate: {val_ref}")
    print(f"  优化 evaluate: {val_opt}")
    print(f"  差异: {abs(val_ref - val_opt)}")
