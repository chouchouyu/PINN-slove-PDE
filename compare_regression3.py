import numpy as np
import sys
sys.path.append("/Users/susan/Downloads/hdp-master/src")
sys.path.append("/Users/susan/PINN-slove-PDE")

np.random.seed(555)

from blackscholes.mc.American import American as AmericanRef
from blackscholes.utils.GBM import GBM
from blackscholes.utils.Regression import Regression as RegressionRef
from cqf_1_mc_American_optimized import MC_American_Option, Paths_generater
from cqf_1_Regression_optimized import Regression as RegressionOpt

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
print("详细比较 Regression 类的拟合结果")
print("=" * 70)

np.random.seed(555)
gbm = GBM(T=T, N=days, init_price_vec=init_price_vec, ir=ir, vol_vec=vol_vec, dividend_vec=dividend_vec, corr_mat=corr_mat)
simulation_result = gbm.simulateV2(1000)

t = 200

prices_ref = np.array([x[:, t] for x in simulation_result])
discounted_cashflow = np.random.rand(1000) * 10

print(f"\nprices_ref shape: {prices_ref.shape}")

r_ref = RegressionRef(prices_ref, discounted_cashflow, payoff_func=test_payoff)
print(f"\n参考 Regression:")
print(f"  dimension: {r_ref.dimension}")
print(f"  basis.monomials 数量: {len(r_ref.basis.monomials)}")
for i, m in enumerate(r_ref.basis.monomials):
    print(f"    单项式 {i}: a_vec = {m.a_vec}")

prices_opt = simulation_result[:, :, t]
prices_opt_3d = prices_opt.reshape(1000, 2, 1)

r_opt = RegressionOpt(prices_opt_3d, discounted_cashflow, payoff_func=test_payoff)
print(f"\n优化 Regression:")
print(f"  n_features: {r_opt.n_features}")
print(f"  poly_degree: {r_opt.poly_degree}")
print(f"  n_basis_cols: {r_opt.n_basis_cols}")
print(f"  coeffs shape: {r_opt.coeffs.shape}")
print(f"  coeffs: {r_opt.coeffs}")

print("\n" + "=" * 70)
print("比较多项式基的构建")
print("=" * 70)

test_X = prices_ref[0]
print(f"\n测试点: {test_X}")

basis_ref = r_ref.basis.evaluate(test_X)
print(f"参考 basis.evaluate: {basis_ref}")

basis_opt = r_opt._create_polynomial_basis(test_X.reshape(1, -1), r_opt.poly_degree)
print(f"优化 _create_polynomial_basis:\n{basis_opt}")

print("\n" + "=" * 70)
print("使用相同的输入数据测试回归")
print("=" * 70)

X_itm_ref = prices_ref[r_ref.index]
Y_itm_ref = discounted_cashflow[r_ref.index]

X_itm_opt = prices_opt_3d[r_opt.index]
Y_itm_opt = discounted_cashflow[r_opt.index]

print(f"\nX_itm_ref shape: {X_itm_ref.shape}")
print(f"X_itm_opt shape: {X_itm_opt.shape}")

if len(r_ref.index) > 0:
    target_matrix_A = np.array([r_ref.basis.evaluate(x) for x in X_itm_ref])
    print(f"\n参考目标矩阵形状: {target_matrix_A.shape}")
    print(f"参考目标矩阵 (前5行):\n{target_matrix_A[:5]}")

if len(r_opt.index) > 0:
    basis_opt_full = r_opt._create_polynomial_basis(X_itm_opt[:, :, 0], r_opt.poly_degree)
    print(f"\n优化基矩阵形状: {basis_opt_full.shape}")
    print(f"优化基矩阵 (前5行):\n{basis_opt_full[:5]}")

print("\n" + "=" * 70)
print("验证 evaluate 方法")
print("=" * 70)

print(f"\n参考 Regression coeffs: {r_ref.coefficients}")
print(f"优化 Regression coeffs: {r_opt.coeffs}")

if len(r_ref.index) > 0 and len(r_opt.index) > 0:
    val_ref = r_ref.evaluate(test_X)
    val_opt = r_opt.evaluate(test_X)
    
    print(f"\n测试点: {test_X}")
    print(f"参考 evaluate: {val_ref}")
    print(f"优化 evaluate: {val_opt}")
    print(f"差异: {abs(val_ref - val_opt)}")
