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
print("比较 Regression 类的输入和输出")
print("=" * 70)

np.random.seed(555)
gbm = GBM(T=T, N=days, init_price_vec=init_price_vec, ir=ir, vol_vec=vol_vec, dividend_vec=dividend_vec, corr_mat=corr_mat)
simulation_result = gbm.simulateV2(1000)

t = 200

prices_ref = np.array([x[:, t] for x in simulation_result])
discounted_cashflow = np.random.rand(1000) * 10

print(f"\nprices_ref shape: {prices_ref.shape}")
print(f"discounted_cashflow shape: {discounted_cashflow.shape}")

r_ref = RegressionRef(prices_ref, discounted_cashflow, payoff_func=test_payoff)
print(f"\n参考 Regression:")
print(f"  has_intrinsic_value: {r_ref.has_intrinsic_value}")
print(f"  index length: {len(r_ref.index)}")
print(f"  dimension: {r_ref.dimension}")

prices_opt = simulation_result[:, :, t]
prices_opt_3d = prices_opt.reshape(1000, 2, 1)

r_opt = RegressionOpt(prices_opt_3d, discounted_cashflow, payoff_func=test_payoff)
print(f"\n优化 Regression:")
print(f"  has_intrinsic_value: {r_opt.has_intrinsic_value}")
print(f"  index length: {len(r_opt.index)}")

if r_ref.has_intrinsic_value and r_opt.has_intrinsic_value:
    print(f"\nindex 差异分析:")
    common_idx = set(r_ref.index) & set(r_opt.index)
    only_ref = set(r_ref.index) - set(r_opt.index)
    only_opt = set(r_opt.index) - set(r_ref.index)
    
    print(f"  共同索引数量: {len(common_idx)}")
    print(f"  仅参考有: {len(only_ref)}")
    print(f"  仅优化有: {len(only_opt)}")
    
    if len(only_ref) > 0:
        print(f"  仅参考有的索引 (前10): {list(only_ref)[:10]}")
    if len(only_opt) > 0:
        print(f"  仅优化有的索引 (前10): {list(only_opt)[:10]}")

print("\n" + "=" * 70)
print("测试 evaluate 方法")
print("=" * 70)

test_X = prices_ref[0]
print(f"\n测试点: {test_X}")
print(f"测试点形状: {test_X.shape}")

try:
    val_ref = r_ref.evaluate(test_X)
    print(f"参考 evaluate 结果: {val_ref}")
except Exception as e:
    print(f"参考 evaluate 错误: {e}")

try:
    val_opt = r_opt.evaluate(test_X)
    print(f"优化 evaluate 结果: {val_opt}")
except Exception as e:
    print(f"优化 evaluate 错误: {e}")
