import numpy as np
import sys
sys.path.append("/Users/susan/Downloads/hdp-master/src")

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

from blackscholes.utils.GBM import GBM
from blackscholes.mc.American import American

random_walk = GBM(1, 50, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)

def test_payoff(*l):
    return max(np.max(l) - strike, 0)

print("参考 American 定价:")
np.random.seed(123)
opt = American(test_payoff, random_walk)
price = opt.price(100)
print(f"价格: {price}")

print("\n检查单个时间点的 Regression:")
from blackscholes.utils.Regression import Regression as RegressionRef

simulation_result = random_walk.simulateV2(100)
t = 25
prices = simulation_result[:, :, t]
print(f"prices shape: {prices.shape}")

discounted_cashflow = np.random.rand(100) * 10

r = RegressionRef(prices, discounted_cashflow, payoff_func=test_payoff)
print(f"has_intrinsic_value: {r.has_intrinsic_value}")
print(f"index length: {len(r.index)}")
print(f"basis monomials count: {len(r.basis.monomials)}")
print(f"coefficients length: {len(r.coefficients)}")
