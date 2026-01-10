import numpy as np
import sys
sys.path.append("/Users/susan/Downloads/hdp-master/src")

np.random.seed(555)

print("="*60)
print("1D 测试")
print("="*60)

from blackscholes.utils.GBM import GBM
from blackscholes.mc.American import American

strike = 1
asset_num = 1
init_price_vec = 0.99 * np.ones(asset_num)
vol_vec = 0.2 * np.ones(asset_num)
ir = 0.03
dividend_vec = np.zeros(asset_num)
corr_mat = np.eye(asset_num)
random_walk = GBM(1, 300, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)

def test_payoff(*l):
    return max(strike - np.sum(l), 0)

opt1 = American(test_payoff, random_walk)
np.random.seed(444)
price1d = opt1.price(3000)
print(f"1D价格: {price1d}")

print("\n" + "="*60)
print("2D 测试")
print("="*60)

np.random.seed(555)
strike = 100
asset_num = 2
init_price_vec = 100 * np.ones(asset_num)
vol_vec = 0.2 * np.ones(asset_num)
ir = 0.05
dividend_vec = 0.1 * np.ones(asset_num)
corr_mat = np.eye(asset_num)
corr_mat[0, 1] = 0.3
corr_mat[1, 0] = 0.3

random_walk2 = GBM(1, 300, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)

def test_payoff2(*l):
    return max(np.max(l) - strike, 0)

opt2 = American(test_payoff2, random_walk2)
np.random.seed(555)
price2d = opt2.price(3000)
print(f"2D价格: {price2d}")

print("\n" + "="*60)
print("100D 测试")
print("="*60)

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

random_walk100 = GBM(1, 50, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)

def test_payoff100(*l):
    return max(np.max(l) - strike, 0)

opt100 = American(test_payoff100, random_walk100)
np.random.seed(123)
price100d = opt100.price(100)
print(f"100D价格: {price100d}")
