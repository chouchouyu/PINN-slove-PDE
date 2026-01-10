import numpy as np
import sys
sys.path.append("/Users/susan/Downloads/hdp-master/src")

from blackscholes.utils.GBM import GBM
from blackscholes.mc.American import American

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
print(f"2D价格 (参考): {price2d}")
