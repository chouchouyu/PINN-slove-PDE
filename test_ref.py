import sys, os
sys.path.append('/Users/susan/Downloads/hdp-master/src')
from blackscholes.mc.American import American
from blackscholes.utils.GBM import GBM
import numpy as np

np.random.seed(555)
strike = 100
asset_num = 2
init_price_vec = 100*np.ones(asset_num)
vol_vec = 0.2*np.ones(asset_num)
ir = 0.05
dividend_vec = 0.1*np.ones(asset_num)
corr_mat = np.eye(asset_num)
corr_mat[0, 1] = 0.3
corr_mat[1, 0] = 0.3
random_walk = GBM(1, 300, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
def test_payoff(*l):
    return max(np.max(l) - strike, 0)
opt = American(test_payoff, random_walk)
put = opt.price(3000)
print(f'Reference implementation result: {put}')
