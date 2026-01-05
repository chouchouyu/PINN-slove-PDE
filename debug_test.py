import numpy as np
from cqf_mc_American import MC_American_Option, Paths_generater
from cqf_utils import set_seed

set_seed(555)
strike = 100
asset_num = 2
init_price_vec = 100*np.ones(asset_num)
vol_vec = 0.2*np.ones(asset_num)
ir = 0.05
dividend_vec = 0.1*np.ones(asset_num)
corr_mat = np.eye(asset_num)
corr_mat[0, 1] = 0.3
corr_mat[1, 0] = 0.3

random_walk = Paths_generater(1, 300, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)

def test_payoff(*l):
    return max(np.max(l) - strike, 0)

opt = MC_American_Option(random_walk, test_payoff)
paths = random_walk.gbm(10)
print(f"Paths shape: {paths.shape}")
print(f"First path at t=300: {paths[0, :, 300]}")
print(f"Payoff for first path at t=300: {test_payoff(*paths[0, :, 300])}")

put = opt.price(10)
print(f"Price with 10 sims: {put}")
