import numpy as np
import sys
sys.path.append('/Users/susan/Downloads/hdp-master/src')

from blackscholes.mc.American import American
from blackscholes.utils.GBM import GBM
from cqf_mc_American import MC_American_Option, Paths_generater

def test_payoff_2d(*l):
    return max(np.max(l) - 100, 0)

np.random.seed(555)

asset_num = 2
init_price_vec = 100*np.ones(asset_num)
vol_vec = 0.2*np.ones(asset_num)
ir = 0.05
dividend_vec = 0.1*np.ones(asset_num)
corr_mat = np.eye(asset_num)
corr_mat[0, 1] = 0.3
corr_mat[1, 0] = 0.3

print("=" * 80)
print("Testing Reference Implementation (American.py)")
print("=" * 80)

random_walk_gbm = GBM(1, 300, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
opt_ref = American(test_payoff_2d, random_walk_gbm)
result_ref = opt_ref.price(3000)
print(f"\nReference result: {result_ref}")

print("\n" + "=" * 80)
print("Testing CQF Implementation (cqf_mc_American.py)")
print("=" * 80)

random_walk_pg = Paths_generater(1, 300, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
opt_cqf = MC_American_Option(random_walk_pg, test_payoff_2d)
result_cqf = opt_cqf.price(3000)
print(f"\nCQF result: {result_cqf}")

print("\n" + "=" * 80)
print("Comparison")
print("=" * 80)
print(f"Difference: {abs(result_ref - result_cqf)}")
print(f"Relative difference: {abs(result_ref - result_cqf) / result_ref * 100:.2f}%")
