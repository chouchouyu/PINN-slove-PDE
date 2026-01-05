import numpy as np
import sys
sys.path.append('/Users/susan/Downloads/hdp-master/src')

from blackscholes.utils.GBM import GBM
from cqf_mc_American import Paths_generater

np.random.seed(555)

asset_num = 2
init_price_vec = 100*np.ones(asset_num)
vol_vec = 0.2*np.ones(asset_num)
ir = 0.05
dividend_vec = 0.1*np.ones(asset_num)
corr_mat = np.eye(asset_num)
corr_mat[0, 1] = 0.3
corr_mat[1, 0] = 0.3

print("Testing path generators with seed 555...")

random_walk_gbm = GBM(1, 300, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
paths_gbm = random_walk_gbm.simulateV2(10)
print(f"GBM paths shape: {paths_gbm.shape}")

random_walk_pg = Paths_generater(1, 300, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
paths_pg = random_walk_pg.gbm(10)
print(f"Paths_generater paths shape: {paths_pg.shape}")

print(f"\nPaths are equal: {np.allclose(paths_gbm, paths_pg)}")
print(f"Max difference: {np.max(np.abs(paths_gbm - paths_pg))}")
print(f"\nFirst few values at t=298:")
print(f"GBM: {paths_gbm[:, :, 298][:5]}")
print(f"PG:  {paths_pg[:, :, 298][:5]}")
