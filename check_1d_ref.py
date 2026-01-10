import numpy as np
import sys
sys.path.append("/Users/susan/Downloads/hdp-master/src")

np.random.seed(444)

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
print(f"1D价格 (参考): {price1d}")
print(f"期望值 (来自参考测试): 0.07187167189125372")

# 测试优化代码
from cqf_1_mc_American_optimized import MC_American_Option, Paths_generater

np.random.seed(444)

random_walk_opt = Paths_generater(T=1, days=300, s0=init_price_vec, r=ir, sigma=vol_vec, dividend=dividend_vec, corr_mat=corr_mat)

def test_payoff_opt(*l):
    return max(strike - np.sum(l), 0)

opt1_opt = MC_American_Option(random_walk_opt, test_payoff_opt)
np.random.seed(444)
price1d_opt = opt1_opt.price(3000)
print(f"\n1D价格 (优化): {price1d_opt}")
print(f"差异: {abs(price1d - price1d_opt)}")

# 检查GBM类是否生成相同的路径
print("\n检查GBM类的参数...")
gbm = GBM(T=1, N=300, init_price_vec=init_price_vec, ir=ir, vol_vec=vol_vec, dividend_vec=dividend_vec, corr_mat=corr_mat)

# 检查我的Paths_generater
pg = Paths_generater(T=1, days=300, s0=init_price_vec, r=ir, sigma=vol_vec, dividend=dividend_vec, corr_mat=corr_mat)
print(f"GBM T: {gbm.T}, GBM N: {gbm.N}")
print(f"PG T: {pg.T}, PG days: {pg.days}")
print(f"GBM dt: {gbm.dt}")
print(f"PG dt: {pg.dt}")
