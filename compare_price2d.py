import sys, os
import numpy as np

# 添加原始项目路径
sys.path.append('/Users/susan/PINN-slove-PDE/hdp-master 2/src')
from blackscholes.utils.GBM import GBM
from blackscholes.mc.American import American

# 导入用户实现
from cqf_mc_American import Paths_generater, MC_American_Option

# 设置测试参数
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
T = 1
days = 300
n_simulations = 3000

# 定义支付函数
def test_payoff(*l):
    return max(np.max(l) - strike, 0)

print("=== 原始实现测试 ===")
# 原始实现
random_walk_original = GBM(T, days, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
opt_original = American(test_payoff, random_walk_original)
put_original = opt_original.price(n_simulations)
print(f"原始实现计算得到的美式期权价格: {put_original}")

print("\n=== 用户实现测试 ===")
# 用户实现
paths_generater_user = Paths_generater(T=T, days=days, S0_vec=init_price_vec, r=ir, vol_vec=vol_vec, dividend_vec=dividend_vec, corr_mat=corr_mat)
opt_user = MC_American_Option(paths_generater_user, test_payoff)
put_user = opt_user.price(n_simulations)
print(f"用户实现计算得到的美式期权价格: {put_user}")

print(f"\n价格差异: {abs(put_original - put_user)}")

# 比较路径生成
print("\n=== 路径生成比较 ===")
# 重新设置种子以确保路径相同
np.random.seed(555)
paths_original = random_walk_original.simulateV2(n_simulations)

np.random.seed(555)
paths_user = paths_generater_user.gbm(n_simulations)

print(f"原始路径形状: {paths_original.shape}")
print(f"用户路径形状: {paths_user.shape}")

# 比较前几个路径的到期价格
print("\n前5个路径的到期价格比较:")
print("原始实现:", [test_payoff(paths_original[i, :, days]) for i in range(5)])
print("用户实现:", [test_payoff(paths_user[i, :, days]) for i in range(5)])

# 比较路径生成器的内部参数
print("\n=== 路径生成器参数比较 ===")
print(f"原始dt: {random_walk_original.T / random_walk_original.N}")
print(f"用户dt: {paths_generater_user.dt}")
print(f"原始drift: {random_walk_original.r - dividend_vec}")
print(f"用户drift: {paths_generater_user.r - paths_generater_user.dividend_vec}")
print(f"原始vol: {random_walk_original.vol}")
print(f"用户vol: {paths_generater_user.vol_vec}")
print(f"原始corr_mat: {random_walk_original.corrMat}")
print(f"用户corr_mat: {paths_generater_user.corr_mat}")
