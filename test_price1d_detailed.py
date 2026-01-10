import numpy as np
from cqf_mc_American import Paths_generater, MC_American_Option

# 完全按照原始实现的setUp方法设置参数
strike = 1
asset_num = 1
init_price_vec = 0.99*np.ones(asset_num)
vol_vec = 0.2*np.ones(asset_num)
ir = 0.03
dividend_vec = np.zeros(asset_num)
corr_mat = np.eye(asset_num)
T = 1
days = 300
n_simulations = 3000

# 定义与原始实现完全相同的支付函数
def test_payoff(*l):
    return max(strike - np.sum(l), 0)

print("=== 测试参数 ===")
print(f"strike: {strike}")
print(f"asset_num: {asset_num}")
print(f"init_price_vec: {init_price_vec}")
print(f"vol_vec: {vol_vec}")
print(f"ir: {ir}")
print(f"dividend_vec: {dividend_vec}")
print(f"corr_mat: {corr_mat}")
print(f"T: {T}")
print(f"days: {days}")
print(f"n_simulations: {n_simulations}")

# 创建路径生成器和期权对象
print("\n=== 创建对象 ===")
paths_generater = Paths_generater(T=T, days=days, S0_vec=init_price_vec, r=ir, vol_vec=vol_vec, dividend_vec=dividend_vec, corr_mat=corr_mat)
opt = MC_American_Option(paths_generater, test_payoff)

# 计算期权价格
print("\n=== 计算期权价格 ===")
np.random.seed(444)
price = opt.price(n_simulations)
expected_price = 0.1333426194642927
print(f"计算得到的1D美式期权价格: {price}")
print(f"期望价格: {expected_price}")
print(f"差异: {abs(price - expected_price)}")
