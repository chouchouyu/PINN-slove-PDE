import numpy as np
from cqf_mc_American import Paths_generater, MC_American_Option

# 设置测试参数
np.random.seed(444)
astrike = 1
asset_num = 1
init_price_vec = 0.99*np.ones(asset_num)
vol_vec = 0.2*np.ones(asset_num)
ir = 0.03
dividend_vec = np.zeros(asset_num)
corr_mat = np.eye(asset_num)
T = 1
days = 300
n_simulations = 3000

# 定义支付函数
def test_payoff(l):
    return max(1 - np.sum(l), 0)

# 创建路径生成器和期权对象
paths_generater = Paths_generater(T=T, days=days, S0_vec=init_price_vec, r=ir, vol_vec=vol_vec, dividend_vec=dividend_vec, corr_mat=corr_mat)
opt = MC_American_Option(paths_generater, test_payoff)

# 计算期权价格
price = opt.price(n_simulations)
expected_price = 0.1333426194642927
print(f"计算得到的1D美式期权价格: {price}")
print(f"期望价格: {expected_price}")
print(f"差异: {abs(price - expected_price)}")
