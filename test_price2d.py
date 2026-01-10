import numpy as np
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

# 创建路径生成器和期权对象
paths_generater = Paths_generater(T=T, days=days, S0_vec=init_price_vec, r=ir, vol_vec=vol_vec, dividend_vec=dividend_vec, corr_mat=corr_mat)
opt = MC_American_Option(paths_generater, test_payoff)

# 计算期权价格
put = opt.price(n_simulations)
print(f"计算得到的美式期权价格: {put}")
print(f"期望价格: 9.600595700658364")
print(f"价格差异: {abs(put - 9.600595700658364)}")
