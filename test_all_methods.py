import numpy as np
from cqf_mc_American import Paths_generater, MC_American_Option

print("=== 测试 _get_discounted_cashflow_at_t0 方法 ===")
# 测试 _get_discounted_cashflow_at_t0 方法
random_walk = Paths_generater(T=3, days=3, S0_vec=np.ones(1), r=0.03, vol_vec=np.ones(1), dividend_vec=np.zeros(1), corr_mat=np.eye(1))
def test_payoff(*l):
    return max(3 - np.sum(l), 0)
opt = MC_American_Option(random_walk, test_payoff)
cashflow_matrix = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 2, 0, 0]])
discount = opt._get_discounted_cashflow_at_t0(cashflow_matrix)
expected_discount = (0 + np.exp(-2*0.03) + 2*np.exp(-1*0.03))/3
print(f"计算得到的平均折现价值: {discount}")
print(f"期望的平均折现价值: {expected_discount}")
print(f"差异: {abs(discount - expected_discount)}")
assert abs(discount - expected_discount) < 0.00000001, "_get_discounted_cashflow_at_t0 方法测试失败！"
print("✓ _get_discounted_cashflow_at_t0 方法测试通过！")

print("\n=== 测试 _get_discounted_cashflow 方法 ===")
# 测试 _get_discounted_cashflow 方法
cashflow_matrix = np.array([[0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 2]])
t = 2
n_paths = 3
discounted = opt._get_discounted_cashflow(t, cashflow_matrix, n_paths)
expected_discounted = np.array([2.9113366, 0, 1.94089107])
print(f"计算得到的折现现金流: {discounted}")
print(f"期望的折现现金流: {expected_discounted}")
print(f"差异: {sum(abs(discounted - expected_discounted))}")
assert sum(abs(discounted - expected_discounted)) < 0.00000001, "_get_discounted_cashflow 方法测试失败！"

cashflow_matrix2 = np.array([[0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 2]])
t = 0
discounted2 = opt._get_discounted_cashflow(t, cashflow_matrix2, n_paths)
expected_discounted2 = np.array([2.8252936, 0, 1.82786237])
print(f"\n第二组测试计算得到的折现现金流: {discounted2}")
print(f"第二组测试期望的折现现金流: {expected_discounted2}")
print(f"差异: {sum(abs(discounted2 - expected_discounted2))}")
assert sum(abs(discounted2 - expected_discounted2)) < 0.00000001, "_get_discounted_cashflow 方法测试失败！"
print("✓ _get_discounted_cashflow 方法测试通过！")

print("\n=== 测试 test_price1d 方法 ===")
# 测试 test_price1d 方法
np.random.seed(444)
asset_num = 1
S0_vec = 0.99*np.ones(asset_num)
T = 1
days = 300
r = 0.03
dividend_vec = np.zeros(asset_num)
sigma_vec = 0.2*np.ones(asset_num)
corr_mat = np.eye(asset_num)

# 看跌期权支付函数
def payoff(*l):
    return max(1 - np.sum(l), 0)

paths_generater = Paths_generater(T=T, days=days, S0_vec=S0_vec, r=r, vol_vec=sigma_vec, dividend_vec=dividend_vec, corr_mat=corr_mat)
opt = MC_American_Option(paths_generater, payoff)
price = opt.price(3000)
expected_price = 0.1333426194642927
print(f"计算得到的1D美式期权价格: {price}")
print(f"期望价格: {expected_price}")
print(f"差异: {abs(price - expected_price)}")
assert abs(price - expected_price) < 1e-10, "test_price1d 方法测试失败！"
print("✓ test_price1d 方法测试通过！")

print("\n=== 测试 test_price2d 方法 ===")
# 测试 test_price2d 方法
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

paths_generater = Paths_generater(T=T, days=days, S0_vec=init_price_vec, r=ir, vol_vec=vol_vec, dividend_vec=dividend_vec, corr_mat=corr_mat)
opt = MC_American_Option(paths_generater, test_payoff)
put = opt.price(n_simulations)
expected_put = 9.600595700658364
print(f"计算得到的2D美式期权价格: {put}")
print(f"期望价格: {expected_put}")
print(f"差异: {abs(put - expected_put)}")
assert abs(put - expected_put) < 1e-10, "test_price2d 方法测试失败！"
print("✓ test_price2d 方法测试通过！")

print("\n=== 所有测试通过！ ===")
