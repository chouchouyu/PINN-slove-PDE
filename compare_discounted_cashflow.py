import sys, os
import numpy as np

# 添加原始项目路径
sys.path.append('/Users/susan/PINN-slove-PDE/hdp-master 2/src')
from blackscholes.utils.GBM import GBM
from blackscholes.mc.American import American

# 导入用户实现
from cqf_mc_American import Paths_generater, MC_American_Option

# 设置测试参数
T = 3
days = 3
init_price_vec = np.ones(1)
vol_vec = np.ones(1)
ir = 0.03
dividend_vec = np.zeros(1)
corr_mat = np.eye(1)

# 定义支付函数
def test_payoff(*l):
    return max(3 - np.sum(l), 0)

# 原始实现
random_walk_original = GBM(T, days, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
opt_original = American(test_payoff, random_walk_original)

# 用户实现
paths_generater_user = Paths_generater(T=T, days=days, S0_vec=init_price_vec, r=ir, vol_vec=vol_vec, dividend_vec=dividend_vec, corr_mat=corr_mat)
opt_user = MC_American_Option(paths_generater_user, test_payoff)

# 测试_get_discounted_cashflow方法
print("=== _get_discounted_cashflow方法比较 ===")
cashflow_matrix = np.array([[0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 2]])
t = 2
n_paths = 3

# 原始实现结果
discounted_original = opt_original._get_discounted_cashflow(t, cashflow_matrix, n_paths)
print(f"原始实现结果: {discounted_original}")

# 用户实现结果
discounted_user = opt_user._get_discounted_cashflow(t, cashflow_matrix, n_paths)
print(f"用户实现结果: {discounted_user}")

# 计算差异
print(f"差异: {sum(abs(discounted_original - discounted_user))}")

# 测试_get_discounted_cashflow_at_t0方法
print("\n=== _get_discounted_cashflow_at_t0方法比较 ===")
cashflow_matrix2 = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 2, 0, 0]])

# 原始实现结果
discounted_t0_original = opt_original._get_discounted_cashflow_at_t0(cashflow_matrix2)
print(f"原始实现结果: {discounted_t0_original}")

# 用户实现结果
discounted_t0_user = opt_user._get_discounted_cashflow_at_t0(cashflow_matrix2)
print(f"用户实现结果: {discounted_t0_user}")

# 计算差异
print(f"差异: {abs(discounted_t0_original - discounted_t0_user)}")

# 测试更复杂的现金流矩阵
print("\n=== 更复杂的现金流矩阵测试 ===")
cashflow_matrix3 = np.array([
    [0, 0, 0, 5],
    [0, 3, 0, 0],
    [0, 0, 4, 0],
    [0, 2, 0, 0],
    [0, 0, 0, 0]
])

# 原始实现结果
discounted_original3 = opt_original._get_discounted_cashflow(1, cashflow_matrix3, 5)
print(f"原始实现结果 (t=1): {discounted_original3}")

# 用户实现结果
discounted_user3 = opt_user._get_discounted_cashflow(1, cashflow_matrix3, 5)
print(f"用户实现结果 (t=1): {discounted_user3}")

# 计算差异
print(f"差异: {sum(abs(discounted_original3 - discounted_user3))}")
