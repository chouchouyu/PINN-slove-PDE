import sys, os
import numpy as np

# 添加原始实现的路径
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/hdp-master 2/src")

# 导入原始实现
from blackscholes.utils.GBM import GBM
from blackscholes.mc.American import American

# 导入用户实现
import cqf_mc_American

# 设置相同的参数
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
n_simulations = 10  # 使用少量模拟路径便于调试

def test_payoff(*l):
    return max(np.max(l) - strike, 0)

print("=== 比较原始实现和用户实现 ===")

# 原始实现
print("\n1. 原始实现 (American 类):")
original_gbm = GBM(1, 300, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
original_opt = American(test_payoff, original_gbm)
original_price = original_opt.price(n_simulations)
print(f"原始实现价格: {original_price}")

# 用户实现
print("\n2. 用户实现 (MC_American_Option 类):")
user_gbm = cqf_mc_American.Paths_generater(1, 300, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
user_opt = cqf_mc_American.MC_American_Option(user_gbm, test_payoff)
user_price = user_opt.price(n_simulations)
print(f"用户实现价格: {user_price}")

# 详细比较中间结果
print("\n=== 详细比较中间结果 ===")

# 重新运行原始实现，保存中间结果
print("\n3. 原始实现详细步骤:")
original_gbm = GBM(1, 300, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
original_opt = American(test_payoff, original_gbm)
original_opt.simulation_result = original_gbm.simulateV2(n_simulations)
original_cashflow = np.zeros([n_simulations, original_gbm.N+1])
original_cur_price = np.array([x[:, -1] for x in original_opt.simulation_result])
original_cur_payoff = np.array(list(map(test_payoff, original_cur_price)))
original_cashflow[:, original_gbm.N] = original_cur_payoff
print(f"原始实现到期日现金流:")
print(original_cashflow[:, -1])

# 重新运行用户实现，保存中间结果
print("\n4. 用户实现详细步骤:")
user_gbm = cqf_mc_American.Paths_generater(1, 300, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
user_opt = cqf_mc_American.MC_American_Option(user_gbm, test_payoff)
user_opt.simulations_paths = user_gbm.gbm(n_simulations=n_simulations)
user_cashflow = np.zeros([n_simulations, user_gbm.days + 1])

# 比较到期日现金流
user_maturity_price = user_opt.simulations_paths[:, :, -1]
user_maturity_payoff = np.array(list(map(test_payoff, user_maturity_price)))
user_cashflow[:, -1] = user_maturity_payoff
print(f"用户实现到期日现金流:")
print(user_cashflow[:, -1])

print(f"\n到期日现金流差异:")
print(original_cashflow[:, -1] - user_cashflow[:, -1])

# 比较模拟路径
print(f"\n5. 模拟路径比较 (第1个路径的最后几个时间点):")
print(f"原始实现: {original_opt.simulation_result[0, :, -5:]}")
print(f"用户实现: {user_opt.simulations_paths[0, :, -5:]}")

# 比较_get_discounted_cashflow方法
print(f"\n6. _get_discounted_cashflow方法比较:")
t = original_gbm.N - 1
original_discounted = original_opt._get_discounted_cashflow(t, original_cashflow, n_simulations)
user_discounted = user_opt._get_discounted_cashflow(t, user_cashflow, n_simulations)
print(f"t={t}时的折现现金流:")
print(f"原始实现: {original_discounted}")
print(f"用户实现: {user_discounted}")
print(f"差异: {original_discounted - user_discounted}")

# 比较循环中的cash_flow变化
print(f"\n7. 循环中cash_flow变化比较:")

# 原始实现的循环
print("\n原始实现循环:")
for t in range(original_gbm.N-1, original_gbm.N-3, -1):  # 只显示最后几个时间点
    print(f"\nt={t}:")
    original_discounted = original_opt._get_discounted_cashflow(t, original_cashflow, n_simulations)
    print(f"  折现现金流: {original_discounted}")
    from blackscholes.utils.Regression import Regression
    r = Regression(original_opt.simulation_result[:, :, t], original_discounted, payoff_func=test_payoff)
    print(f"  是否有内在价值: {r.has_intrinsic_value}")
    if r.has_intrinsic_value:
        cur_price = np.array([x[:, t] for x in original_opt.simulation_result])
        cur_payoff = np.array(list(map(test_payoff, cur_price[r.index])))
        continuation = np.array([r.evaluate(X) for X in cur_price[r.index]])
        print(f"  内在价值: {cur_payoff}")
        print(f"  继续价值: {continuation}")
        exercise_index = r.index[cur_payoff >= continuation]
        print(f"  行使索引: {exercise_index}")
        if len(exercise_index) > 0:
            original_cashflow[exercise_index] = np.zeros(original_cashflow[exercise_index].shape)
            original_cashflow[exercise_index, t] = np.array(list(map(test_payoff, cur_price)))[exercise_index]
            print(f"  行使后的cash_flow:")
            for i in exercise_index:
                print(f"    路径{i}: {original_cashflow[i, t-2:t+3]}")

# 用户实现的循环
print("\n\n用户实现循环:")
for t in range(user_gbm.days-1, user_gbm.days-3, -1):  # 只显示最后几个时间点
    print(f"\nt={t}:")
    user_discounted = user_opt._get_discounted_cashflow(t, user_cashflow, n_simulations)
    print(f"  折现现金流: {user_discounted}")
    r = cqf_mc_American.cqf_Regression.Regression(user_opt.simulations_paths[:, :, t], user_discounted, payoff_func=test_payoff)
    print(f"  是否有内在价值: {r.has_intrinsic_value}")
    if r.has_intrinsic_value:
        cur_price = user_opt.simulations_paths[:, :, t]
        all_cur_payoff = np.array(list(map(test_payoff, cur_price)))
        continuation_value = np.array([r.evaluate(X) for X in cur_price[r.index]])
        print(f"  内在价值: {all_cur_payoff[r.index]}")
        print(f"  继续价值: {continuation_value}")
        exercise_index = r.index[all_cur_payoff[r.index] >= continuation_value]
        print(f"  行使索引: {exercise_index}")
        if len(exercise_index) > 0:
            user_cashflow[exercise_index] = np.zeros(user_cashflow[exercise_index].shape)
            user_cashflow[exercise_index, t] = all_cur_payoff[exercise_index]
            print(f"  行使后的cash_flow:")
            for i in exercise_index:
                print(f"    路径{i}: {user_cashflow[i, t-2:t+3]}")

# 比较最终价格
print(f"\n8. 最终价格比较:")
original_final_price = original_opt._get_discounted_cashflow_at_t0(original_cashflow)
user_final_price = user_opt._get_discounted_cashflow_at_t0(user_cashflow)
print(f"原始实现最终价格: {original_final_price}")
print(f"用户实现最终价格: {user_final_price}")
print(f"价格差异: {original_final_price - user_final_price}")