import numpy as np
import sys
sys.path.append('/Users/susan/PINN-slove-PDE')
from cqf_mc_American import MC_American_Option, Paths_generater

def test_cqf_american_logic():
    """
    追踪 cqf_mc_American.py 的 price 方法逻辑
    """
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
    random_walk = Paths_generater(1, 300, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
    
    def test_payoff(*l):
        return max(np.max(l) - strike, 0)
    
    opt = MC_American_Option(random_walk, test_payoff)
    
    # 手动追踪 price 方法的逻辑
    n_simulations = 3000
    opt.simulations_paths = random_walk.gbm(n_simulations=n_simulations)
    cash_flow = np.zeros([n_simulations, random_walk.days+1])
    
    print(f"初始化后 cash_flow 形状: {cash_flow.shape}")
    
    exercise_count = 0
    early_return_count = 0
    
    for t in range(random_walk.days, 0, -1):
        prices_at_t = opt.simulations_paths[:, :, t]
        
        if t == random_walk.days:
            maturity_price = prices_at_t
            maturity_payoff = np.array(list(map(test_payoff, maturity_price)))
            cash_flow[:, -1] = maturity_payoff
        else:
            discounted_cashflow = cash_flow[:, t + 1] * np.exp(-random_walk.r * random_walk.dt)
            
            import cqf_Regression
            r = cqf_Regression.Regression(prices_at_t, discounted_cashflow, payoff_func=test_payoff)
            
            if r.has_intrinsic_value:
                all_cur_payoff = np.array(list(map(test_payoff, prices_at_t)))
                continuation_value = np.array([r.evaluate(X) for X in prices_at_t[r.index]])
                
                exercise_index = r.index[all_cur_payoff[r.index] >= continuation_value]
                
                if len(exercise_index) > 0:
                    exercise_count += len(exercise_index)
                    cash_flow[exercise_index] = 0
                    cash_flow[exercise_index, t] = all_cur_payoff[exercise_index]
                    early_return_count += 1
                    print(f"t={t}: 发生提前返回! exercise_count={exercise_count}")
                    break
            else:
                continue
    
    print(f"\n总行权次数: {exercise_count}")
    print(f"提前返回次数: {early_return_count}")
    print(f"现金流矩阵中非零元素数量: {np.count_nonzero(cash_flow)}")
    
    if early_return_count > 0:
        print("\n⚠️ 发现提前返回! 这是一个 bug!")
        print("cash_flow 应该继续循环到 t=1，但代码在 t=300 时就返回了")
    
    result = opt._get_discounted_cashflow_at_t0(cash_flow)
    print(f"cqf_mc_American.py 最终结果: {result}")

if __name__ == "__main__":
    test_cqf_american_logic()
