import numpy as np
import sys
sys.path.append('/Users/susan/Downloads/hdp-master/src')
from blackscholes.utils.GBM import GBM
from blackscholes.mc.American import American

def test_american_logic():
    """
    逐行追踪 American.py 的 price 方法逻辑
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
    random_walk = GBM(1, 300, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
    
    def test_payoff(*l):
        return max(np.max(l) - strike, 0)
    
    opt = American(test_payoff, random_walk)
    
    # 手动追踪 price 方法的逻辑
    path_num = 3000
    opt.simulation_result = random_walk.simulateV2(path_num)
    cashflow_matrix = np.zeros([path_num, random_walk.N+1])
    cur_price = np.array([x[:, -1] for x in opt.simulation_result])
    cur_payoff = np.array(list(map(test_payoff, cur_price)))
    cashflow_matrix[:, random_walk.N] = cur_payoff
    
    print(f"初始化后 cashflow_matrix 形状: {cashflow_matrix.shape}")
    print(f"最后一行不为零的元素数量: {np.count_nonzero(cashflow_matrix[-1])}")
    print(f"最后一行非零值: {cashflow_matrix[-1, cashflow_matrix[-1] != 0]}")
    
    # 模拟循环
    exercise_count = 0
    for t in range(random_walk.N-1, 0, -1):
        discounted_cashflow = opt._get_discounted_cashflow(t, cashflow_matrix, path_num)
        
        from blackscholes.utils.Regression import Regression
        r = Regression(opt.simulation_result[:, :, t], discounted_cashflow, payoff_func=test_payoff)
        
        if not r.has_intrinsic_value:
            continue
            
        cur_price_t = np.array([x[:, t] for x in opt.simulation_result])
        cur_payoff_t = np.array(list(map(test_payoff, cur_price_t[r.index])))
        continuation = np.array([r.evaluate(X) for X in cur_price_t[r.index]])
        
        exercise_index = r.index[cur_payoff_t >= continuation]
        
        if len(exercise_index) > 0:
            exercise_count += len(exercise_index)
            cashflow_matrix[exercise_index] = np.zeros(cashflow_matrix[exercise_index].shape)
            all_cur_payoff = np.array(list(map(test_payoff, cur_price_t)))
            cashflow_matrix[exercise_index, t] = all_cur_payoff[exercise_index]
            
        if t % 50 == 0:
            print(f"t={t}: exercise_count={exercise_count}, non-zero elements={np.count_nonzero(cashflow_matrix)}")
    
    print(f"\n总行权次数: {exercise_count}")
    print(f"现金流矩阵中非零元素数量: {np.count_nonzero(cashflow_matrix)}")
    
    result = opt._get_discounted_cashflow_at_t0(cashflow_matrix)
    print(f"American.py 最终结果: {result}")

if __name__ == "__main__":
    test_american_logic()
