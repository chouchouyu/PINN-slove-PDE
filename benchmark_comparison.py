import sys, os
import time
import numpy as np

sys.path.append("/Users/susan/Downloads/hdp-master/src")
sys.path.append("/Users/susan/PINN-slove-PDE")

from blackscholes.utils.GBM import GBM as GBM_ref
from blackscholes.mc.American import American as American_ref
from blackscholes.utils.Regression import Regression as Regression_ref

from cqf_mc_American import MC_American_Option, Paths_generater
import cqf_Regression

np.random.seed(123)

def benchmark_100d():
    strike = 100
    asset_num = 100
    init_price_vec = 100 + np.random.randn(asset_num) * 5
    vol_vec = 0.2 + np.random.rand(asset_num) * 0.1
    ir = 0.05
    dividend_vec = 0.02 + np.random.rand(asset_num) * 0.03
    
    corr_mat = np.eye(asset_num)
    for i in range(asset_num):
        for j in range(i+1, asset_num):
            corr = np.random.rand() * 0.1
            corr_mat[i, j] = corr
            corr_mat[j, i] = corr
    
    def test_payoff(*l):
        return max(np.max(l) - strike, 0)
    
    print("=" * 60)
    print("1. 路径生成对比")
    print("=" * 60)
    
    np.random.seed(123)
    gbm_ref = GBM_ref(1, 50, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
    
    start = time.time()
    paths_ref = gbm_ref.simulateV2(100)
    time_gbm_generate = time.time() - start
    print(f"GBM 路径生成时间: {time_gbm_generate:.4f}秒")
    print(f"GBM 路径 shape: {paths_ref.shape}")
    
    np.random.seed(123)
    pg_cqf = Paths_generater(1, 50, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
    
    start = time.time()
    paths_cqf = pg_cqf.gbm(100)
    time_pg_generate = time.time() - start
    print(f"Paths_generater 路径生成时间: {time_pg_generate:.4f}秒")
    print(f"Paths_generater 路径 shape: {paths_cqf.shape}")
    
    print(f"\n路径生成差异: {abs(time_gbm_generate - time_pg_generate):.4f}秒")
    
    print("\n" + "=" * 60)
    print("2. 回归测试对比 (单次)")
    print("=" * 60)
    
    np.random.seed(456)
    prices = paths_ref[0, :, 25]
    cashflow = np.random.rand(100) * 10
    
    def test_payoff(*l):
        return max(np.max(l) - strike, 0)
    
    start = time.time()
    r_ref = Regression_ref(prices.reshape(1, -1), cashflow, payoff_func=test_payoff)
    time_reg_ref = time.time() - start
    print(f"Reference Regression 时间: {time_reg_ref:.6f}秒")
    
    start = time.time()
    r_cqf = cqf_Regression.Regression(prices.reshape(1, -1), cashflow, payoff_func=test_payoff)
    time_reg_cqf = time.time() - start
    print(f"cqf Regression 时间: {time_reg_cqf:.6f}秒")
    
    print(f"回归差异: {abs(time_reg_ref - time_reg_cqf):.6f}秒")
    
    print("\n" + "=" * 60)
    print("3. 折现现金流计算对比")
    print("=" * 60)
    
    cashflow_matrix = np.random.rand(100, 51) * 100
    cashflow_matrix[cashflow_matrix < 30] = 0
    
    opt_ref = American_ref(test_payoff, gbm_ref)
    start = time.time()
    for _ in range(1000):
        discounted = opt_ref._get_discounted_cashflow(25, cashflow_matrix, 100)
    time_disc_ref = time.time() - start
    print(f"Reference _get_discounted_cashflow (1000次): {time_disc_ref:.4f}秒")
    
    opt_cqf = MC_American_Option(pg_cqf, test_payoff)
    start = time.time()
    for _ in range(1000):
        discounted = opt_cqf._get_discounted_cashflow(25, cashflow_matrix, 100)
    time_disc_cqf = time.time() - start
    print(f"cqf _get_discounted_cashflow (1000次): {time_disc_cqf:.4f}秒")
    
    print(f"折现差异: {abs(time_disc_ref - time_disc_cqf):.4f}秒")
    
    print("\n" + "=" * 60)
    print("4. 完整期权定价对比 (移除print语句)")
    print("=" * 60)
    
    def test_payoff(*l):
        return max(np.max(l) - strike, 0)
    
    start = time.time()
    opt_ref = American_ref(test_payoff, gbm_ref)
    price_ref = opt_ref.price(100)
    time_price_ref = time.time() - start
    print(f"Reference American price 时间: {time_price_ref:.4f}秒")
    print(f"Reference American price: {price_ref}")
    
    start = time.time()
    opt_cqf = MC_American_Option(pg_cqf, test_payoff)
    price_cqf = opt_cqf.price(100)
    time_price_cqf = time.time() - start
    print(f"cqf MC_American price 时间: {time_price_cqf:.4f}秒")
    print(f"cqf MC_American price: {price_cqf}")
    
    print("\n" + "=" * 60)
    print("5. 总结")
    print("=" * 60)
    print(f"路径生成差异: {time_pg_generate - time_gbm_generate:+.4f}秒")
    print(f"完整定价差异: {time_price_cqf - time_price_ref:+.4f}秒")
    print(f"价格差异: {abs(price_cqf - price_ref):.10f}")

if __name__ == "__main__":
    benchmark_100d()
