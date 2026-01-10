import sys, os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cqf_1_mc_American_optimized import MC_American_Option, Paths_generater

import numpy as np


class TestOptimized:
    
    def test_get_discounted_cashflow_at_t0(self):
        np.random.seed(444)
        random_walk = Paths_generater(dt=1, days=3, s0=np.ones(1), r=0.03, sigma=np.ones(1), dividend=np.zeros(1), corr_mat=np.eye(1))
        
        def test_payoff(*l):
            return max(3 - np.sum(l), 0)
        
        opt = MC_American_Option(random_walk, test_payoff)
        
        cashflow_matrix = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 2, 0, 0]])
        discount = opt._get_discounted_cashflow_at_t0(cashflow_matrix)
        
        expected = 1.4413278003406325
        
        print(f"折现值: {discount}")
        print(f"期望值: {expected}")
        assert abs(discount - expected) < 1e-10, f"折现值计算错误: {discount} != {expected}"
        print("✓ test_get_discounted_cashflow_at_t0 测试通过")
    
    def test_get_discounted_cashflow(self):
        random_walk = Paths_generater(dt=1, days=3, s0=np.ones(1), r=0.03, sigma=np.ones(1), dividend=np.zeros(1), corr_mat=np.eye(1))
        
        def test_payoff(*l):
            return max(3 - np.sum(l), 0)
        
        opt = MC_American_Option(random_walk, test_payoff)
        
        cashflow_matrix = np.array([[0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 2]])
        discounted = opt._get_discounted_cashflow_optimized(2, cashflow_matrix, 3)
        
        expected = np.array([2.9113366, 0, 1.94089107])
        
        print(f"折现现金流: {discounted}")
        print(f"期望值: {expected}")
        assert sum(abs(discounted - expected)) < 0.00000001, f"折现现金流计算错误"
        print("✓ 测试1通过")
        
        cashflow_matrix2 = np.array([[0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 2]])
        discounted2 = opt._get_discounted_cashflow_optimized(0, cashflow_matrix2, 3)
        
        expected2 = np.array([2.8252936, 0, 1.82786237])
        
        print(f"折现现金流2: {discounted2}")
        print(f"期望值2: {expected2}")
        assert sum(abs(discounted2 - expected2)) < 0.00000001, f"折现现金流计算错误"
        print("✓ 测试2通过")
        print("✓ test_get_discounted_cashflow 测试通过")
    
    def test_price1d(self):
        np.random.seed(444)
        strike = 1
        asset_num = 1
        init_price_vec = 0.99 * np.ones(asset_num)
        vol_vec = 0.2 * np.ones(asset_num)
        ir = 0.03
        dividend_vec = np.zeros(asset_num)
        corr_mat = np.eye(asset_num)
        
        random_walk = Paths_generater(dt=1/300, days=300, s0=init_price_vec, r=ir, sigma=vol_vec, dividend=dividend_vec, corr_mat=corr_mat)
        
        def test_payoff(*l):
            return max(strike - np.sum(l), 0)
        
        start_time = time.time()
        opt = MC_American_Option(random_walk, test_payoff)
        price = opt.price(3000)
        elapsed = time.time() - start_time
        
        ref_price = 0.1333426194642927
        
        print(f"1D期权价格: {price}")
        print(f"参考价格: {ref_price}")
        print(f"相对误差: {abs(price - ref_price) / ref_price * 100:.4f}%")
        print(f"运行时间: {elapsed:.4f} 秒")
        
        assert abs(price - ref_price) / ref_price < 0.05, f"1D价格与参考实现差异过大: {price} vs {ref_price}"
        print("✓ test_price1d 测试通过")
    
    def test_price2d(self):
        np.random.seed(555)
        strike = 100
        asset_num = 2
        init_price_vec = 100 * np.ones(asset_num)
        vol_vec = 0.2 * np.ones(asset_num)
        ir = 0.05
        dividend_vec = 0.1 * np.ones(asset_num)
        corr_mat = np.eye(asset_num)
        corr_mat[0, 1] = 0.3
        corr_mat[1, 0] = 0.3
        
        random_walk = Paths_generater(dt=1/300, days=300, s0=init_price_vec, r=ir, sigma=vol_vec, dividend=dividend_vec, corr_mat=corr_mat)
        
        def test_payoff(*l):
            return max(np.max(l) - strike, 0)
        
        start_time = time.time()
        opt = MC_American_Option(random_walk, test_payoff)
        put = opt.price(3000)
        elapsed = time.time() - start_time
        
        ref_price = 14.33637984169992
        
        print(f"计算得到的美式期权价格: {put}")
        print(f"参考价格: {ref_price}")
        print(f"相对误差: {abs(put - ref_price) / ref_price * 100:.4f}%")
        print(f"运行时间: {elapsed:.4f} 秒")
        
        assert abs(put - ref_price) / ref_price < 0.05, f"2D价格与参考实现差异过大: {put} vs {ref_price}"
        print("✓ test_price2d 测试通过")
    
    def test_100d_pricing(self):
        np.random.seed(123)
        strike = 100
        asset_num = 100
        init_price_vec = 100 + np.random.randn(asset_num) * 5
        vol_vec = 0.2 + np.random.rand(asset_num) * 0.1
        ir = 0.05
        dividend_vec = 0.02 + np.random.rand(asset_num) * 0.03
        
        corr_mat = np.eye(asset_num)
        for i in range(asset_num):
            for j in range(i + 1, asset_num):
                corr = np.random.rand() * 0.1
                corr_mat[i, j] = corr
                corr_mat[j, i] = corr
        
        random_walk = Paths_generater(dt=1/50, days=50, s0=init_price_vec, r=ir, sigma=vol_vec, dividend=dividend_vec, corr_mat=corr_mat)
        
        def test_payoff(*l):
            return max(np.max(l) - strike, 0)
        
        opt = MC_American_Option(random_walk, test_payoff)
        
        start_time = time.time()
        price = opt.price(100)
        end_time = time.time()
        
        elapsed = end_time - start_time
        
        ref_price = 94.52734654476743
        
        print(f"100维美式期权价格: {price}")
        print(f"参考价格: {ref_price}")
        print(f"相对误差: {abs(price - ref_price) / ref_price * 100:.4f}%")
        print(f"运行时间: {elapsed:.4f} 秒")
        print(f"预期运行时间: < 60秒")
        
        assert abs(price - ref_price) / ref_price < 0.05, f"100D价格与参考实现差异过大: {price} vs {ref_price}"
        assert elapsed < 60, f"100D运行时间超出预期: {elapsed:.2f}秒"
        print("✓ test_100d_pricing 测试通过")


class TestComparison:
    
    def compare_all_tests(self, n_simulations=500):
        print("\n" + "=" * 70)
        print("优化代码 vs 原始代码对比测试")
        print(f"模拟次数: {n_simulations}")
        print("=" * 70)
        
        results = {}
        
        print("\n1. 1D期权定价对比:")
        print("-" * 50)
        np.random.seed(444)
        
        strike = 1
        asset_num = 1
        init_price_vec = 0.99 * np.ones(asset_num)
        vol_vec = 0.2 * np.ones(asset_num)
        ir = 0.03
        dividend_vec = np.zeros(asset_num)
        corr_mat = np.eye(asset_num)
        
        random_walk_opt = Paths_generater(dt=1/300, days=300, s0=init_price_vec, r=ir, sigma=vol_vec, dividend=dividend_vec, corr_mat=corr_mat)
        random_walk_orig = Paths_generater_orig(T=1, days=300, S0_vec=init_price_vec, r=ir, vol_vec=vol_vec, dividend_vec=dividend_vec, corr_mat=corr_mat)
        
        def test_payoff_1d(*l):
            return max(strike - np.sum(l), 0)
        
        start = time.time()
        opt = MC_American_Option(random_walk_opt, test_payoff_1d)
        price_opt = opt.price(n_simulations)
        time_opt = time.time() - start
        
        print(f"优化代码价格: {price_opt:.6f}")
        print(f"优化代码时间: {time_opt:.4f}秒")
        
        from cqf_mc_American import MC_American_Option as MC_American_Option_orig
        from cqf_mc_American import Paths_generater as Paths_generater_orig
        
        random_walk_orig = Paths_generater_orig(1, 300, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
        
        start = time.time()
        opt_orig = MC_American_Option_orig(random_walk_orig, test_payoff_1d)
        price_orig = opt_orig.price(n_simulations)
        time_orig = time.time() - start
        
        print(f"原始代码价格: {price_orig:.6f}")
        print(f"原始代码时间: {time_orig:.4f}秒")
        
        price_diff = abs(price_opt - price_orig)
        print(f"价格差异: {price_diff:.6f}")
        print(f"速度提升: {(time_orig - time_opt) / time_orig * 100:.2f}%")
        
        results['1d'] = {'opt_price': price_opt, 'opt_time': time_opt,
                         'orig_price': price_orig, 'orig_time': time_orig}
        
        print("\n2. 2D期权定价对比:")
        print("-" * 50)
        np.random.seed(555)
        
        strike = 100
        asset_num = 2
        init_price_vec = 100 * np.ones(asset_num)
        vol_vec = 0.2 * np.ones(asset_num)
        ir = 0.05
        dividend_vec = 0.1 * np.ones(asset_num)
        corr_mat = np.eye(asset_num)
        corr_mat[0, 1] = 0.3
        corr_mat[1, 0] = 0.3
        
        random_walk_opt = Paths_generater(dt=1/300, days=300, s0=init_price_vec, r=ir, sigma=vol_vec, dividend=dividend_vec, corr_mat=corr_mat)
        random_walk_orig = Paths_generater_orig(T=1, days=300, S0_vec=init_price_vec, r=ir, vol_vec=vol_vec, dividend_vec=dividend_vec, corr_mat=corr_mat)
        
        def test_payoff_2d(*l):
            return max(np.max(l) - strike, 0)
        
        start = time.time()
        opt = MC_American_Option(random_walk_opt, test_payoff_2d)
        price_opt = opt.price(n_simulations)
        time_opt = time.time() - start
        
        print(f"优化代码价格: {price_opt:.6f}")
        print(f"优化代码时间: {time_opt:.4f}秒")
        
        start = time.time()
        opt_orig = MC_American_Option_orig(random_walk_orig, test_payoff_2d)
        price_orig = opt_orig.price(n_simulations)
        time_orig = time.time() - start
        
        print(f"原始代码价格: {price_orig:.6f}")
        print(f"原始代码时间: {time_orig:.4f}秒")
        
        price_diff = abs(price_opt - price_orig)
        print(f"价格差异: {price_diff:.6f}")
        print(f"速度提升: {(time_orig - time_opt) / time_orig * 100:.2f}%")
        
        results['2d'] = {'opt_price': price_opt, 'opt_time': time_opt,
                         'orig_price': price_orig, 'orig_time': time_orig}
        
        print("\n3. 50维期权定价对比:")
        print("-" * 50)
        np.random.seed(123)
        
        strike = 100
        asset_num = 50
        init_price_vec = 100 + np.random.randn(asset_num) * 5
        vol_vec = 0.2 + np.random.rand(asset_num) * 0.1
        ir = 0.05
        dividend_vec = 0.02 + np.random.rand(asset_num) * 0.03
        
        corr_mat = np.eye(asset_num)
        for i in range(asset_num):
            for j in range(i + 1, asset_num):
                corr = np.random.rand() * 0.1
                corr_mat[i, j] = corr
                corr_mat[j, i] = corr
        
        random_walk_opt = Paths_generater(dt=1/50, days=50, s0=init_price_vec, r=ir, sigma=vol_vec, dividend=dividend_vec, corr_mat=corr_mat)
        random_walk_orig = Paths_generater_orig(T=1, days=50, S0_vec=init_price_vec, r=ir, vol_vec=vol_vec, dividend_vec=dividend_vec, corr_mat=corr_mat)
        
        def test_payoff_md(*l):
            return max(np.max(l) - strike, 0)
        
        start = time.time()
        opt = MC_American_Option(random_walk_opt, test_payoff_md)
        price_opt = opt.price(n_simulations)
        time_opt = time.time() - start
        
        print(f"优化代码价格: {price_opt:.6f}")
        print(f"优化代码时间: {time_opt:.4f}秒")
        
        start = time.time()
        opt_orig = MC_American_Option_orig(random_walk_orig, test_payoff_md)
        price_orig = opt_orig.price(n_simulations)
        time_orig = time.time() - start
        
        print(f"原始代码价格: {price_orig:.6f}")
        print(f"原始代码时间: {time_orig:.4f}秒")
        
        price_diff = abs(price_opt - price_orig)
        print(f"价格差异: {price_diff:.6f}")
        print(f"速度提升: {(time_orig - time_opt) / time_orig * 100:.2f}%")
        
        results['50d'] = {'opt_price': price_opt, 'opt_time': time_opt,
                          'orig_price': price_orig, 'orig_time': time_orig}
        
        print("\n" + "=" * 70)
        print("总结对比")
        print("=" * 70)
        print(f"{'维度':<10} {'优化价格':<15} {'优化时间':<12} {'原始价格':<15} {'原始时间':<12} {'速度提升':<10}")
        print("-" * 70)
        
        total_speedup = 0
        count = 0
        
        for dim in ['1d', '2d', '50d']:
            r = results[dim]
            speedup = (r['orig_time'] - r['opt_time']) / r['orig_time'] * 100 if r['orig_time'] > 0 else 0
            total_speedup += speedup
            count += 1
            print(f"{dim:<10} {r['opt_price']:<15.6f} {r['opt_time']:<12.4f} {r['orig_price']:<15.6f} {r['orig_time']:<12.4f} {speedup:<10.2f}%")
        
        avg_speedup = total_speedup / count
        print("-" * 70)
        print(f"{'平均':<10} {'':<15} {'':<12} {'':<15} {'':<12} {avg_speedup:<10.2f}%")
        print("=" * 70)
        
        return results


if __name__ == '__main__':
    # print("=" * 70)
    # print("优化代码验证测试")
    # print("=" * 70)
    
    test = TestOptimized()
    
    # print("\n运行基本测试...")
    # print("-" * 50)
    
    # try:
    test.test_get_discounted_cashflow_at_t0()
    # except Exception as e:
    #     print(f"✗ test_get_discounted_cashflow_at_t0 失败: {e}")
    
    # try:
    #     test.test_get_discounted_cashflow()
    # except Exception as e:
    #     print(f"✗ test_get_discounted_cashflow 失败: {e}")
    
    # try:
    #     test.test_price1d()
    # except Exception as e:
    #     print(f"✗ test_price1d 失败: {e}")
    
    # try:
    #     test.test_price2d()
    # except Exception as e:
    #     print(f"✗ test_price2d 失败: {e}")
    
    # try:
    #     test.test_100d_pricing()
    # except Exception as e:
    #     print(f"✗ test_100d_pricing 失败: {e}")
    
    # print("\n" + "=" * 70)
    # print("与原始代码对比测试")
    # print("=" * 70)
    
    # comparison = TestComparison()
    # try:
    #     comparison.compare_all_tests(n_simulations=500)
    # except Exception as e:
    #     print(f"✗ 对比测试失败: {e}")
    #     import traceback
    #     traceback.print_exc()
