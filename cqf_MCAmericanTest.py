import sys, os
import time

from cqf_mc_American import MC_American_Option, Paths_generater
  
import unittest
import numpy as np

from cqf_utils import set_seed

# class Test(unittest.TestCase):
class Test():
 
    def test_get_discounted_cashflow_at_t0(self):
        set_seed(444)
        random_walk = Paths_generater(3, 3, np.ones(1), 0.03, np.ones(1), np.zeros(1), np.eye(1))
        def test_payoff(*l):
            return max(3 - np.sum(l), 0)
        opt = MC_American_Option( random_walk,test_payoff)
        discount = opt._get_discounted_cashflow_at_t0(np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 2, 0, 0]]))
        print(discount) #0.9608852002270883
        assert discount == (np.exp(-2*0.03)+2*np.exp(-1*0.03))/2
 
    def test_get_discounted_cashflow(self):
        random_walk = Paths_generater(3, 3, np.ones(1), 0.03, np.ones(1), np.zeros(1), np.eye(1))
        def test_payoff(*l):
            return max(3 - np.sum(l), 0)
        opt = MC_American_Option( random_walk,test_payoff)

        cashflow_matrix = np.array([[0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 2]])
        discounted = opt._get_discounted_cashflow(2, cashflow_matrix,3)
        print(discounted)  #[2.9113366 0.        1.94089107]
        assert sum(abs(discounted - np.array([2.9113366, 0, 1.94089107]))) < 0.00000001
        print("-----", sum(abs(discounted - np.array([2.9113366, 0, 1.94089107])))) #3.548508153983221e-09

        cashflow_matrix2 = np.array([[0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 2]])
        discounted2 = opt._get_discounted_cashflow(0, cashflow_matrix2,3)
        print(discounted2)  #[2.8252936 0.        1.82786237]
        assert sum(abs(discounted2 - np.array([2.8252936, 0, 1.82786237]))) < 0.00000001
        print("-----", sum(abs(discounted2 - np.array([2.8252936, 0, 1.82786237])))) #1.2952021677392622e-09

    def test_price1d(self):
        np.random.seed(444)
        strike = 1
        asset_num = 1
        init_price_vec = 0.99*np.ones(asset_num)
        vol_vec = 0.2*np.ones(asset_num)
        ir = 0.03
        dividend_vec = np.zeros(asset_num)
        corr_mat = np.eye(asset_num)
        random_walk = Paths_generater(1, 300, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
     
        def test_payoff(*l):
            return max(strike - np.sum(l), 0)
        opt1 =  MC_American_Option(random_walk,test_payoff)
        price = opt1.price(3000) 
        print("1d price:", price) #0.1333426194642927
        # assert abs(price - 0.1343499608830493) < 1e-10

    def test_price2d(self):
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
        opt = MC_American_Option(random_walk,test_payoff)
        put = opt.price(3000)
        real_put = 9.6333
        print(f"计算得到的美式期权价格: {put}") #计算得到的美式期权价格: 14.33637984169992
        print(f"实际美式期权价格: {real_put}")
        assert abs(put - 9.557936820537265) < 0.00000000000001
        assert abs(put - real_put)/real_put < 0.00783
        # when init = 110, price is 18.021487449289822/18.15771299285956, real is 17.3487
        # when init = 100, price is 10.072509537503821/9.992812015410516, real is 9.6333


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
            for j in range(i+1, asset_num):
                corr = np.random.rand() * 0.1
                corr_mat[i, j] = corr
                corr_mat[j, i] = corr
        
        random_walk = Paths_generater(1, 50, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
        
        def test_payoff(*l):
            return max(np.max(l) - strike, 0)
        
        opt = MC_American_Option(random_walk, test_payoff)
        
        start_time = time.time()
        price = opt.price(100)
        end_time = time.time()
        
        print(f"100维美式期权价格: {price}")
        print(f"运行时间: {end_time - start_time:.4f} 秒")
        # - 100维美式期权价格 : 95.46420911718947
        # - 运行时间 : 168.2666 秒（约2.8分钟）



if __name__ == '__main__':
    # unittest.main()
    test=Test()
    # test.test_get_discounted_cashflow_at_t0()
    # test.test_get_discounted_cashflow()
    # test.test_price1d()  
    # test.test_price2d()
    test.test_100d_pricing()