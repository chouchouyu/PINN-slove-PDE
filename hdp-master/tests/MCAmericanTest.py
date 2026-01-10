import sys, os

import pandas
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.utils.GBM import GBM
from blackscholes.mc.American import American
import unittest
import numpy as np

# class Test(unittest.TestCase):
class Test():
    def setUp(self):
        strike = 1
        asset_num = 1
        init_price_vec = 0.99*np.ones(asset_num)
        vol_vec = 0.2*np.ones(asset_num)
        ir = 0.03
        dividend_vec = np.zeros(asset_num)
        corr_mat = np.eye(asset_num)
        random_walk = GBM(1, 300, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
        
        def test_payoff(*l):
            return max(strike - np.sum(l), 0)
        self.opt1 = American(test_payoff, random_walk)
        
    def test_price1d(self):
        np.random.seed(444)
        price = self.opt1.price(3000) 
        print(f"计算得到的美式期权价格: {price:.8f}") #计算得到的美式期权价格: 0.07187167189125372
        # print("random_walk:", pandas.DataFrame(self.opt1.simulation_result))
        # print("random_walk:", self.opt1.simulation_result)
        assert abs(price - 0.07187167189125372) < 1e-10

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
        random_walk = GBM(1, 300, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
        def test_payoff(*l):
            return max(np.max(l) - strike, 0)
        opt = American(test_payoff, random_walk)
        put = opt.price(3000)
        real_put = 9.6333
        print(f"计算得到的美式期权价格: {put}") #计算得到的美式期权价格: 9.55793682
        print(f"实际美式期权价格: {real_put}")
        assert abs(put - 9.557936820537265) < 0.00000000000001
        assert abs(put - real_put)/real_put < 0.00783
        # when init = 110, price is 18.021487449289822/18.15771299285956, real is 17.3487
        # when init = 100, price is 10.072509537503821/9.992812015410516, real is 9.6333

    def test_get_discounted_cashflow(self):
        random_walk = GBM(3, 3, np.ones(1), 0.03, np.ones(1), np.zeros(1), np.eye(1))
        def test_payoff(*l):
            return max(3 - np.sum(l), 0)
        opt = American(test_payoff, random_walk)
        cashflow_matrix = np.array([[0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 2]])
        discounted = opt._get_discounted_cashflow(2, cashflow_matrix, 3)
        print(discounted)  #[2.9113366 0.        1.94089107]
        assert sum(abs(discounted - np.array([2.9113366, 0, 1.94089107]))) < 0.00000001
        print("-----", sum(abs(discounted - np.array([2.9113366, 0, 1.94089107])))) #3.548508153983221e-09

        cashflow_matrix2 = np.array([[0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 2]])
        discounted2 = opt._get_discounted_cashflow(0, cashflow_matrix2, 3)
        print(discounted2)  #[2.8252936 0.        1.82786237]
        assert sum(abs(discounted2 - np.array([2.8252936, 0, 1.82786237]))) < 0.00000001
        print("-----", sum(abs(discounted2 - np.array([2.8252936, 0, 1.82786237])))) #1.2952021677392622e-09
        
    def test_get_discounted_cashflow_at_t0(self):
        random_walk = GBM(3, 3, np.ones(1), 0.03, np.ones(1), np.zeros(1), np.eye(1))
        def test_payoff(*l):
            return max(3 - np.sum(l), 0)
        opt = American(test_payoff, random_walk)
        discount = opt._get_discounted_cashflow_at_t0(np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 2, 0, 0]]))
        print(discount) #0.9608852002270883
        assert discount == (0+np.exp(-2*0.03)+2*np.exp(-1*0.03))/3

if __name__ == '__main__':
    # unittest.main()
    test=Test()
    # test.setUp()
    # test.test_price1d()  
    # test.test_price2d()  
    # test.test_get_discounted_cashflow()
    test.test_get_discounted_cashflow_at_t0()