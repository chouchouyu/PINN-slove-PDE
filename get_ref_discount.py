import numpy as np
import sys
sys.path.append("/Users/susan/Downloads/hdp-master/src")

from blackscholes.mc.American import American as AmericanRef

class MockRandomWalkRef:
    def __init__(self):
        self.T = 3
        self.days = 3
        self.N = 3
        self.dt = 1
        self.s0 = np.ones(1)
        self.ir = 0.03
        self.sigma = np.ones(1)
        self.dividend = np.zeros(1)
        self.corr_mat = np.eye(1)
    
    def simulateV2(self, n_simulations=1000):
        return np.zeros((n_simulations, 1, 4))

def test_payoff(*l):
    return max(3 - np.sum(l), 0)

opt_ref = AmericanRef(test_payoff, MockRandomWalkRef())

cashflow_matrix = np.array([[0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 2]])
discounted = opt_ref._get_discounted_cashflow(2, cashflow_matrix, 3)

print(f"折现现金流 (参考): {discounted}")

cashflow_matrix2 = np.array([[0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 2]])
discounted2 = opt_ref._get_discounted_cashflow(0, cashflow_matrix2, 3)

print(f"折现现金流2 (参考): {discounted2}")
