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

class MockRandomWalkOpt:
    def __init__(self):
        self.T = 3
        self.days = 3
        self.N = 3
        self.dt = 1
        self.s0 = np.ones(1)
        self.r = 0.03
        self.sigma = np.ones(1)
        self.dividend = np.zeros(1)
        self.corr_mat = np.eye(1)
    
    def gbm(self, n_simulations=1000):
        return np.zeros((n_simulations, 1, 4))

random_walk_ref = MockRandomWalkRef()
random_walk_opt = MockRandomWalkOpt()

def test_payoff(*l):
    return max(3 - np.sum(l), 0)

opt_ref = AmericanRef(test_payoff, random_walk_ref)

from cqf_1_mc_American_optimized import Paths_generater, MC_American_Option

opt_opt = MC_American_Option(random_walk_opt, test_payoff)

cashflow_matrix = np.array([[0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 2]])

discounted_ref = opt_ref._get_discounted_cashflow(2, cashflow_matrix, 3)
discounted_opt = opt_opt._get_discounted_cashflow_optimized(2, cashflow_matrix, 3)

print(f"折现现金流 (参考): {discounted_ref}")
print(f"折现现金流 (优化): {discounted_opt}")
print(f"差异: {discounted_ref - discounted_opt}")
