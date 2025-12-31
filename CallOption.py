import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from FBSNNs import FBSNN


class CallOption(FBSNN):
    def __init__(self, Xi, T, M, N, D, Mm, layers, mode, activation):
        # Constructor for the BlackScholesBarenblatt class
        # Initializes a new instance with specified parameters for the neural network
        # Inherits from FBSNN (Forward-Backward Stochastic Neural Network)
        # Parameters:
        # Xi: Initial condition
        # T: Time horizon
        # M: Batch size
        # N: Number of time discretization steps
        # D: Dimension of the problem
        # layers: Configuration of the neural network layers
        # mode: Operation mode
        # activation: Activation function for the neural network
        super().__init__(Xi, T, M, N, D, Mm, layers, mode, activation)

    def phi_tf(self, t, X, Y, Z):
        # 漂移项（BSDE的条件期望）
        # φ = r·Y  
# 表示折现率的影响
 
        # Defines the drift term in the Black-Scholes-Barenblatt equation for a batch
        # t: Batch of current times, size M x 1
        # X: Batch of current states, size M x D
        # Y: Batch of current value functions, size M x 1
        # Z: Batch of gradients of the value function with respect to X, size M x D
        # Returns the drift term for each instance in the batch, size M x 1
        rate = 0.01  # Risk-free interest rate
        return rate * (Y) # M x 1

    def g_tf(self, X):  
        # g(X) = max(∑X - K, 0) # 当前为1D：max(X - K, 0)

        # Terminal condition for the Black-Scholes-Barenblatt equation for a batch
        # X: Batch of terminal states, size M x D
        # Returns the terminal condition for each instance in the batch, size M x 1
        temp = torch.sum(X, dim=1, keepdim=True)
        return torch.maximum(temp - self.strike, torch.tensor(0.0))

    def mu_tf(self, t, X, Y, Z): 
        # μ = r·X
# 符合几何布朗运动模型

        # Drift coefficient of the underlying stochastic process for a batch
        # Inherits from the superclass FBSNN without modification
        # Parameters are the same as in phi_tf, with batch sizes
        rate = 0.01
        return rate * X # M x D

    def sigma_tf(self, t, X, Y):  
        # σ = 0.25 · diag(X)
# 对角矩阵，每个资产独立波动

        # Diffusion coefficient of the underlying stochastic process for a batch
        # t: Batch of current times, size M x 1
        # X: Batch of current states, size M x D
        # Y: Batch of current value functions, size M x 1 (not used in this method)
        # Returns a batch of diagonal matrices, each of size D x D, for the diffusion coefficients
        # Each matrix is scaled by 0.4 times the corresponding state in X
        sigma = 0.25 # Volatility
        return sigma * torch.diag_embed(X)  # M x D x D