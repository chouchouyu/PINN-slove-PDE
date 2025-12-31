import time
import matplotlib.pyplot as plt
import torch
import numpy as np
from BlackScholesBarenblatt import *
from FBSNNs import *
import sys
import os
sys.path.append(os.path.abspath("Algorithms/"))
sys.path.append(os.path.abspath("models/"))

M = 100  # number of trajectories (batch size)
N = 500 # number of time snapshots
D = 10  # number of dimensions
Mm = N

layers = [D + 1] + 4 * [256] + [1]

Xi = np.array([1.0, 0.5] * int(D / 2))[None, :]
T = 1.0

"Available architectures"
mode = "FC"  # FC and Naisnet are available
activation = "Sine"  # Sine, ReLU and Tanh are available
model = BlackScholesBarenblatt(Xi, T, M, N, D, Mm, layers, mode, activation)

n_iter = 250
lr = 1e-3
tot = time.time()
print(model.device)
graph = model.train(n_iter, lr)
print("total time:", time.time() - tot, "s")

