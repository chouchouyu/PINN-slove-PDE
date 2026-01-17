import torch
import torch.nn as nn

class U0Network(nn.Module):
    """Neural network for approximating the initial solution value u0"""
    def __init__(self, d, hls):
        # Initialize the parent class nn.Module
        super(U0Network, self).__init__()
        # Define a sequential neural network architecture
        self.network = nn.Sequential(
            # First linear layer: input dimension d, output dimension hls
            nn.Linear(d, hls),
            # ReLU activation function
            nn.ReLU(),
            # Second linear layer: input and output dimension hls
            nn.Linear(hls, hls),
            # ReLU activation function
            nn.ReLU(),
            # Output linear layer: input dimension hls, output dimension 1
            nn.Linear(hls, 1)
        )

    def forward(self, x):
        # Forward pass: compute network output for input x
        return self.network(x)

class SigmaTGradUNetwork(nn.Module):
    """Neural network for approximating σᵀ∇u, with one independent network per time step"""
    def __init__(self, d, hls):
        # Initialize the parent class nn.Module
        super(SigmaTGradUNetwork, self).__init__()
        # Define a sequential neural network architecture
        self.network = nn.Sequential(
            # First linear layer: input dimension d, output dimension hls
            nn.Linear(d, hls),
            # ReLU activation function
            nn.ReLU(),
            # Second linear layer: input and output dimension hls
            nn.Linear(hls, hls),
            # ReLU activation function
            nn.ReLU(),
            # Third linear layer: input and output dimension hls
            nn.Linear(hls, hls),
            # ReLU activation function
            nn.ReLU(),
            # Output linear layer: input dimension hls, output dimension d
            nn.Linear(hls, d)
        )

    def forward(self, x):
        # Forward pass: compute network output for input x
        return self.network(x)