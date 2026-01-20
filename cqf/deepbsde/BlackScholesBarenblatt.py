# Import the PyTorch library for tensor operations and neural networks
import torch


# Define a class for the Black-Scholes-Barenblatt equation
class BlackScholesBarenblatt:
    """Base class for Black-Scholes-Barenblatt equation"""

    # Constructor method to initialize the class instance
    def __init__(self, d=30):
        # Set the dimension of the problem (number of assets)
        self.d = d
        # Set the risk-free interest rate (5%)
        self.r = 0.05
        # Set the volatility parameter (40%)
        self.sigma = 0.4

    # Define the terminal condition function g(X)
    def g_tf(self, X):
        """Terminal condition: g(X) = sum(X^2)"""
        # Calculate the sum of squares of X along dimension 1, keeping the dimension
        return torch.sum(X**2, dim=1, keepdim=True)

    # Define the nonlinear term phi in the PDE
    def phi_tf(self, X, u, sigma_grad_u, t):
        """Nonlinear term: phi(X, u, σᵀ∇u, p, t) = r * (u - sum(X * σᵀ∇u))"""
        # Calculate the nonlinear term: r * (u - sum of element-wise product of X and sigma_grad_u)
        return self.r * (u - torch.sum(X * sigma_grad_u, dim=1, keepdim=True))

    # Define the drift term mu (zero in this case)
    def mu_tf(self, X, t):
        """Drift term: μ(X, p, t) = 0"""
        # Return a tensor of zeros with the same shape as X
        return torch.zeros_like(X)

    # Define the diffusion term sigma
    def sigma_tf(self, X, t):
        """Diffusion term: σ(X, p, t) = Diagonal(sigma * X)"""
        # Check if X is 1-dimensional (single sample)
        if len(X.shape) == 1:
            # For 1D case: return a 2D diagonal matrix with sigma * X on the diagonal
            return torch.diag(self.sigma * X)
        else:
            # For 2D case (batch): return a 3D batch of diagonal matrices
            # Get the batch size from the first dimension of X
            batch_size = X.shape[0]
            # Create diagonal matrices for each sample in the batch
            return torch.diag_embed(self.sigma * X)

    # Define the analytical solution to the Black-Scholes-Barenblatt equation
    def analytical_solution(self, x, t, T=1.0):
        """Analytical solution"""
        # Calculate the exponent: (r + sigma^2) * (T - t)
        exponent = (self.r + self.sigma**2) * (T - t)
        # Return the analytical solution: exp(exponent) * sum(x^2)
        return torch.exp(torch.tensor(exponent, device=x.device)) * torch.sum(x**2)
