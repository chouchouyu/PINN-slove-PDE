import numpy as np


# Class to generate asset price paths using Geometric Brownian Motion (GBM) with correlation
class Paths_generater:
    # Constructor to initialize the path generator with simulation parameters
    def __init__(self, T, days, S0_vec, r: float, vol_vec, dividend_vec, corr_mat):
        # Verify that all input vectors and matrices have consistent dimensions
        assert (
            len(S0_vec)
            == len(vol_vec)
            == len(dividend_vec)
            == corr_mat.shape[0]
            == corr_mat.shape[1]
        ), "Vectors' lengths are different"
        # Time step size
        self.dt = T / days
        # Total simulation time
        self.T = T
        # Number of time steps
        self.days = days
        # Initial asset prices vector
        self.S0_vec = S0_vec
        # Risk-free interest rate
        self.r = r
        # Volatility vector for each asset
        self.vol_vec = vol_vec
        # Dividend yield vector for each asset
        self.dividend_vec = dividend_vec
        # Correlation matrix between assets
        self.corr_mat = corr_mat
        # Number of assets
        self.asset_num = len(S0_vec)

    # Method to generate GBM paths for multiple assets
    def gbm(self, n_simulations: int) -> np.ndarray:
        # Initialize list to store price paths for all simulations
        price_paths = []  # np.zeros((n_simulations, self.asset_num, self.days + 1))
        # Perform Cholesky decomposition to get lower triangular matrix for correlated random variables
        L = np.linalg.cholesky(self.corr_mat)
        # Calculate drift vector: risk-free rate minus dividend yield
        drift_vec = self.r - self.dividend_vec
        # Generate n_simulations paths
        for _ in range(n_simulations):
            # Create array to store price path for current simulation: assets x time steps
            price_path = np.zeros((self.asset_num, self.days + 1))
            # Set initial prices at time 0
            price_path[:, 0] = self.S0_vec
            # Generate path for each time step
            for t in range(1, self.days + 1):
                # Generate correlated random increments: L * Z * sqrt(dt)
                dW = L.dot(np.random.normal(size=self.asset_num)) * np.sqrt(self.dt)
                # Multiply volatility with random increments
                rand_term = np.multiply(self.vol_vec, dW)
                # Update prices using GBM formula: S_t = S_{t-1} * exp((drift - vol^2/2)*dt + rand_term)
                price_path[:, t] = np.multiply(
                    price_path[:, t - 1],
                    np.exp((drift_vec - self.vol_vec**2 / 2) * self.dt + rand_term),
                )
            # Append the complete path for this simulation
            price_paths.append(price_path)

        # Convert list of paths to numpy array
        return np.array(price_paths)
