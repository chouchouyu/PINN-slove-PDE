"""
100-Dimensional Black-Scholes Option Pricing Model
Enhanced implementation with configurable parameters and basket option support

Author: Generated for PINN-solve-PDE project
Date: 2025-12-19
"""

import numpy as np
from scipy.stats import norm
from typing import Union, Dict, List, Tuple, Optional
import warnings


class CallOption100D:
    """
    100-Dimensional Black-Scholes Model for European Call Options
    
    Supports:
    - Single asset pricing
    - Basket options (weighted sum of assets)
    - Configurable parameters per dimension
    - Multiple pricing methods (analytical, Monte Carlo, quasi-Monte Carlo)
    - Greeks computation
    """
    
    def __init__(
        self,
        dimensions: int = 100,
        spot_prices: Union[float, np.ndarray] = None,
        strike: float = 100.0,
        time_to_maturity: float = 1.0,
        risk_free_rate: float = 0.05,
        volatilities: Union[float, np.ndarray] = None,
        dividends: Union[float, np.ndarray] = None,
        correlation_matrix: np.ndarray = None,
        weights: np.ndarray = None,
        basket_type: str = 'arithmetic'
    ):
        """
        Initialize 100-Dimensional Black-Scholes Model
        
        Parameters
        ----------
        dimensions : int
            Number of dimensions (default: 100)
        spot_prices : float or ndarray
            Initial spot prices for each dimension
        strike : float
            Strike price for the option
        time_to_maturity : float
            Time to maturity in years
        risk_free_rate : float
            Risk-free interest rate
        volatilities : float or ndarray
            Volatility for each dimension
        dividends : float or ndarray
            Dividend yield for each dimension
        correlation_matrix : ndarray
            Correlation matrix between dimensions (100x100)
        weights : ndarray
            Weights for basket option (must sum to 1)
        basket_type : str
            Type of basket: 'arithmetic' or 'geometric'
        """
        self.dimensions = dimensions
        self.strike = strike
        self.T = time_to_maturity
        self.r = risk_free_rate
        self.basket_type = basket_type
        
        # Initialize spot prices
        if spot_prices is None:
            self.S0 = np.ones(dimensions) * 100.0
        elif isinstance(spot_prices, (int, float)):
            self.S0 = np.ones(dimensions) * spot_prices
        else:
            self.S0 = np.asarray(spot_prices)
            if len(self.S0) != dimensions:
                raise ValueError(f"Spot prices dimension mismatch: expected {dimensions}, got {len(self.S0)}")
        
        # Initialize volatilities
        if volatilities is None:
            self.sigma = np.ones(dimensions) * 0.2
        elif isinstance(volatilities, (int, float)):
            self.sigma = np.ones(dimensions) * volatilities
        else:
            self.sigma = np.asarray(volatilities)
            if len(self.sigma) != dimensions:
                raise ValueError(f"Volatility dimension mismatch: expected {dimensions}, got {len(self.sigma)}")
        
        # Initialize dividend yields
        if dividends is None:
            self.q = np.zeros(dimensions)
        elif isinstance(dividends, (int, float)):
            self.q = np.ones(dimensions) * dividends
        else:
            self.q = np.asarray(dividends)
            if len(self.q) != dimensions:
                raise ValueError(f"Dividend dimension mismatch: expected {dimensions}, got {len(self.q)}")
        
        # Initialize correlation matrix
        if correlation_matrix is None:
            self.rho = np.eye(dimensions)
        else:
            self.rho = correlation_matrix
            if self.rho.shape != (dimensions, dimensions):
                raise ValueError(f"Correlation matrix shape mismatch: expected ({dimensions}, {dimensions}), got {self.rho.shape}")
        
        # Initialize weights for basket option
        if weights is None:
            self.weights = np.ones(dimensions) / dimensions
        else:
            self.weights = np.asarray(weights)
            if len(self.weights) != dimensions:
                raise ValueError(f"Weights dimension mismatch: expected {dimensions}, got {len(self.weights)}")
            if not np.isclose(np.sum(self.weights), 1.0):
                warnings.warn(f"Weights sum to {np.sum(self.weights)}, normalizing to sum to 1.0")
                self.weights = self.weights / np.sum(self.weights)
        
        # Validate correlation matrix
        self._validate_correlation_matrix()
    
    def _validate_correlation_matrix(self) -> None:
        """Validate correlation matrix is positive semi-definite."""
        eigenvalues = np.linalg.eigvalsh(self.rho)
        if np.any(eigenvalues < -1e-10):
            warnings.warn("Correlation matrix has negative eigenvalues. Adjusting to make it positive semi-definite.")
            # Adjust to ensure positive semi-definiteness
            D, V = np.linalg.eigh(self.rho)
            D[D < 0] = 0
            self.rho = V @ np.diag(D) @ V.T
    
    def _get_cholesky_decomposition(self) -> np.ndarray:
        """
        Get Cholesky decomposition of correlation matrix.
        
        Returns
        -------
        ndarray
            Lower triangular matrix L such that L @ L.T ≈ correlation matrix
        """
        try:
            L = np.linalg.cholesky(self.rho)
        except np.linalg.LinAlgError:
            # If Cholesky fails, use eigenvalue decomposition
            D, V = np.linalg.eigh(self.rho)
            D[D < 0] = 1e-10
            L = V @ np.diag(np.sqrt(D))
        return L
    
    def black_scholes_single(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Analytical Black-Scholes formula for single asset.
        
        Parameters
        ----------
        S : float
            Current spot price
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        sigma : float
            Volatility
        q : float
            Dividend yield
        
        Returns
        -------
        float
            Call option price
        """
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return max(call_price, 0)
    
    def basket_price_analytical_approximation(self) -> float:
        """
        Analytical approximation for basket option using Curran's method.
        
        Returns
        -------
        float
            Approximated basket option price
        """
        # Compute basket parameters
        basket_spot = np.sum(self.weights * self.S0)
        
        # Compute effective volatility
        V = np.zeros((self.dimensions, self.dimensions))
        for i in range(self.dimensions):
            for j in range(self.dimensions):
                V[i, j] = self.weights[i] * self.weights[j] * self.sigma[i] * self.sigma[j] * self.rho[i, j]
        
        basket_variance = np.sum(V)
        basket_sigma = np.sqrt(basket_variance) / basket_spot if basket_spot > 0 else 0.5
        
        # Compute effective dividend yield
        basket_q = np.sum(self.weights * self.q)
        
        # Apply Black-Scholes to basket
        return self.black_scholes_single(basket_spot, self.strike, self.T, self.r, basket_sigma, basket_q)
    
    def monte_carlo_pricing(self, num_simulations: int = 10000, seed: Optional[int] = None) -> Tuple[float, float]:
        """
        Price option using Monte Carlo simulation.
        
        Parameters
        ----------
        num_simulations : int
            Number of Monte Carlo paths
        seed : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        tuple
            (option_price, standard_error)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Get Cholesky decomposition
        L = self._get_cholesky_decomposition()
        
        # Generate correlated random numbers
        Z = np.random.standard_normal((num_simulations, self.dimensions))
        X = Z @ L.T
        
        # Simulate asset prices at maturity
        dt = self.T
        ST = np.zeros((num_simulations, self.dimensions))
        for i in range(self.dimensions):
            drift = (self.r - self.q[i] - 0.5 * self.sigma[i] ** 2) * dt
            diffusion = self.sigma[i] * np.sqrt(dt) * X[:, i]
            ST[:, i] = self.S0[i] * np.exp(drift + diffusion)
        
        # Compute payoff
        if self.basket_type == 'arithmetic':
            basket_values = np.sum(self.weights[:, np.newaxis] * ST, axis=0)
        elif self.basket_type == 'geometric':
            log_basket = np.sum(self.weights[:, np.newaxis] * np.log(ST), axis=0)
            basket_values = np.exp(log_basket)
        else:
            raise ValueError(f"Unknown basket type: {self.basket_type}")
        
        payoffs = np.maximum(basket_values - self.strike, 0)
        discounted_payoffs = payoffs * np.exp(-self.r * self.T)
        
        option_price = np.mean(discounted_payoffs)
        standard_error = np.std(discounted_payoffs) / np.sqrt(num_simulations)
        
        return option_price, standard_error
    
    def quasi_monte_carlo_pricing(self, num_simulations: int = 10000) -> Tuple[float, float]:
        """
        Price option using Quasi-Monte Carlo with Sobol sequences.
        
        Parameters
        ----------
        num_simulations : int
            Number of quasi-random samples
        
        Returns
        -------
        tuple
            (option_price, standard_error)
        """
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.dimensions, scramble=True)
            samples = sampler.random(num_simulations)
        except ImportError:
            warnings.warn("scipy.stats.qmc not available, falling back to regular Monte Carlo")
            return self.monte_carlo_pricing(num_simulations)
        
        # Convert uniform samples to normal
        Z = norm.ppf(samples)
        
        # Get Cholesky decomposition
        L = self._get_cholesky_decomposition()
        X = Z @ L.T
        
        # Simulate asset prices at maturity
        dt = self.T
        ST = np.zeros((num_simulations, self.dimensions))
        for i in range(self.dimensions):
            drift = (self.r - self.q[i] - 0.5 * self.sigma[i] ** 2) * dt
            diffusion = self.sigma[i] * np.sqrt(dt) * X[:, i]
            ST[:, i] = self.S0[i] * np.exp(drift + diffusion)
        
        # Compute payoff
        if self.basket_type == 'arithmetic':
            basket_values = np.sum(self.weights[:, np.newaxis] * ST, axis=0)
        elif self.basket_type == 'geometric':
            log_basket = np.sum(self.weights[:, np.newaxis] * np.log(ST), axis=0)
            basket_values = np.exp(log_basket)
        else:
            raise ValueError(f"Unknown basket type: {self.basket_type}")
        
        payoffs = np.maximum(basket_values - self.strike, 0)
        discounted_payoffs = payoffs * np.exp(-self.r * self.T)
        
        option_price = np.mean(discounted_payoffs)
        standard_error = np.std(discounted_payoffs) / np.sqrt(num_simulations)
        
        return option_price, standard_error
    
    def compute_greeks_monte_carlo(self, num_simulations: int = 10000, bump: float = 0.01) -> Dict[str, float]:
        """
        Compute Greeks using finite differences with Monte Carlo.
        
        Parameters
        ----------
        num_simulations : int
            Number of Monte Carlo simulations
        bump : float
            Bump size for finite differences
        
        Returns
        -------
        dict
            Dictionary containing Greeks: delta, gamma, vega, theta, rho
        """
        # Base price
        base_price, _ = self.monte_carlo_pricing(num_simulations)
        
        greeks = {}
        
        # Delta: average delta across all dimensions
        deltas = []
        for i in range(self.dimensions):
            self.S0[i] += bump
            up_price, _ = self.monte_carlo_pricing(num_simulations)
            self.S0[i] -= 2 * bump
            down_price, _ = self.monte_carlo_pricing(num_simulations)
            self.S0[i] += bump
            
            delta = (up_price - down_price) / (2 * bump)
            deltas.append(delta)
        
        greeks['delta_vector'] = np.array(deltas)
        greeks['delta_weighted'] = np.sum(self.weights * deltas)
        greeks['delta_portfolio'] = np.sum(deltas)
        
        # Gamma: average gamma
        gammas = []
        for i in range(self.dimensions):
            self.S0[i] += bump
            up_price, _ = self.monte_carlo_pricing(num_simulations)
            self.S0[i] -= 2 * bump
            down_price, _ = self.monte_carlo_pricing(num_simulations)
            self.S0[i] += bump
            mid_price = base_price
            
            gamma = (up_price - 2 * mid_price + down_price) / (bump ** 2)
            gammas.append(gamma)
        
        greeks['gamma_vector'] = np.array(gammas)
        greeks['gamma_weighted'] = np.sum(self.weights * gammas)
        
        # Vega: sensitivity to volatility
        vegas = []
        vega_bump = 0.01  # 1% bump in volatility
        for i in range(self.dimensions):
            self.sigma[i] += vega_bump
            up_price, _ = self.monte_carlo_pricing(num_simulations)
            self.sigma[i] -= vega_bump
            
            vega = (up_price - base_price) / vega_bump
            vegas.append(vega)
        
        greeks['vega_vector'] = np.array(vegas)
        greeks['vega_weighted'] = np.sum(self.weights * vegas)
        
        # Theta: time decay
        self.T += bump
        theta_up, _ = self.monte_carlo_pricing(num_simulations)
        self.T -= bump
        greeks['theta'] = (theta_up - base_price) / bump
        
        # Rho: interest rate sensitivity
        self.r += bump
        rho_up, _ = self.monte_carlo_pricing(num_simulations)
        self.r -= bump
        greeks['rho'] = (rho_up - base_price) / bump
        
        return greeks
    
    def price(self, method: str = 'analytical') -> Union[float, Tuple[float, float]]:
        """
        Price the option using specified method.
        
        Parameters
        ----------
        method : str
            Pricing method: 'analytical', 'monte_carlo', or 'quasi_monte_carlo'
        
        Returns
        -------
        float or tuple
            Option price (and standard error for Monte Carlo methods)
        """
        if method == 'analytical':
            return self.basket_price_analytical_approximation()
        elif method == 'monte_carlo':
            return self.monte_carlo_pricing()
        elif method == 'quasi_monte_carlo':
            return self.quasi_monte_carlo_pricing()
        else:
            raise ValueError(f"Unknown pricing method: {method}")
    
    def summary(self) -> Dict:
        """
        Return summary of model parameters and settings.
        
        Returns
        -------
        dict
            Summary of model configuration
        """
        return {
            'dimensions': self.dimensions,
            'spot_prices': {
                'mean': np.mean(self.S0),
                'min': np.min(self.S0),
                'max': np.max(self.S0)
            },
            'strike': self.strike,
            'time_to_maturity': self.T,
            'risk_free_rate': self.r,
            'volatilities': {
                'mean': np.mean(self.sigma),
                'min': np.min(self.sigma),
                'max': np.max(self.sigma)
            },
            'dividends': {
                'mean': np.mean(self.q),
                'min': np.min(self.q),
                'max': np.max(self.q)
            },
            'basket_type': self.basket_type,
            'weights': {
                'sum': np.sum(self.weights),
                'mean': np.mean(self.weights),
                'max': np.max(self.weights)
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Basic 100D option with uniform parameters
    print("=" * 80)
    print("Example 1: Basic 100-Dimensional Black-Scholes Model")
    print("=" * 80)
    
    option1 = CallOption100D(
        dimensions=100,
        spot_prices=100.0,
        strike=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatilities=0.2
    )
    
    print("Model Summary:")
    summary = option1.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Price using analytical approximation
    price_analytical = option1.price(method='analytical')
    print(f"\nAnalytical Approximation Price: ${price_analytical:.4f}")
    
    # Price using Monte Carlo
    price_mc, se_mc = option1.monte_carlo_pricing(num_simulations=5000)
    print(f"Monte Carlo Price: ${price_mc:.4f} ± ${se_mc:.4f}")
    
    # Example 2: Heterogeneous volatilities and spot prices
    print("\n" + "=" * 80)
    print("Example 2: Heterogeneous Parameters with Basket Option")
    print("=" * 80)
    
    np.random.seed(42)
    spot_prices = np.random.uniform(80, 120, 100)
    volatilities = np.random.uniform(0.1, 0.4, 100)
    
    option2 = CallOption100D(
        dimensions=100,
        spot_prices=spot_prices,
        strike=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatilities=volatilities,
        basket_type='arithmetic'
    )
    
    price_basket = option2.price(method='analytical')
    print(f"Basket Option (Arithmetic) Price: ${price_basket:.4f}")
    
    # Example 3: Geometric basket
    option3 = CallOption100D(
        dimensions=100,
        spot_prices=100.0,
        strike=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatilities=0.2,
        basket_type='geometric'
    )
    
    price_geometric_mc, se_geometric = option3.monte_carlo_pricing(num_simulations=5000)
    print(f"Basket Option (Geometric) Price: ${price_geometric_mc:.4f} ± ${se_geometric:.4f}")
    
    print("\n" + "=" * 80)
    print("Examples completed successfully!")
    print("=" * 80)
