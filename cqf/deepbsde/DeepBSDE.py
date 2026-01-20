import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from .Models import U0Network, SigmaTGradUNetwork
from .BlackScholesBarenblatt import BlackScholesBarenblatt


# Define function to calculate relative L2 error between predicted and analytical values
def rel_error_l2(u, uanal):
    """Calculate relative L2 error between predicted value u and analytical solution uanal"""
    # Check if analytical value is sufficiently large to avoid division by very small numbers
    if abs(uanal) >= 10 * np.finfo(float).eps:
        # Calculate relative error when analytical value is significant
        return np.sqrt((u - uanal) ** 2 / uanal**2)
    else:
        # Return absolute error when analytical value is very small
        return abs(u - uanal)


# Define the main solver class that inherits from BlackScholesBarenblatt
class BlackScholesBarenblattSolver(BlackScholesBarenblatt):
    """Solver for Black-Scholes-Barenblatt equation using deep learning methods"""

    # Class constructor with default parameters
    def __init__(
        self,
        d=30,
        x0=None,
        tspan=(0.0, 1.0),
        dt=0.25,
        m=30,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        # Call parent class constructor with dimension parameter
        super().__init__(d)

        # Set computation device (GPU if available, else CPU)
        self.device = device

        # Set initial conditions - create alternating pattern of 1.0 and 0.5 if not provided
        if x0 is None:
            # Create tensor with alternating 1.0 and 0.5 values
            self.x0 = torch.tensor(
                [1.0 if i % 2 == 0 else 0.5 for i in range(d)],
                dtype=torch.float32,
                device=device,
            )
        else:
            # Use provided initial conditions and remove extra dimensions
            self.x0 = torch.tensor(x0, dtype=torch.float32, device=device).squeeze(0)

        # Set time span for the simulation
        self.tspan = tspan
        # Set time step size
        self.dt = dt
        # Calculate number of time steps based on time span and step size
        self.time_steps = int((self.tspan[1] - self.tspan[0]) / self.dt)
        # Set number of training trajectories
        self.m = m

        # Parameters for Legendre transform (used for bounds calculation)
        # Create array of control variable values
        self.A = torch.linspace(-2.0, 2.0, 401, device=device)
        # Create array of u values for domain sampling
        self.u_domain = torch.linspace(-500.0, 500.0, 10001, device=device)

        # Neural network initialization
        # Set hidden layer size based on problem dimension
        self.hls = 10 + d
        # Initialize network for initial value u0
        self.u0 = U0Network(d, self.hls).to(device)
        # Initialize list of networks for sigma transpose gradient u at each time step
        self.sigma_grad_u = nn.ModuleList(
            [SigmaTGradUNetwork(d, self.hls).to(device) for _ in range(self.time_steps)]
        )

        # Optimizer setup - combine parameters from all networks
        self.optimizer = optim.Adam(
            # Include parameters from u0 network
            list(self.u0.parameters()) +
            # Include parameters from all sigma_grad_u networks
            [param for net in self.sigma_grad_u for param in net.parameters()],
            # Set learning rate
            lr=0.001,
        )

        # Training history tracking
        # List to store loss values during training
        self.losses = []
        # List to store u0 values during training
        self.u0_history = []

    # Method to generate trajectories for training
    def generate_trajectories(self, batch_size=None):
        """Generate sample trajectories for training the neural network"""
        # Use default batch size if not specified
        if batch_size is None:
            batch_size = self.m

        # Replicate initial condition for all samples in batch
        X = self.x0.repeat(batch_size, 1)
        # Get initial u value from neural network
        u = self.u0(X)

        # Create time points array
        ts = torch.arange(
            self.tspan[0], self.tspan[1] + self.dt / 2, self.dt, device=self.device
        )

        # Iterate through time steps
        for i in range(len(ts) - 1):
            # Get current time as Python float
            t = ts[i].item()

            # Get sigma transpose gradient u value from appropriate network
            sigma_grad_u_val = self.sigma_grad_u[i](X)
            # Generate random Brownian motion increments
            dW = torch.randn(batch_size, self.d, device=self.device) * np.sqrt(self.dt)

            # Update u value using the PDE drift term
            f_val = self.phi_tf(X, u, sigma_grad_u_val, t)
            u = (
                u
                - f_val * self.dt
                + torch.sum(sigma_grad_u_val * dW, dim=1, keepdim=True)
            )

            # Update state variable X
            mu_val = self.mu_tf(X, t)
            sigma_val = self.sigma_tf(X, t)

            # Handle different tensor shapes for matrix multiplication
            if len(sigma_val.shape) == 2:  # Single sample case (2D matrix)
                X = X + mu_val * self.dt + torch.matmul(dW, sigma_val)
            else:  # Batch case (3D tensor)
                # Expand dimensions for matrix multiplication
                dW_expanded = dW.unsqueeze(-1)  # [batch_size, d, 1]
                sigma_val_expanded = sigma_val
                # Calculate state update using matrix multiplication
                X_update = torch.matmul(sigma_val_expanded, dW_expanded).squeeze(
                    -1
                )  # [batch_size, d]
                X = X + mu_val * self.dt + X_update

        # Return final state and value
        return X, u

    # Method to calculate loss function
    def loss_function(self):
        """Calculate the loss function for training"""
        # Generate trajectories and get final values
        X_final, u_final = self.generate_trajectories()
        # Calculate terminal condition g(X)
        g_X = self.g_tf(X_final)
        # Calculate mean squared error loss
        loss = torch.mean((g_X - u_final) ** 2)
        return loss

    # Training method
    def train(self, maxiters=150, abstol=1e-8, verbose=True):
        """Train the neural network model"""
        # Training loop for specified number of iterations
        for epoch in range(maxiters):
            # Reset gradients before backward pass
            self.optimizer.zero_grad()
            # Calculate loss
            loss = self.loss_function()
            # Backward pass to compute gradients
            loss.backward()
            # Update model parameters
            self.optimizer.step()

            # Store training history
            self.losses.append(loss.item())
            # Get current u0 estimate
            current_u0 = self.u0(self.x0.unsqueeze(0))[0, 0].item()
            self.u0_history.append(current_u0)

            # Print progress at regular intervals
            if verbose and (epoch % 10 == 0 or epoch == maxiters - 1):
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}, u0: {current_u0:.6f}")

            # Check for convergence
            if loss.item() < abstol:
                if verbose:
                    print(f"Converged at epoch {epoch}")
                break

    # Main solver method with options for bounds calculation
    def solve(
        self,
        limits=False,
        trajectories_upper=1000,
        trajectories_lower=1000,
        maxiters_limits=10,
        verbose=True,
        save_everystep=False,
        maxiters=150,
    ):
        """Main solving function with optional bounds calculation"""

        # Train the main network
        self.train(maxiters=maxiters, verbose=verbose)

        # Get point estimate from trained model
        u0_estimate = self.u0(self.x0.unsqueeze(0))[0, 0].item()

        # Calculate analytical solution for comparison
        u_analytical = self.analytical_solution(self.x0, self.tspan[0]).item()

        # If bounds calculation is not requested
        if not limits:
            # Print results without bounds
            if verbose:
                print(f"Point estimate: {u0_estimate:.6f}")
                print(f"Analytical solution: {u_analytical:.6f}")
                # Calculate relative error
                error = rel_error_l2(u0_estimate, u_analytical)
                print(f"Relative error: {error:.6f}")

            # Define inner class for solution storage
            class PIDESolution:
                def __init__(
                    self, X0, ts, losses, u0_estimate, u0_network, limits=None
                ):
                    self.X0 = X0
                    self.ts = ts
                    self.losses = losses
                    self.us = u0_estimate
                    self.u0 = u0_network
                    self.limits = limits

            # Create time array
            ts_array = (
                torch.arange(self.tspan[0], self.tspan[1] + self.dt / 2, self.dt)
                .cpu()
                .numpy()
            )

            # Return solution with or without per-step history
            if save_everystep:
                return PIDESolution(
                    self.x0.cpu().numpy(),
                    ts_array,
                    self.losses,
                    self.u0_history,
                    self.u0,
                )
            else:
                return PIDESolution(
                    self.x0.cpu().numpy(), ts_array, self.losses, u0_estimate, self.u0
                )

        else:
            # Calculate upper and lower bounds using imported functions
            from .BoundsCalculator import compute_upper_bound, compute_lower_bound

            # Compute upper bound estimate
            u_high = compute_upper_bound(
                self,
                trajectories=trajectories_upper,
                maxiters_limits=maxiters_limits,
                verbose=verbose,
            )

            # Compute lower bound estimate
            u_low = compute_lower_bound(
                self, trajectories=trajectories_lower, verbose=verbose
            )

            # Print results with bounds
            if verbose:
                print(f"\nSolution bounds:")
                print(f"Lower bound: {u_low:.6f}")
                print(f"Point estimate: {u0_estimate:.6f}")
                print(f"Upper bound: {u_high:.6f}")
                print(f"Analytical solution: {u_analytical:.6f}")
                # Check if point estimate is within calculated bounds
                print(f"Within bounds: {u_low <= u0_estimate <= u_high}")

            # Calculate relative error
            error = rel_error_l2(u0_estimate, u_analytical)

            if verbose:
                print(f"Relative error: {error:.6f}")

            # Define inner class for solution storage with bounds
            class PIDESolution:
                def __init__(
                    self, X0, ts, losses, u0_estimate, u0_network, limits=None
                ):
                    self.X0 = X0
                    self.ts = ts
                    self.losses = losses
                    self.us = u0_estimate
                    self.u0 = u0_network
                    self.limits = limits

            # Create time array
            ts_array = (
                torch.arange(self.tspan[0], self.tspan[1] + self.dt / 2, self.dt)
                .cpu()
                .numpy()
            )

            # Return solution with bounds
            if save_everystep:
                return PIDESolution(
                    self.x0.cpu().numpy(),
                    ts_array,
                    self.losses,
                    self.u0_history,
                    self.u0,
                    (u_low, u_high),
                )
            else:
                return PIDESolution(
                    self.x0.cpu().numpy(),
                    ts_array,
                    self.losses,
                    u0_estimate,
                    self.u0,
                    (u_low, u_high),
                )
