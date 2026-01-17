import numpy as np
import torch
import torch.optim as optim
from typing import Optional, Tuple

from .Models import U0Network, SigmaTGradUNetwork


def compute_upper_bound(
    solver, trajectories=1000, maxiters_limits=10, verbose=True
) -> float:
    """Calculate upper bound

    Args:
        solver: BlackScholesBarenblattSolver instance
        trajectories: Number of trajectories, default 1000
        maxiters_limits: Optimization iterations, default 10
        verbose: Whether to print detailed information, default True

    Returns:
        u_high: Upper bound estimate
    """
    if verbose:
        print("Calculating upper bound...")

    device = solver.device
    d = solver.d
    time_steps = solver.time_steps
    hls = solver.hls
    x0 = solver.x0
    tspan = solver.tspan
    dt = solver.dt

    # Create independent network copies
    u0_high = U0Network(d, hls).to(device)
    u0_high.load_state_dict(solver.u0.state_dict())

    sigma_grad_u_high = torch.nn.ModuleList(
        [SigmaTGradUNetwork(d, hls).to(device) for _ in range(time_steps)]
    )
    for i, net in enumerate(sigma_grad_u_high):
        net.load_state_dict(solver.sigma_grad_u[i].state_dict())

    # Optimizer
    high_opt = optim.Adam(
        list(u0_high.parameters())
        + [param for net in sigma_grad_u_high for param in net.parameters()],
        lr=0.01,
    )

    ts = torch.arange(tspan[0], tspan[1] + dt / 2, dt, device=device)

    def upper_bound_loss():
        """Upper bound loss function"""
        total = torch.tensor(0.0, device=device, requires_grad=True)

        for _ in range(trajectories):
            # Forward SDE simulation
            X = x0.clone().unsqueeze(0)  # [1, d]
            X_trajectory = [X.clone()]

            # Forward process - no gradient needed
            with torch.no_grad():
                for i in range(len(ts) - 1):
                    t = ts[i].item()
                    dW = torch.randn(1, d, device=device) * np.sqrt(dt)
                    mu_val = solver.mu_tf(X, t)
                    sigma_val = solver.sigma_tf(X, t)

                    if len(sigma_val.shape) == 2:  # 2D matrix
                        X = X + mu_val * dt + torch.matmul(dW, sigma_val)
                    else:  # 3D tensor
                        dW_expanded = dW.unsqueeze(-1)  # [1, d, 1]
                        X_update = torch.matmul(sigma_val, dW_expanded).squeeze(
                            -1
                        )  # [1, d]
                        X = X + mu_val * dt + X_update

                    X_trajectory.append(X.clone())

            # Reverse calculation for upper bound - gradient needed
            U = solver.g_tf(X)

            for i in range(len(ts) - 2, -1, -1):
                t = ts[i].item()
                X_prev = X_trajectory[i]
                sigma_grad_u_val = sigma_grad_u_high[i](X_prev)
                dW = torch.randn(1, d, device=device) * np.sqrt(dt)

                f_val = solver.phi_tf(X_prev, U, sigma_grad_u_val, t)
                U = (
                    U
                    + f_val * dt
                    - torch.sum(sigma_grad_u_val * dW, dim=1, keepdim=True)
                )

            total = total + U

        return total / trajectories

    # Optimize upper bound
    for i in range(maxiters_limits):
        high_opt.zero_grad()

        # Calculate loss - we want to maximize upper bound, so minimize negative value
        upper_bound = upper_bound_loss()
        loss = -upper_bound  # Negative sign because we want to maximize upper bound
        loss.backward()
        high_opt.step()

        if verbose and (i % 2 == 0 or i == maxiters_limits - 1):
            with torch.no_grad():
                current_bound = -upper_bound_loss().item()
            print(f"Upper bound optimization {i}: {current_bound:.6f}")

    # Calculate final upper bound estimate
    with torch.no_grad():
        final_upper_bound = upper_bound_loss()
        u_high = final_upper_bound.item()

    if verbose:
        print(f"Upper bound: {u_high:.6f}")

    return u_high


def compute_lower_bound(solver, trajectories=1000, verbose=True) -> float:
    """Calculate lower bound using Legendre transform

    Args:
        solver: BlackScholesBarenblattSolver instance
        trajectories: Number of trajectories, default 1000
        verbose: Whether to print detailed information, default True

    Returns:
        u_low: Lower bound estimate
    """
    if verbose:
        print("Calculating lower bound with Legendre transform...")

    device = solver.device
    d = solver.d
    tspan = solver.tspan
    dt = solver.dt
    x0 = solver.x0
    A = solver.A
    u_domain = solver.u_domain
    r = solver.r

    ts = torch.arange(tspan[0], tspan[1] + dt / 2, dt, device=device)
    total_lower = torch.tensor(0.0, device=device)

    for _ in range(trajectories):
        # Initialize u and X
        u = solver.u0(x0.unsqueeze(0))[0, 0]  # Scalar
        X = x0.clone()  # 1D tensor [d]
        I = torch.tensor(0.0, device=device)
        Q = torch.tensor(0.0, device=device)

        for i in range(len(ts) - 1):
            t = ts[i].item()

            # Get σᵀ∇u
            sigma_grad_u_val = solver.sigma_grad_u[i](X.unsqueeze(0)).squeeze(0)  # [d]
            dW = torch.randn(d, device=device) * np.sqrt(dt)  # [d]

            # Update u
            X_2d = X.unsqueeze(0)  # [1, d]
            u_2d = u.unsqueeze(0).unsqueeze(-1)  # [1, 1]
            sigma_grad_u_val_2d = sigma_grad_u_val.unsqueeze(0)  # [1, d]

            f_val = solver.phi_tf(X_2d, u_2d, sigma_grad_u_val_2d, t)[0, 0]  # Scalar

            # Dot product calculation
            dot_product = torch.dot(sigma_grad_u_val, dW)
            u = u - f_val * dt + dot_product

            # Update X
            mu_val = solver.mu_tf(X, t)  # [d]
            sigma_val = solver.sigma_tf(X, t)  # [d, d]

            # Matrix-vector multiplication
            X_update = torch.matmul(sigma_val, dW.unsqueeze(-1)).squeeze(-1)  # [d]
            X = X + mu_val * dt + X_update

            # Legendre transform
            # Calculate dot product sum(X * sigma_grad_u_val)
            X_dot_sigma_grad_u = torch.sum(X * sigma_grad_u_val)

            # Calculate f matrix
            f_matrix = r * (u_domain - X_dot_sigma_grad_u)  # Shape: [n_u]

            # Calculate Legendre transform value
            a_expanded = A.unsqueeze(1)  # Shape: [n_A, 1]
            u_expanded = u_domain.unsqueeze(0)  # Shape: [1, n_u]
            f_expanded = f_matrix.unsqueeze(0)  # Shape: [1, n_u]

            # Calculate a*u - f(u) for all combinations of a and u
            le_matrix = a_expanded * u_expanded - f_expanded  # Shape: [n_A, n_u]

            # Take maximum for each a (over u dimension)
            legendre_values, _ = torch.max(le_matrix, dim=1)  # Shape: [n_A]

            # Find optimal control a: maximize a*u - F(a)
            a_u_minus_F = A * u - legendre_values
            optimal_idx = torch.argmax(a_u_minus_F)
            a_optimal = A[optimal_idx]

            # Get Legendre transform value for optimal a
            F_optimal = legendre_values[optimal_idx]

            # Accumulate dual process
            I = I + a_optimal * dt
            Q = Q + torch.exp(I) * F_optimal

        g_X = solver.g_tf(X.unsqueeze(0))[0, 0]  # Add batch dimension to calculate g(X)
        total_lower = total_lower + torch.exp(I) * g_X - Q

    u_low = (total_lower / trajectories).item()

    if verbose:
        print(f"Lower bound: {u_low:.6f}")

    return u_low
