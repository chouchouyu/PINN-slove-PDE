import numpy as np
import matplotlib.pyplot as plt

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from cqf.deepbsde.BlackScholesBarenblatt import BlackScholesBarenblatt
from cqf.deepbsde.DeepBSDE import BlackScholesBarenblattSolver, rel_error_l2


def test_standard_deepbsde(d=30):
    """Test standard DeepBSDE algorithm

    Parameters:
    d: Problem dimension, default is 30

    Returns:
    solver_std: Standard algorithm solver
    result_std: Standard algorithm solving result
    error_std: Standard algorithm error
    """
    # Print header for 30D BSB equation solving
    print("=== 30D Black-Scholes-Barenblatt Equation Solving ===")
    # Print description of standard algorithm
    print("\n1. Standard DeepBSDE Algorithm:")

    # Test standard version (limits=false)
    # Create solver instance with dimension d
    solver_std = BlackScholesBarenblattSolver(d=d)
    # Solve the equation without bounds calculation
    result_std = solver_std.solve(limits=False, verbose=True)

    # Validate standard version results
    # Get predicted solution, handle both scalar and array cases
    u_pred_std = result_std.us if hasattr(result_std.us, "__len__") else result_std.us
    # Get analytical solution for the initial condition
    u_anal_std = solver_std.analytical_solution(
        solver_std.x0, solver_std.tspan[0]
    ).item()
    # Calculate error based on data type
    if hasattr(u_pred_std, "__len__"):
        # If u_pred_std is an array, use the last element
        error_std = rel_error_l2(u_pred_std[-1], u_anal_std)
    else:
        # If u_pred_std is a scalar, use directly
        error_std = rel_error_l2(u_pred_std, u_anal_std)

    # Print the standard algorithm error
    print(f"Standard algorithm error: {error_std:.6f}")

    # Return solver, result, and error
    return solver_std, result_std, error_std


def test_legendre_deepbsde(d=30):
    """Test DeepBSDE with Legendre transform dual method

    Parameters:
    d: Problem dimension, default is 30

    Returns:
    solver_limits: Solver with Legendre transform
    result_limits: Solving result with Legendre transform
    error_limits: Solving error with Legendre transform
    """
    # Print description of Legendre transform method
    print("\n2. DeepBSDE with Legendre Transform Dual Method:")

    # Test version with Legendre transform (limits=true)
    # Create solver instance with dimension d
    solver_limits = BlackScholesBarenblattSolver(d=d)
    # Solve the equation with bounds calculation
    result_limits = solver_limits.solve(
        limits=True,  # Enable bounds calculation
        trajectories_upper=1000,  # Number of trajectories for upper bound (matches Julia original)
        trajectories_lower=1000,  # Number of trajectories for lower bound (matches Julia original)
        maxiters_limits=10,  # Optimization iterations for bounds (matches Julia original)
        verbose=True,
    )

    # Validate bounded version results
    # Get predicted solution, handle both scalar and array cases
    u_pred_limits = (
        result_limits.us if hasattr(result_limits.us, "__len__") else result_limits.us
    )
    # Get analytical solution for the initial condition
    u_anal_limits = solver_limits.analytical_solution(
        solver_limits.x0, solver_limits.tspan[0]
    ).item()
    # Calculate error based on data type
    if hasattr(u_pred_limits, "__len__"):
        # If u_pred_limits is an array, use the last element
        error_limits = rel_error_l2(u_pred_limits[-1], u_anal_limits)
    else:
        # If u_pred_limits is a scalar, use directly
        error_limits = rel_error_l2(u_pred_limits, u_anal_limits)

    # Print the dual method error
    print(f"Dual method error: {error_limits:.6f}")

    # Return solver, result, and error
    return solver_limits, result_limits, error_limits


# Main execution block
if __name__ == "__main__":
    # Call the two test methods
    # Test standard DeepBSDE
    solver_std, result_std, error_std = test_standard_deepbsde()
    # Test DeepBSDE with Legendre transform
    solver_limits, result_limits, error_limits = test_legendre_deepbsde()

    # Plot training curves
    # Create figure with 1 row, 3 columns, and specified size
    plt.figure(figsize=(12, 4))

    # First subplot: Standard algorithm loss
    plt.subplot(1, 3, 1)
    # Plot loss curve with log scale on y-axis
    plt.semilogy(solver_std.losses)
    # Set x-axis label
    plt.xlabel("Epoch")
    # Set y-axis label
    plt.ylabel("Loss")
    # Set subplot title
    plt.title("Training Loss (Standard)")
    # Add grid with transparency
    plt.grid(True, alpha=0.3)

    # Second subplot: Legendre method loss
    plt.subplot(1, 3, 2)
    # Plot loss curve with log scale on y-axis
    plt.semilogy(solver_limits.losses)
    # Set x-axis label
    plt.xlabel("Epoch")
    # Set y-axis label
    plt.ylabel("Loss")
    # Set subplot title
    plt.title("Training Loss (With Limits)")
    # Add grid with transparency
    plt.grid(True, alpha=0.3)

    # Third subplot: Solution with bounds
    plt.subplot(1, 3, 3)
    # Check if bounds are available
    if hasattr(result_limits, "limits") and result_limits.limits is not None:
        # Unpack lower and upper bounds
        u_low, u_high = result_limits.limits
        # Get point estimate, handle both scalar and array cases
        if hasattr(result_limits.us, "__len__"):
            u_point = result_limits.us[-1]
        else:
            u_point = result_limits.us
        # Get analytical solution
        u_anal = solver_limits.analytical_solution(
            solver_limits.x0, solver_limits.tspan[0]
        ).item()

        # Plot analytical solution as green dashed line
        plt.axhline(
            y=u_anal, color="green", linestyle="--", label="Analytical", alpha=0.7
        )
        # Plot point estimate as blue solid line
        plt.axhline(
            y=u_point, color="blue", linestyle="-", label="Point Estimate", alpha=0.7
        )
        # Plot confidence interval as red shaded region
        plt.axhspan(u_low, u_high, alpha=0.3, color="red", label="Confidence Interval")
        # Set y-axis label
        plt.ylabel("Solution Value")
        # Set subplot title
        plt.title("Solution with Bounds")
        # Add legend
        plt.legend()
        # Add grid with transparency
        plt.grid(True, alpha=0.3)

    # Create Figures directory (if it doesn't exist)
    import os

    # Define directory name
    figures_dir = "Figures"
    # Check if directory exists
    if not os.path.exists(figures_dir):
        # Create directory if it doesn't exist
        os.makedirs(figures_dir)

    # Adjust subplot layout to prevent overlap
    plt.tight_layout()
    # Save the figure to file
    plt.savefig(os.path.join(figures_dir, "DeepBSDE_standard_vs_Legendre.png"))
    # Close the figure to free memory
    plt.close()

    # Verify results
    print(f"\nVerification Results:")
    # Check if standard algorithm error is acceptable (< 1.0)
    print(
        f"Standard algorithm error: {error_std:.6f} {'✓ < 1.0' if error_std < 1.0 else '✗ >= 1.0'}"
    )
    # Check if Legendre method error is acceptable (< 1.0)
    print(
        f"Dual method error: {error_limits:.6f} {'✓ < 1.0' if error_limits < 1.0 else '✗ >= 1.0'}"
    )

    # Check if point estimate is within calculated bounds
    if hasattr(result_limits, "limits") and result_limits.limits is not None:
        # Unpack lower and upper bounds
        u_low, u_high = result_limits.limits
        # Get point estimate, handle both scalar and array cases
        if hasattr(result_limits.us, "__len__"):
            u_point = result_limits.us[-1]
        else:
            u_point = result_limits.us
        # Check if point estimate is within bounds
        if u_low <= u_point <= u_high:
            print("✓ Point estimate is within lower and upper bounds")
        else:
            print("✗ Point estimate is outside lower and upper bounds, unreliable")
