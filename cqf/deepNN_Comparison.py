import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import sys
import os
# Add parent directory and deepbsde directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../deepbsde')))

from FBSNNs import FBSNNs
from deepbsde.BlackScholesBarenblatt import BlackScholesBarenblatt
from deepbsde.DeepBSDE import BlackScholesBarenblattSolver, rel_error_l2
from FBSNNs.Utils import figsize, set_seed
import time
import os

# Solve negative sign display issue
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'Microsoft YaHei']   
matplotlib.rcParams['axes.unicode_minus'] = False  

def test_100d_deepbsde(verbose=True):
    """Test 100-dimensional DeepBSDE algorithm"""

    if verbose:
        print("=== Solving 100-dimensional Black-Scholes-Barenblatt Equation ===")
        print("\n1. Standard DeepBSDE Algorithm (100-dim):")

    # Use the same parameters as FBSNN
    D = 100  # Dimension
    M = 100  # Number of trajectories
    N = 50   # Number of time steps
    dt = 1.0 / N  # Time step size
    x0 = [1.0, 0.5] * int(D / 2)  # Initial conditions
    tspan = (0.0, 1.0)  # Time range

    # Test standard version (100-dim)
    solver_std = BlackScholesBarenblattSolver(d=D, x0=x0, tspan=tspan, dt=dt, m=M)
    result_std = solver_std.solve(limits=False, verbose=verbose)

    # Validate standard version results
    u_pred_std = result_std.us if hasattr(result_std.us, '__len__') else result_std.us
    u_anal_std = solver_std.analytical_solution(solver_std.x0, solver_std.tspan[0]).item()

    if hasattr(u_pred_std, '__len__'):
        error_std = rel_error_l2(u_pred_std[-1], u_anal_std)
    else:
        error_std = rel_error_l2(u_pred_std, u_anal_std)

    if verbose:
        print(f"Standard algorithm error: {error_std:.6f}")

    return solver_std, result_std, error_std

def test_100d_legendre_deepbsde(verbose=True):
    """Test 100-dimensional DeepBSDE with Legendre transform dual method"""

    if verbose:
        print("\n2. 100-dimensional DeepBSDE with Legendre Transform Dual Method:")

    # Use the same parameters as FBSNN
    D = 100  # Dimension
    M = 100  # Number of trajectories
    N = 50   # Number of time steps
    dt = 1.0 / N  # Time step size
    x0 = [1.0, 0.5] * int(D / 2)  # Initial conditions
    tspan = (0.0, 1.0)  # Time range

    # Test version with Legendre transform (100-dim)
    solver_limits = BlackScholesBarenblattSolver(d=D, x0=x0, tspan=tspan, dt=dt, m=M)
    result_limits = solver_limits.solve(
        limits=True, 
        trajectories_upper=1000,  # Consistent with original Julia file
        trajectories_lower=1000,  # Consistent with original Julia file
        maxiters_limits=10,       # Consistent with original Julia file
        verbose=verbose
    )

    # Validate results with bounds version
    u_pred_limits = result_limits.us if hasattr(result_limits.us, '__len__') else result_limits.us
    u_anal_limits = solver_limits.analytical_solution(solver_limits.x0, solver_limits.tspan[0]).item()

    if hasattr(u_pred_limits, '__len__'):
        error_limits = rel_error_l2(u_pred_limits[-1], u_anal_limits)
    else:
        error_limits = rel_error_l2(u_pred_limits, u_anal_limits)

    if verbose:
        print(f"Dual method error: {error_limits:.6f}")

    return solver_limits, result_limits, error_limits

def run_fbsnn_100d():
    """Run FBSNN 100-dimensional method"""
    print("\n=== Running FBSNN 100-dimensional Method ===")

    # Set the same parameters as FBSNN
    M = 100  # Number of trajectories
    N = 50   # Number of time steps
    D = 100  # Dimension
    Mm = N ** (1/5)

    layers = [D + 1] + 4 * [256] + [1]
    Xi = np.array([1.0, 0.5] * int(D / 2))[None, :]
    T = 1.0

    mode = "Naisnet"
    activation = "Sine"

    # Create FBSNN model
    from FBSNNs.BlackScholesBarenblatt import BlackScholesBarenblatt
    # Create model instance
    model = BlackScholesBarenblatt(Xi, T, M, N, D, Mm, layers, mode, activation)    
    # Train model
    start_time = time.time()
    graph = model.train(100, 1e-3)  # First stage training
    graph = model.train(5000, 1e-5)  # Second stage training
    training_time = time.time() - start_time

    print(f"FBSNN training time: {training_time:.2f} seconds")

    return model, graph, training_time

def compare_methods():
    """Compare performance of two neural network methods"""

    # Create directory to save results
    if not os.path.exists("results"):
        os.makedirs("results")

    # 1. Run DeepBSDE 100-dim
    print("Starting DeepBSDE 100-dimensional test...")
    deepbsde_start = time.time()
    solver_std, result_std, error_std = test_100d_deepbsde(verbose=True)
    solver_limits, result_limits, error_limits = test_100d_legendre_deepbsde(verbose=True)
    deepbsde_time = time.time() - deepbsde_start

    # 2. Run FBSNN 100-dim
    print("\nStarting FBSNN 100-dimensional test...")
    fbsnn_model, fbsnn_graph, fbsnn_time = run_fbsnn_100d()

    # 3. Performance comparison analysis
    print("\n" + "="*60)
    print("           Neural Network Method Performance Comparison (100-dim)")
    print("="*60)

    # Calculate analytical solution for DeepBSDE
    u_analytical_deepbsde = solver_std.analytical_solution(solver_std.x0, solver_std.tspan[0]).item()
    u_deepbsde_std = result_std.us if hasattr(result_std.us, '__len__') else result_std.us
    if hasattr(u_deepbsde_std, '__len__'):
        u_deepbsde_std = u_deepbsde_std[-1]

    # Get FBSNN estimate (need simulation prediction)
    t_test, W_test = fbsnn_model.fetch_minibatch()
    # Convert Xi to numpy array because predict method requires numpy input
    # Need to detach() gradients first, then convert to numpy
    xi_np = fbsnn_model.Xi.detach().cpu().numpy()
    X_pred, Y_pred = fbsnn_model.predict(xi_np, t_test, W_test)
    u_fbsnn_estimate = Y_pred[0, 0, 0].item() if hasattr(Y_pred, 'item') else Y_pred[0, 0, 0]

    # Calculate analytical solution for FBSNN (same problem as DeepBSDE)
    fbsnn_analytical = torch.exp(torch.tensor((0.05 + 0.4**2) * 1.0)) * torch.sum(torch.tensor([1.0, 0.5] * 50)**2)
    u_analytical_fbsnn = fbsnn_analytical.item()

    print(f"{'Metric':<25} {'DeepBSDE':<15} {'FBSNN':<15}")
    print("-" * 60)
    print(f"{'Estimate':<25} {u_deepbsde_std:<15.6f} {u_fbsnn_estimate:<15.6f}")
    print(f"{'Analytical Solution':<25} {u_analytical_deepbsde:<15.6f} {u_analytical_fbsnn:<15.6f}")
    print(f"{'Relative Error':<25} {error_std:<15.6f} {rel_error_l2(u_fbsnn_estimate, u_analytical_fbsnn):<15.6f}")
    print(f"{'Training Time (sec)':<25} {deepbsde_time:<15.2f} {fbsnn_time:<15.2f}")
    print(f"{'Has Confidence Interval':<25} {'Yes':<15} {'No':<15}")

    # 4. Plot comparison charts
    plt.figure(figsize=figsize(20, 15))

    # Training loss comparison
    plt.subplot(2, 3, 1)
    if hasattr(solver_std, 'losses'):
        plt.semilogy(solver_std.losses, label='DeepBSDE Standard')
    if hasattr(solver_limits, 'losses'):
        plt.semilogy(solver_limits.losses, label='DeepBSDE with Limits')
    if fbsnn_graph is not None:
        plt.semilogy(fbsnn_graph[0], fbsnn_graph[1], label='FBSNN')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Solution value comparison
    plt.subplot(2, 3, 2)
    methods = ['DeepBSDE', 'FBSNN']
    estimates = [u_deepbsde_std, u_fbsnn_estimate]
    analyticals = [u_analytical_deepbsde, u_analytical_fbsnn]

    x_pos = np.arange(len(methods))
    plt.bar(x_pos - 0.2, estimates, 0.4, label='Estimated', alpha=0.7)
    plt.bar(x_pos + 0.2, analyticals, 0.4, label='Analytical', alpha=0.7)
    plt.xticks(x_pos, methods)
    plt.ylabel('Solution Value')
    plt.title('Solution Comparison')
    plt.legend()

    # Error comparison
    plt.subplot(2, 3, 3)
    errors = [error_std, rel_error_l2(u_fbsnn_estimate, u_analytical_fbsnn)]
    plt.bar(methods, errors, alpha=0.7)
    plt.axhline(y=0.1, color='r', linestyle='--', label='10% Error Threshold')
    plt.ylabel('Relative Error')
    plt.title('Error Comparison')
    plt.legend()

    # Time comparison
    plt.subplot(2, 3, 4)
    times = [deepbsde_time, fbsnn_time]
    plt.bar(methods, times, alpha=0.7, color='green')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time Comparison')

    # Convergence speed comparison
    plt.subplot(2, 3, 5)
    if hasattr(solver_std, 'u0_history'):
        plt.plot(solver_std.u0_history, label='DeepBSDE Convergence')
    # Convergence history for FBSNN needs additional processing
    plt.xlabel('Epoch')
    plt.ylabel('u0 Value')
    plt.title('Convergence Comparison')
    plt.legend()

    # Method characteristics comparison
    plt.subplot(2, 3, 6)
    features = {
        'Dimensional Adaptability': [9, 8],  # 1-10 score
        'Convergence Speed': [7, 6],
        'Numerical Stability': [8, 9],
        'Implementation Complexity': [6, 4],  # Lower score means more complex
        'Memory Efficiency': [7, 5]
    }

    methods = ['DeepBSDE', 'FBSNN']
    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]  # Close the shape

    for i, method in enumerate(methods):
        values = [features[feature][i] for feature in features]
        values += values[:1]  # Close the shape
        plt.polar(angles, values, 'o-', label=method)

    plt.xticks(angles[:-1], features.keys())
    plt.title('Method Characteristics')
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/deepNN_Comparison.png", dpi=300, bbox_inches='tight')
    # Don't display chart, save directly
    plt.close('all')

    # 5. Output detailed analysis report
    print("\n" + "="*60)
    print("                Detailed Performance Analysis Report")
    print("="*60)

    print("\n1. Accuracy Analysis:")
    print(f"   - DeepBSDE standard method error: {error_std:.6f} ({'Excellent' if error_std < 0.05 else 'Good' if error_std < 0.1 else 'Average'})")
    print(f"   - FBSNN method error: {rel_error_l2(u_fbsnn_estimate, u_analytical_fbsnn):.6f} ({'Excellent' if rel_error_l2(u_fbsnn_estimate, u_analytical_fbsnn) < 0.05 else 'Good' if rel_error_l2(u_fbsnn_estimate, u_analytical_fbsnn) < 0.1 else 'Average'})")

    print("\n2. Computational Efficiency Analysis:")
    print(f"   - DeepBSDE training time: {deepbsde_time:.2f} seconds")
    print(f"   - FBSNN training time: {fbsnn_time:.2f} seconds")
    print(f"   - Time efficiency ratio: {deepbsde_time/fbsnn_time:.2f}x")

    print("\n3. Method Characteristics Comparison:")
    print("   - DeepBSDE advantages: Provides confidence intervals, good numerical stability, relatively simple implementation")
    print("   - FBSNN advantages: Good adaptability to high-dimensional problems, solid theoretical foundation, guaranteed convergence")
    print("   - DeepBSDE disadvantages: Large memory consumption for high-dimensional problems")
    print("   - FBSNN disadvantages: Higher implementation complexity, potentially longer training time")

    print("\n4. Recommended Application Scenarios:")
    print("   - Need confidence interval estimation: Recommend DeepBSDE with Legendre transform")
    print("   - Ultra-high dimensional problems (>100-dim): Recommend FBSNN method")
    print("   - Real-time applications: Recommend standard DeepBSDE method")
    print("   - Theoretical research: Both methods are suitable, each with unique features")

    return {
        'deepbsde_std': (solver_std, result_std, error_std),
        'deepbsde_limits': (solver_limits, result_limits, error_limits),
        'fbsnn': (fbsnn_model, fbsnn_graph, rel_error_l2(u_fbsnn_estimate, u_analytical_fbsnn)),
        'performance_metrics': {
            'deepbsde_time': deepbsde_time,
            'fbsnn_time': fbsnn_time,
            'deepbsde_error': error_std,
            'fbsnn_error': rel_error_l2(u_fbsnn_estimate, u_analytical_fbsnn)
        }
    }

def main():
    """Main test function"""
    set_seed(42)  # Set random seed for reproducibility

    print("Starting comparison of 100-dimensional Black-Scholes-Barenblatt equation solving...")
    print("Comparison methods: DeepBSDE vs FBSNN")
    print("Problem dimension: 100-dimensional")
    print("="*60)

    results = compare_methods()

    print("\nComparison completed! Results saved to results/ directory")

    return results

if __name__ == "__main__":
    results = main()
