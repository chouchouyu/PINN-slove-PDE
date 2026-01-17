import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
from cqf.deepbsde.BlackScholesBarenblatt import BlackScholesBarenblatt
from cqf.deepbsde.DeepBSDE import BlackScholesBarenblattSolver, rel_error_l2
from cqf.fbsnn.Utils import figsize

# Set model parameters - same structure as BlackScholesBarenblatt100D.py
D = 30  # Problem dimension (30D Black-Scholes-Barenblatt equation)
M = 100  # Number of trajectories per training batch
N = 50  # Number of discretization points on the time axis

# Create Figures directory
figures_dir = "Figures"
# Check if the directory exists
if not os.path.exists(figures_dir):
    # Create the directory if it doesn't exist
    os.makedirs(figures_dir)

# Print header for the 30D BSB equation solving
print("=== 30D Black-Scholes-Barenblatt Equation Solving ===")
print("Using DeepBSDE method")

# Create solver instance
start_time = time.time()
# Initialize the solver with dimension D
solver = BlackScholesBarenblattSolver(d=D)

# Train the solver - using standard DeepBSDE algorithm
# limits=False means we don't compute upper/lower bounds
result = solver.solve(limits=False, verbose=True)

# Calculate total computation time
total_time = time.time() - start_time
print(f"Total computation time: {total_time:.2f} seconds")

# Get training history
# Create list of iteration numbers
iterations = list(range(len(solver.losses)))
# Store training losses
training_loss = solver.losses

# 1. Plot the training loss curve
plt.figure(figsize=figsize(1))
# Plot iterations vs training loss with blue line
plt.plot(iterations, training_loss, 'b')
# Set x-axis label
plt.xlabel('Iterations')
# Set y-axis label
plt.ylabel('Loss')
# Use log scale for y-axis
plt.yscale("log")
# Set plot title
plt.title('Evolution of the training loss')
# Save the figure to file
plt.savefig(f"{figures_dir}/DeepBSDE_BlackScholesBarenblattSolver30D_Loss.png")
# Close the plot to free memory
plt.close()
print("Training loss curve saved")

# Calculate relative error between DeepBSDE estimate and analytical solution
# Get initial condition
x0 = solver.x0
# Get DeepBSDE estimated solution
u_deepbsde = result.us
# Calculate analytical solution at time 0
u_analytical = solver.analytical_solution(x0, 0).item()
# Calculate relative L2 error
relative_error = rel_error_l2(u_deepbsde, u_analytical)

# Output summary of results
print("\n=== Results Summary ===")
print(f"DeepBSDE estimate (u0): {u_deepbsde:.6f}")
print(f"Analytical solution (u0): {u_analytical:.6f}")
print(f"Relative error: {relative_error:.6f} ({relative_error*100:.4f}%)")
# Print path where results are saved
print(f"\nResult images saved to {os.path.abspath(figures_dir)}/DeepBSDE_BlackScholesBarenblattSolver30D_Loss.png")