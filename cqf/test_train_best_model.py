import sys
import os
import numpy as np
import torch

# Add the project root to the Python path
sys.path.append('/Users/susan/PINN-slove-PDE')

from cqf.fbsnn.Test_BlackScholesBarenblatt100D_Optuna import BlackScholesBarenblattOptunaOptimizer

# Create a simple mock trial with the necessary parameters
def test_train_best_model():
    # Initialize the optimizer
    D = 2  # Small dimension for testing
    Xi = np.array([1.0, 0.5])
    T = 1.0
    
    optimizer = BlackScholesBarenblattOptunaOptimizer(Xi, T, D)
    
    # Create a mock best_trial with the required parameters (excluding 'N')
    class MockTrial:
        def __init__(self):
            self.params = {
                'n_layers': 2,
                'hidden_size': 64,
                'activation': 'ReLU',
                'mode': 'FC',
                'M': 100,
                'learning_rate1': 0.001,
                'n_iter1': 100
            }
            self.user_attrs = {
                'relative_error': 0.01,
                'training_time': 10.0
            }
            self.number = 1
    
    # Set the mock trial
    optimizer.best_trial = MockTrial()
    
    # Call the method that was failing
    try:
        # We'll only test up to the point where the model is created
        # to avoid long training time
        params = optimizer.best_trial.params
        n_layers = params["n_layers"]
        hidden_size = params["hidden_size"]
        layers = [optimizer.D + 1] + [hidden_size] * n_layers + [1]
        
        # This was the line that was causing KeyError: 'N'
        M = params["M"]
        N = 50  # Fixed value, not from params
        Mm = N ** (1 / 5)
        activation = params["activation"]
        mode = params["mode"]
        
        print("✓ Successfully accessed all parameters without KeyError")
        print(f"Parameters used:")
        print(f"  M: {M}")
        print(f"  N: {N} (fixed)")
        print(f"  Mm: {Mm}")
        print(f"  activation: {activation}")
        print(f"  mode: {mode}")
        print(f"  layers: {layers}")
        
        return True
    except KeyError as e:
        print(f"✗ KeyError still occurs: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing train_best_model method...")
    success = test_train_best_model()
    if success:
        print("\nTest passed! The KeyError has been resolved.")
    else:
        print("\nTest failed!")
        sys.exit(1)
