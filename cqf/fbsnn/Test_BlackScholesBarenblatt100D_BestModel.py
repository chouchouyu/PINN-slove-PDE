import sys
import os
import numpy as np
import torch
import time
import datetime
import warnings

warnings.filterwarnings("ignore")
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fbsnn.BlackScholesBarenblatt import BlackScholesBarenblatt, u_exact
from fbsnn.Utils import figsize, set_seed, setup_device

 
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


class FormalTrainer:
    """Formal Trainer: Train BlackScholesBarenblatt model using optimal parameters from Optuna optimization"""

    def __init__(self, model_path=None, study_path=None):
        """
        Initialize formal trainer

        Parameters:
        model_path: Path to saved best model
        study_path: Path to saved Optuna study
        """
        self.model_path = model_path
        self.study_path = study_path
        self.loaded_model = None
        self.best_params = None
        self.Xi = None
        self.T = None
        self.D = None

    def load_model_and_params(self):
        """
        Load saved model and parameters
        """
        import joblib

        if self.model_path:
            # Load from model file
            print(f"Loading from model file: {self.model_path}")
            
            # Get best available device
            device, _ = setup_device()
            
            # Load model file with specified device mapping
            save_dict = torch.load(
                self.model_path, 
                weights_only=False,  # Must be False to load hyperparameters
                map_location=device  # Use auto-detected device
            )

            # Extract parameters
            self.best_params = save_dict["best_params"]
            self.Xi = save_dict["Xi"]
            self.T = save_dict["T"]
            self.D = save_dict["D"]

            print(f"Problem dimension: {self.D}D")
            print(f"Time interval: [0, {self.T}]")

            # Build network layers
            n_layers = self.best_params["n_layers"]
            hidden_size = self.best_params["hidden_size"]
            layers = [self.D + 1] + [hidden_size] * n_layers + [1]

            # Create model instance
            M = self.best_params["M"]
            N = 50
            Mm = N ** (1 / 5)
            activation = self.best_params["activation"]
            mode = self.best_params["mode"]

            self.loaded_model = BlackScholesBarenblatt(
                self.Xi, self.T, M, N, self.D, Mm, layers, mode, activation
            )

            # Load model weights
            self.loaded_model.model.load_state_dict(save_dict["model_state_dict"])
            print("✓ Model weights loaded successfully")

            # Print optimal parameters
            print("\nOptimal hyperparameter configuration:")
            for key, value in self.best_params.items():
                print(f"  {key}: {value}")

        elif self.study_path:
            # Load from study file
            print(f"Loading from study file: {self.study_path}")
            study = joblib.load(self.study_path)
            self.best_params = study.best_trial.params

            # Print optimal parameters
            print("\nOptimal hyperparameter configuration:")
            for key, value in self.best_params.items():
                print(f"  {key}: {value}")

        else:
            raise ValueError("Must provide either model path or study path")

    def get_model(self):
        """
        Get loaded model instance
        """
        if not self.loaded_model:
            raise ValueError("Please load model and parameters first")
        return self.loaded_model

    def get_params(self):
        """
        Get optimal parameters
        """
        if not self.best_params:
            raise ValueError("Please load model and parameters first")
        return self.best_params

    def get_problem_params(self):
        """
        Get problem parameters
        """
        if self.Xi is None or self.T is None or self.D is None:
            raise ValueError("Please load model and parameters first")
        return self.Xi, self.T, self.D


def run_model(model, N_Iter1, learning_rate1, Xi, T, D, M):
    """
    Run model training and evaluation
    
    Parameters:
    model: Neural network model instance
    N_Iter1: Number of training iterations
    learning_rate1: Learning rate for training
    Xi: Initial state
    T: Terminal time
    D: Dimension
    M: Batch size
    """
    # Quick reproduction: skip retraining and use pre-trained model directly
    total_start_time = time.time()
    samples = 5  # Number of sample trajectories to visualize
    print(f"Using device: {model.device}")
    
    # Generate timestamp for file naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate test data
    t_test, W_test = model.fetch_minibatch()
    X_pred, Y_pred = model.predict(Xi, t_test, W_test)

    # Convert tensors to numpy arrays if needed
    if type(t_test).__module__ != "numpy":
        t_test = t_test.cpu().numpy()
    if type(X_pred).__module__ != "numpy":
        X_pred = X_pred.cpu().detach().numpy()
    if type(Y_pred).__module__ != "numpy":
        Y_pred = Y_pred.cpu().detach().numpy()

    # Calculate exact solution for comparison
    Y_test = np.reshape(
        u_exact(
            np.reshape(t_test[0:M, :, :], [-1, 1]),
            np.reshape(X_pred[0:M, :, :], [-1, D]),
            T,
        ),
        [M, -1, 1],
    )

    # Create output directory
    save_dir = "Figures"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Plot learned vs exact solution
    plt.figure(figsize=figsize(1))
    plt.plot(t_test[0:1, :, 0].T, Y_pred[0:1, :, 0].T, "b", label="Learned $u(t,X_t)$")
    plt.plot(t_test[0:1, :, 0].T, Y_test[0:1, :, 0].T, "r--", label="Exact $u(t,X_t)$")
    plt.plot(t_test[0:1, -1, 0], Y_test[0:1, -1, 0], "ko", label="$Y_T = u(T,X_T)$")

    # Plot additional sample trajectories
    plt.plot(t_test[1:samples, :, 0].T, Y_pred[1:samples, :, 0].T, "b")
    plt.plot(t_test[1:samples, :, 0].T, Y_test[1:samples, :, 0].T, "r--")
    plt.plot(t_test[1:samples, -1, 0], Y_test[1:samples, -1, 0], "ko")

    plt.plot([0], Y_test[0, 0, 0], "ks", label="$Y_0 = u(0,X_0)$")

    plt.xlabel("$t$")
    plt.ylabel("$Y_t = u(t,X_t)$")
    plt.title(
        "D="
        + str(D)
        + " Black-Scholes-Barenblatt, "
        + model.mode
        + "-"
        + model.activation
    )
    plt.legend()
    plt.savefig(
        f"{save_dir}/BSB_{model.D}D_{model.mode}_{model.activation}_{timestamp}.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Calculate and plot relative errors
    errors = np.sqrt((Y_test - Y_pred) ** 2 / Y_test**2)
    mean_errors = np.mean(errors, 0)
    std_errors = np.std(errors, 0)

    plt.figure(figsize=figsize(1))
    plt.plot(t_test[0, :, 0], mean_errors, "b", label="mean")
    plt.plot(
        t_test[0, :, 0],
        mean_errors + 2 * std_errors,
        "r--",
        label="mean + two standard deviations",
    )
    plt.xlabel("$t$")
    plt.ylabel("relative error")
    plt.title(
        "D="
        + str(D)
        + " Black-Scholes-Barenblatt, "
        + model.mode
        + "-"
        + model.activation
    )
    plt.legend()
    plt.savefig(
        f"{save_dir}/BSB_{model.D}D_{model.mode}_{model.activation}_{timestamp}_Errors.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Show all plots
    plt.show()

    return model


if __name__ == "__main__":
    # Configuration parameters
    MODEL_PATH = "optuna_outcomes/models/bsb_best_model_20260118_150740.pth"  # Path to saved best model
    STUDY_PATH = "optuna_outcomes/studies/bsb_optuna_study.pkl"  # Path to saved study
    REPORT_PATH = "optuna_outcomes/reports/bsb_optuna_report_20260118_150807.txt"  # Optuna optimization report path

    # Set random seed for reproducibility
    set_seed(42)
    
    """Main function"""
    print("=" * 80)
    print("           Black-Scholes-Barenblatt Integrated Model Trainer")
    print("           Based on Optuna Optimization Results")
    print("=" * 80)

    # Check if files exist
    if os.path.exists(MODEL_PATH):
        use_model_path = True
        print(f"✓ Found model file: {MODEL_PATH}")
    elif os.path.exists(STUDY_PATH):
        use_model_path = True
        print(f"✓ Found study file: {STUDY_PATH}")
    else:
        print("❌ No model file or study file found")
        sys.exit(1)

    try:
        # Create formal trainer instance
        if use_model_path:
            trainer = FormalTrainer(model_path=MODEL_PATH)
        else:
            trainer = FormalTrainer(study_path=STUDY_PATH)
            # If using study file, provide basic problem parameters
            trainer.Xi = np.array([1.0, 0.5] * (50 // 2))[None, :]  # 50-dimensional example
            trainer.T = 1.0
            trainer.D = 50

        # Load model and parameters
        trainer.load_model_and_params()

        # Get model and parameters
        model = trainer.get_model()
        best_params = trainer.get_params()
        Xi, T, D = trainer.get_problem_params()

        # Automatically read optimal parameters from Optuna report file
        def load_optuna_report_params(report_path):
            """
            Read optimal parameters from Optuna report file
            """
            import re
            
            with open(report_path, 'r') as f:
                content = f.read()
            
            params = {}
            
            # Extract optimal hyperparameter configuration section
            match = re.search(r'Optimal hyperparameter configuration:(.*?)\n\nTrial statistics:', content, re.DOTALL)
            if not match:
                # If trial statistics section not found, try matching end of file
                match = re.search(r'Optimal hyperparameter configuration:(.*?)$', content, re.DOTALL)
            
            if match:
                params_section = match.group(1)
                
                # Extract individual parameters
                param_patterns = [
                    (r'n_layers: (\d+)', 'n_layers', int),
                    (r'hidden_size: (\d+)', 'hidden_size', int),
                    (r'activation: ([\w]+)', 'activation', str),
                    (r'mode: ([\w]+)', 'mode', str),
                    (r'M: (\d+)', 'M', int),
                    (r'learning_rate1: ([\d.e+-]+)', 'learning_rate1', float),
                    (r'n_iter1: (\d+)', 'n_iter1', int),
                ]
                
                for pattern, key, dtype in param_patterns:
                    match = re.search(pattern, params_section)
                    if match:
                        params[key] = dtype(match.group(1))
            
            # Add hardcoded time steps N from Optuna.py
            params['N'] = 50  # Fixed value from Optuna.py
            
            return params
        
        # Use optimal parameters from Optuna report (override parameters from model file)
        print("\n🔧 Loading optimal parameters from Optuna report file...")
        report_best_params = load_optuna_report_params(REPORT_PATH)
        print(f"✓ Parameters read from report: {report_best_params.keys()}")

        # Update best parameters
        best_params.update(report_best_params)
        print("✓ Optuna report parameters applied")

        # Set training parameters
        N_Iter1 = best_params.get("n_iter1", 20000)  # Use 20000 iterations from report
        learning_rate1 = best_params.get("learning_rate1", 0.00023345864076016249)  # Use learning rate from report
        
        # Also update model's N value (number of time steps)
        model.N = best_params.get("N", model.N)
        print(f"\n📊 Updated key parameters:")
        print(f"   N (time steps): {model.N}")
        print(f"   n_iter1 (training steps): {N_Iter1}")
        print(f"   learning_rate1: {learning_rate1}")

        print("\n" + "=" * 80)
        print("           Starting Model Training")
        print("=" * 80)
        print(f"Training phase: {N_Iter1} iterations, learning_rate={learning_rate1}")

        # Get problem parameters
        Xi, T, D = trainer.get_problem_params()
        M = model.M  # Get batch size

        # Print input parameters before calling run_model
        print("\n" + "=" * 60)
        print("run_model input parameters:")
        print("=" * 60)
        print(f"N_Iter1: {N_Iter1}")
        print(f"learning_rate1: {learning_rate1}")
        print(
            f"Xi shape: {Xi.shape}, Xi first few values: {Xi[0, :3] if Xi.size > 0 else 'empty'}"
        )
        print(f"T: {T}")
        print(f"D: {D}")
        print(f"M: {M}")
        print(f"model.mode: {model.mode}")
        print(f"model.activation: {model.activation}")
        print(f"model.D: {model.D}")
        print(f"model.M: {model.M}")
        print(f"model.N: {model.N}")
        print("=" * 60)

        # Pass parameters to run_model function
        # Run model training and evaluation
        final_model = run_model(
            model, N_Iter1, learning_rate1, Xi, T, D, M
        )

        print("\n" + "=" * 80)
        print("          Training Completed!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)