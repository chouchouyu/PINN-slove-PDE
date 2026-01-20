import sys
import os
import optuna
from optuna.trial import TrialState
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import datetime
import warnings
import optuna
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from cqf.fbsnn.BlackScholesBarenblatt import BlackScholesBarenblatt, u_exact
from cqf.fbsnn.Utils import set_seed

plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "Heiti TC",
    "Microsoft YaHei",
]   
plt.rcParams["axes.unicode_minus"] = False   

class BlackScholesBarenblattOptunaOptimizer:
    """Optuna hyperparameter optimizer for the BlackScholesBarenblatt model"""

    def __init__(self, Xi, T, D):
        """
        Initialize the optimizer

        Parameters:
        Xi: Initial condition
        T: Time interval
        D: Problem dimension
        """
        # Store initial condition
        self.Xi = Xi
        # Store time interval
        self.T = T
        # Store problem dimension
        self.D = D

        # Variables to store the best trial, best model, and the Optuna study
        self.best_trial = None
        self.best_model = None
        self.study = None
        # List to store optimization history
        self.optimization_history = []

    # Define the objective function for Optuna optimization
    def objective(self, trial):
        """Optuna objective function - optimize the BlackScholesBarenblatt model"""
        try:
            # Set random seed for reproducibility
            set_seed(42)

            # 1. Define hyperparameter search space
            # Network architecture parameters
            n_layers = trial.suggest_int("n_layers", 2, 6)
            hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
            activation = trial.suggest_categorical(
                "activation", ["Sine", "ReLU", "Tanh"]
            )
            mode = trial.suggest_categorical("mode", ["FC", "Naisnet"])

            # Training parameters
            M = trial.suggest_int("M", 50, 200, step=50)  # Batch size
            N = 50  # Fixed number of time steps, no longer optimized by Optuna
            Mm = N ** (1 / 5)

            # Learning rate and iteration count (single-stage training)
            learning_rate1 = trial.suggest_float("learning_rate1", 1e-5, 1e-2, log=True)
            n_iter1 = trial.suggest_int("n_iter1", 500, 25000, step=500)

            # Build network layers
            layers = [self.D + 1] + [hidden_size] * n_layers + [1]

            # 2. Create the model
            model = BlackScholesBarenblatt(
                self.Xi, self.T, M, N, self.D, Mm, layers, mode, activation
            )

            # Single-stage training
            start_time = time.time()

            # Train the model
            print(f"Trial {trial.number}: Training...")
            graph = model.train(n_iter1, learning_rate1)

            training_time = time.time() - start_time

            # 4. Evaluate model performance
            t_test, W_test = model.fetch_minibatch()
            X_pred, Y_pred = model.predict(self.Xi, t_test, W_test)

            # Convert to numpy arrays
            if hasattr(t_test, "cpu"):
                t_test = t_test.cpu().numpy()
            if hasattr(X_pred, "cpu"):
                X_pred = X_pred.cpu().detach().numpy()
            if hasattr(Y_pred, "cpu"):
                Y_pred = Y_pred.cpu().detach().numpy()

            # Calculate analytical solution as benchmark
            Y_analytical = u_exact(t_test, X_pred, self.T)

            # Calculate relative error
            relative_error = np.mean(
                np.abs(Y_pred[:, -1, 0] - Y_analytical[:, -1, 0])
                / (np.abs(Y_analytical[:, -1, 0]) + 1e-8)
            )

            # Final loss (combines error and training time)
            final_loss = relative_error + 0.001 * training_time  # Balance accuracy and efficiency

            # Record trial attributes
            trial.set_user_attr("training_time", training_time)
            trial.set_user_attr("relative_error", relative_error)
            trial.set_user_attr("final_loss", final_loss)
            trial.set_user_attr("layers", layers)

            print(
                f"Trial {trial.number}: error={relative_error:.6f}, time={training_time:.2f}s, final_loss={final_loss:.6f}"
            )

            # Clean up GPU memory
            # Clear cache for all devices (both CUDA and MPS)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()

            return final_loss

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            # Clean up GPU memory
            # Clear cache for all devices (both CUDA and MPS)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            return float("inf")

    # Method to perform hyperparameter optimization
    def optimize(
        self, n_trials=50, timeout=7200, study_name="black_scholes_barenblatt_optuna"
    ):
        """Execute hyperparameter optimization"""
        print("=" * 80)
        print("           Black-Scholes-Barenblatt Model Hyperparameter Optimization")
        print("=" * 80)
        print(f"Problem dimension: {self.D}D")
        print(f"Time interval: [0, {self.T}]")
        print(f"Number of trials: {n_trials}")
        print(f"Timeout: {timeout} seconds")
        print()

        # Create an Optuna study
        self.study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.HyperbandPruner(),
        )

        # Execute optimization
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            catch=(Exception,),
        )

        # Save the best result
        self.best_trial = self.study.best_trial
        print(f"\nOptimization completed! Best trial: #{self.best_trial.number}")
        print(f"Best final loss: {self.best_trial.value:.6f}")
        print(f"Relative error: {self.best_trial.user_attrs['relative_error']:.6f}")
        print(f"Training time: {self.best_trial.user_attrs['training_time']:.2f}s")

        return self.best_trial

    # Method to train the final model with the best hyperparameters
    def train_best_model(self, save_model=True):
        """Train the final model using the best hyperparameters"""
        if self.best_trial is None:
            raise ValueError("Please run optimization first!")

        print("\n" + "=" * 80)
        print("           Training Final Model with Best Hyperparameters")
        print("=" * 80)

        # Extract the best parameters
        params = self.best_trial.params
        print("Best hyperparameter configuration:")
        for key, value in params.items():
            print(f"  {key}: {value}")

        # Build network layers
        n_layers = params["n_layers"]
        hidden_size = params["hidden_size"]
        layers = [self.D + 1] + [hidden_size] * n_layers + [1]

        # Create the final model
        M = params["M"]
        N = 50  # Fixed number of time steps, no longer optimized by Optuna
        Mm = N ** (1 / 5)
        activation = params["activation"]
        mode = params["mode"]

        model = BlackScholesBarenblatt(
            self.Xi, self.T, M, N, self.D, Mm, layers, mode, activation
        )

        # Single-stage training
        learning_rate1 = params["learning_rate1"]
        n_iter1 = params["n_iter1"]

        print("Training model...")
        model.train(n_iter1, learning_rate1)

        self.best_model = model

        # Save the model
        if save_model:
            self.save_best_model()

        return model

    # Method to evaluate the model performance
    def evaluate_model(self, model=None, n_samples=1000):
        """Evaluate model performance"""
        if model is None:
            if self.best_model is None:
                raise ValueError("No model to evaluate!")
            model = self.best_model

        print("\n" + "=" * 80)
        print("           Model Evaluation Results")
        print("=" * 80)

        # Generate test data
        t_test, W_test = model.fetch_minibatch()
        X_pred, Y_pred = model.predict(self.Xi, t_test, W_test)

        # Repeat sampling to get more test data
        for i in range(15):
            t_test_i, W_test_i = model.fetch_minibatch()
            X_pred_i, Y_pred_i = model.predict(self.Xi, t_test_i, W_test_i)

            if hasattr(t_test, "cpu"):
                t_test = torch.cat([t_test, t_test_i], dim=0)
                X_pred = torch.cat([X_pred, X_pred_i], dim=0)
                Y_pred = torch.cat([Y_pred, Y_pred_i], dim=0)
            else:
                t_test = np.concatenate([t_test, t_test_i], axis=0)
                X_pred = np.concatenate([X_pred, X_pred_i], axis=0)
                Y_pred = np.concatenate([Y_pred, Y_pred_i], axis=0)

        # Convert to numpy
        if hasattr(t_test, "cpu"):
            t_test = t_test.cpu().numpy()[:n_samples]
            X_pred = X_pred.cpu().detach().numpy()[:n_samples]
            Y_pred = Y_pred.cpu().detach().numpy()[:n_samples]

        # Calculate analytical solution
        Y_analytical = u_exact(t_test, X_pred, self.T)

        # Calculate error statistics
        errors = np.abs(Y_pred[:, -1, 0] - Y_analytical[:, -1, 0])
        relative_errors = errors / (np.abs(Y_analytical[:, -1, 0]) + 1e-8)

        rmse = np.sqrt(np.mean(errors**2))
        mean_relative_error = np.mean(relative_errors)
        std_relative_error = np.std(relative_errors)

        print(f"RMSE: {rmse:.6f}")
        print(f"Mean relative error: {mean_relative_error:.6f}")
        print(f"Relative error standard deviation: {std_relative_error:.6f}")
        print(f"Maximum relative error: {np.max(relative_errors):.6f}")
        print(f"Minimum relative error: {np.min(relative_errors):.6f}")

        return {
            "rmse": rmse,
            "mean_relative_error": mean_relative_error,
            "std_relative_error": std_relative_error,
            "t_test": t_test,
            "X_pred": X_pred,
            "Y_pred": Y_pred,
            "Y_analytical": Y_analytical,
        }

    # Method to visualize optimization results and model performance
    def visualize_results(self, save_plots=True):
        """Visualize optimization results and model performance"""
        if self.study is None:
            print("Please run optimization first!")
            return

        # Ensure the main directory exists
        os.makedirs("optuna_outcomes", exist_ok=True)

        # Create results directory (renamed to imgs)
        results_dir = "optuna_outcomes/imgs"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Configure Plotly for Chinese support
            import plotly.io as pio
            import plotly.graph_objects as go

            # Use Arial Unicode MS as Chinese font
            pio.templates.default = "plotly_white"
            go.Layout(font=dict(family="Arial Unicode MS, Heiti TC, Microsoft YaHei"))

            # Get the number of completed trials
            completed_trials = [
                t for t in self.study.trials if t.state == TrialState.COMPLETE
            ]

            # 1. Optimization history
            fig1 = optuna.visualization.plot_optimization_history(self.study)

            # If there is only one trial, adjust the x-axis range to make the data point visible
            if len(completed_trials) == 1:
                fig1.update_layout(xaxis=dict(range=[-0.5, 0.5]))  # Set appropriate x-axis range

            if save_plots:
                fig1.write_image(f"{results_dir}/optimization_history_{timestamp}.png")
            fig1.show()

            # 2. Hyperparameter importance
            fig2 = optuna.visualization.plot_param_importances(self.study)
            if save_plots:
                fig2.write_image(f"{results_dir}/param_importance_{timestamp}.png")
            fig2.show()

            # 3. Slice plot
            fig3 = optuna.visualization.plot_slice(self.study)
            if save_plots:
                fig3.write_image(f"{results_dir}/slice_plot_{timestamp}.png")
            fig3.show()

            # 4. Parallel coordinate plot
            fig4 = optuna.visualization.plot_parallel_coordinate(self.study)
            if save_plots:
                fig4.write_image(f"{results_dir}/parallel_coord_{timestamp}.png")
            fig4.show()

        except Exception as e:
            print(f"Visualization error: {e}")
            # Use matplotlib to create basic visualizations
            self._create_basic_visualizations(results_dir, timestamp)

    # Method to create basic visualizations using matplotlib (fallback)
    def _create_basic_visualizations(self, results_dir, timestamp):
        """Create basic visualizations using matplotlib (backup)"""
        # Optimization history
        plt.figure(figsize=(10, 6))
        trials = [t for t in self.study.trials if t.state == TrialState.COMPLETE]
        values = [t.value for t in trials]
        numbers = [t.number for t in trials]

        plt.plot(numbers, values, "b-", alpha=0.7)
        plt.xlabel("Trial number")
        plt.ylabel("Loss value")
        plt.title("Optimization history")
        plt.grid(True, alpha=0.3)
        plt.savefig(
            f"{results_dir}/optimization_history_{timestamp}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Model performance visualization (if best model exists)
        if self.best_model is not None:
            self._plot_model_performance(results_dir, timestamp)

    # Method to plot model performance
    def _plot_model_performance(self, results_dir, timestamp):
        """Plot model performance"""
        # Evaluate the model
        results = self.evaluate_model(n_samples=500)

        # Create comprehensive performance plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Predicted vs true value scatter plot
        ax1 = axes[0, 0]
        y_true = results["Y_analytical"][:, -1, 0]
        y_pred = results["Y_pred"][:, -1, 0]
        ax1.scatter(y_true, y_pred, alpha=0.6)
        ax1.plot(
            [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2
        )
        ax1.set_xlabel("Analytical solution")
        ax1.set_ylabel("Predicted solution")
        ax1.set_title("Predicted vs True")
        ax1.grid(True, alpha=0.3)

        # 2. Error distribution
        ax2 = axes[0, 1]
        errors = np.abs(y_true - y_pred)
        ax2.hist(errors, bins=50, alpha=0.7, color="red")
        ax2.set_xlabel("Absolute error")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Error distribution")
        ax2.grid(True, alpha=0.3)

        # 3. Time series comparison (first 5 samples)
        ax3 = axes[1, 0]
        n_samples_show = min(5, len(results["t_test"]))
        for i in range(n_samples_show):
            t = results["t_test"][i, :, 0]
            y_pred_sample = results["Y_pred"][i, :, 0]
            y_true_sample = results["Y_analytical"][i, :, 0]
            ax3.plot(t, y_pred_sample, "b-", alpha=0.7, label="Predicted" if i == 0 else "")
            ax3.plot(
                t, y_true_sample, "r--", alpha=0.7, label="Analytical" if i == 0 else ""
            )
        ax3.set_xlabel("Time t")
        ax3.set_ylabel("Solution Y(t)")
        ax3.set_title("Time series comparison")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Relative error statistics
        ax4 = axes[1, 1]
        relative_errors = np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8)
        ax4.boxplot(relative_errors)
        ax4.set_ylabel("Relative error")
        ax4.set_title("Relative error statistics")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{results_dir}/model_performance_{timestamp}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    # Method to save the Optuna study
    def save_study(self, filename="bsb_optuna_study.pkl"):
        """Save the study results"""
        if self.study is None:
            print("No study results to save!")
            return

        import joblib

        # Create directories
        # Ensure the main directory exists
        os.makedirs("optuna_outcomes", exist_ok=True)
        os.makedirs("optuna_outcomes/studies", exist_ok=True)
        filepath = f"optuna_outcomes/studies/{filename}"
        joblib.dump(self.study, filepath)
        print(f"Study saved to: {filepath}")

    # Method to load a saved Optuna study
    def load_study(self, filename="bsb_optuna_study.pkl"):
        """Load the study results"""
        import joblib

        filepath = f"optuna_outcomes/studies/{filename}"
        self.study = joblib.load(filepath)
        self.best_trial = self.study.best_trial
        print(f"Study loaded from {filepath}")

    # Method to save the best model
    def save_best_model(self, filename=None):
        """Save the best model"""
        if self.best_model is None:
            print("No best model to save!")
            return

        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bsb_best_model_{timestamp}.pth"

        # Ensure the main directory exists
        os.makedirs("optuna_outcomes", exist_ok=True)

        # Create model directory (renamed to models)
        os.makedirs("optuna_outcomes/models", exist_ok=True)
        filepath = f"optuna_outcomes/models/{filename}"

        # Save model state and hyperparameters
        save_dict = {
            "model_state_dict": self.best_model.model.state_dict(),
            "best_params": self.best_trial.params,
            "training_loss": getattr(self.best_model, "training_loss", []),
            "iteration": getattr(self.best_model, "iteration", []),
            "Xi": self.Xi,
            "T": self.T,
            "D": self.D,
        }

        torch.save(save_dict, filepath)
        print(f"Best model saved to: {filepath}")

    # Method to generate an optimization report
    def generate_report(self):
        """Generate an optimization report"""
        if self.study is None:
            print("Please run optimization first!")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Ensure the main directory exists
        os.makedirs("optuna_outcomes", exist_ok=True)
        report_dir = "optuna_outcomes/reports"
        os.makedirs(report_dir, exist_ok=True)

        report_file = f"{report_dir}/bsb_optuna_report_{timestamp}.txt"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("           Black-Scholes-Barenblatt Optuna Optimization Report\n")
            f.write("=" * 80 + "\n\n")

            f.write(
                f"Report generation time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Problem dimension: {self.D}D\n")
            f.write(f"Time interval: [0, {self.T}]\n\n")

            f.write("Best trial results:\n")
            f.write(f"  Trial number: #{self.best_trial.number}\n")
            f.write(f"  Final loss: {self.best_trial.value:.6f}\n")
            f.write(f"  Relative error: {self.best_trial.user_attrs['relative_error']:.6f}\n")
            f.write(
                f"  Training time: {self.best_trial.user_attrs['training_time']:.2f} seconds\n\n"
            )

            f.write("Best hyperparameter configuration:\n")
            for key, value in self.best_trial.params.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"  Network structure: {self.best_trial.user_attrs['layers']}\n\n")

            f.write("Trial statistics:\n")
            completed_trials = [
                t for t in self.study.trials if t.state == TrialState.COMPLETE
            ]
            failed_trials = [t for t in self.study.trials if t.state == TrialState.FAIL]
            f.write(f"  Completed trials: {len(completed_trials)}\n")
            f.write(f"  Failed trials: {len(failed_trials)}\n")
            f.write(f"  Total trials: {len(self.study.trials)}\n")

            if completed_trials:
                errors = [
                    t.user_attrs.get("relative_error", float("inf"))
                    for t in completed_trials
                ]
                times = [t.user_attrs.get("training_time", 0) for t in completed_trials]
                f.write(f"  Average error: {np.mean(errors):.6f}\n")
                f.write(f"  Best error: {np.min(errors):.6f}\n")
                f.write(f"  Worst error: {np.max(errors):.6f}\n")
                f.write(f"  Average time: {np.mean(times):.2f} seconds\n")

        print(f"Report generated: {report_file}")


# Main execution block
if __name__ == "__main__":

    # Set random seed for reproducibility
    set_seed(42)
    """Original vanilla functionality, integrated with Optuna optimization"""
    print("=" * 80)
    print("           Black-Scholes-Barenblatt Equation Solving and Optuna Optimization")
    print("=" * 80)

    # Directly run Optuna optimization (100 trials)
    print("Directly running Optuna optimization (100 trials)...")
    """Main function: run Optuna optimization for Black-Scholes-Barenblatt"""
    # Set fixed parameters
    D = 100  # Dimension
    Xi = np.array([1.0, 0.5] * (D // 2))[None, :]  # Initial condition
    T = 1.0  # Time interval

    print("Black-Scholes-Barenblatt equation Optuna hyperparameter optimization")
    print("=" * 60)

    # Create optimizer
    optimizer = BlackScholesBarenblattOptunaOptimizer(Xi, T, D)

    # Directly run optimization (without loading existing study, run 100 new trials)
    print("Directly running 100 new trials...")
    best_trial = optimizer.optimize(
        n_trials=100,  # Number of trials (set to 100)
        timeout=3600,  # 1 hour timeout
        study_name=f"BSB_{D}D_Optuna",
    )

    # Train the best model
    best_model = optimizer.train_best_model(save_model=True)

    # Evaluate the model
    evaluation_results = optimizer.evaluate_model()

    # Visualize results
    optimizer.visualize_results(save_plots=True)

    # Generate report
    optimizer.generate_report()

    # Save the study
    optimizer.save_study()

    print("\n" + "=" * 60)
    print("Optimization completed!")
    print("=" * 60)