"""
Forward-Backward Stochastic Neural Networks (FBSNNs) Enhanced Version
with improved error handling, logging, and batch processing capabilities.

This module provides an enhanced implementation of FBSNNs for solving
forward-backward stochastic differential equations (FBSDEs) using neural networks.

Author: PINN-solve-PDE Project
Date: 2025-12-19
"""

import logging
import warnings
from typing import Optional, Tuple, List, Dict, Any, Callable
from dataclasses import dataclass
from functools import wraps
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('FBSNNs_Enhanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def log_execution(func: Callable) -> Callable:
    """Decorator to log function execution time and parameters."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Executing: {func.__name__}")
        logger.debug(f"Args: {args}, Kwargs: {kwargs}")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Completed: {func.__name__} in {execution_time:.4f}s")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper


def validate_input(func: Callable) -> Callable:
    """Decorator to validate input parameters."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ValueError, TypeError) as e:
            logger.error(f"Input validation error in {func.__name__}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise
    return wrapper


@dataclass
class FBSDEConfig:
    """Configuration dataclass for FBSDE solver parameters."""
    T: float = 1.0
    num_time_steps: int = 100
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 0.001
    num_hidden_layers: int = 3
    hidden_units: int = 128
    dropout_rate: float = 0.2
    validation_split: float = 0.2
    verbose: int = 1
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.T <= 0:
            raise ValueError("T must be positive")
        if self.num_time_steps <= 0:
            raise ValueError("num_time_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")
        logger.info("Configuration validation passed")


class BatchProcessor:
    """Handles batch processing of training data with error checking."""
    
    def __init__(self, batch_size: int, shuffle: bool = True):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Size of each batch
            shuffle: Whether to shuffle data
            
        Raises:
            ValueError: If batch_size is invalid
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.shuffle = shuffle
        logger.info(f"BatchProcessor initialized with batch_size={batch_size}, shuffle={shuffle}")
    
    def create_batches(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Create batches from data.
        
        Args:
            data: Input data array
            
        Returns:
            List of batch arrays
            
        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("data must be a numpy array")
        if data.shape[0] == 0:
            raise ValueError("data cannot be empty")
        
        if self.shuffle:
            indices = np.random.permutation(data.shape[0])
            data = data[indices]
        
        batches = [
            data[i:i+self.batch_size]
            for i in range(0, data.shape[0], self.batch_size)
        ]
        logger.debug(f"Created {len(batches)} batches from data with shape {data.shape}")
        return batches
    
    def create_multiple_batches(self, *arrays: np.ndarray) -> List[Tuple[np.ndarray, ...]]:
        """
        Create synchronized batches from multiple arrays.
        
        Args:
            *arrays: Variable number of input arrays
            
        Returns:
            List of tuples of batch arrays
            
        Raises:
            ValueError: If arrays have inconsistent first dimension
        """
        if not arrays:
            raise ValueError("At least one array must be provided")
        
        first_len = arrays[0].shape[0]
        for arr in arrays[1:]:
            if arr.shape[0] != first_len:
                raise ValueError("All arrays must have the same first dimension")
        
        if self.shuffle:
            indices = np.random.permutation(first_len)
            arrays = tuple(arr[indices] for arr in arrays)
        
        batches = [
            tuple(arr[i:i+self.batch_size] for arr in arrays)
            for i in range(0, first_len, self.batch_size)
        ]
        logger.debug(f"Created {len(batches)} synchronized batches")
        return batches


class NetworkBuilder:
    """Builds neural networks with various architectures."""
    
    @staticmethod
    @log_execution
    @validate_input
    def build_dense_network(
        input_dim: int,
        output_dim: int,
        config: FBSDEConfig
    ) -> keras.Model:
        """
        Build a dense neural network.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            config: FBSDE configuration
            
        Returns:
            Compiled keras model
            
        Raises:
            ValueError: If dimensions are invalid
        """
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Dimensions must be positive")
        
        logger.info(f"Building dense network: {input_dim} -> {output_dim}")
        
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        
        # Hidden layers
        for i in range(config.num_hidden_layers):
            model.add(layers.Dense(
                config.hidden_units,
                activation='relu',
                name=f'dense_{i+1}'
            ))
            if config.dropout_rate > 0:
                model.add(layers.Dropout(config.dropout_rate))
        
        # Output layer
        model.add(layers.Dense(output_dim, name='output'))
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss='mse'
        )
        
        logger.info(f"Network built successfully")
        return model
    
    @staticmethod
    @log_execution
    @validate_input
    def build_residual_network(
        input_dim: int,
        output_dim: int,
        config: FBSDEConfig
    ) -> keras.Model:
        """
        Build a residual neural network.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            config: FBSDE configuration
            
        Returns:
            Compiled keras model
        """
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Dimensions must be positive")
        
        logger.info(f"Building residual network: {input_dim} -> {output_dim}")
        
        inputs = keras.Input(shape=(input_dim,))
        x = layers.Dense(config.hidden_units, activation='relu')(inputs)
        
        # Residual blocks
        for i in range(config.num_hidden_layers):
            residual = x
            x = layers.Dense(config.hidden_units, activation='relu')(x)
            if config.dropout_rate > 0:
                x = layers.Dropout(config.dropout_rate)(x)
            x = layers.Dense(config.hidden_units)(x)
            if x.shape[-1] == residual.shape[-1]:
                x = layers.Add()([x, residual])
            x = layers.Activation('relu')(x)
        
        outputs = layers.Dense(output_dim)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss='mse'
        )
        
        logger.info("Residual network built successfully")
        return model


class FBSNNSolver:
    """Enhanced FBSNN solver with improved error handling and logging."""
    
    def __init__(self, config: FBSDEConfig):
        """
        Initialize FBSNN solver.
        
        Args:
            config: FBSDE configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        config.validate()
        self.config = config
        self.history = {}
        logger.info("FBSNNSolver initialized")
    
    @log_execution
    @validate_input
    def generate_training_data(
        self,
        sde_func: Callable,
        num_samples: int = 1000,
        initial_condition: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data for the FBSDE.
        
        Args:
            sde_func: Function that generates SDE paths
            num_samples: Number of sample paths
            initial_condition: Initial condition for the SDE
            
        Returns:
            Tuple of (paths, time_steps)
            
        Raises:
            ValueError: If parameters are invalid
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        
        logger.info(f"Generating {num_samples} training samples")
        
        try:
            dt = self.config.T / self.config.num_time_steps
            time_steps = np.linspace(0, self.config.T, self.config.num_time_steps + 1)
            
            paths = sde_func(
                num_samples=num_samples,
                num_time_steps=self.config.num_time_steps,
                dt=dt,
                initial_condition=initial_condition
            )
            
            if paths.shape[0] != num_samples:
                raise ValueError(f"Expected {num_samples} paths, got {paths.shape[0]}")
            
            logger.info(f"Generated data shape: {paths.shape}")
            return paths, time_steps
            
        except Exception as e:
            logger.error(f"Error generating training data: {str(e)}")
            raise
    
    @log_execution
    @validate_input
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        network_type: str = 'dense'
    ) -> keras.Model:
        """
        Train the FBSNN model.
        
        Args:
            X_train: Training input data
            y_train: Training target data
            X_val: Validation input data
            y_val: Validation target data
            network_type: Type of network ('dense' or 'residual')
            
        Returns:
            Trained keras model
            
        Raises:
            ValueError: If data is invalid
        """
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must have same number of samples")
        
        logger.info(f"Starting training with network_type={network_type}")
        logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        # Build network
        if network_type == 'dense':
            model = NetworkBuilder.build_dense_network(
                X_train.shape[1],
                y_train.shape[1],
                self.config
            )
        elif network_type == 'residual':
            model = NetworkBuilder.build_residual_network(
                X_train.shape[1],
                y_train.shape[1],
                self.config
            )
        else:
            raise ValueError(f"Unknown network_type: {network_type}")
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            if X_val.shape[0] != y_val.shape[0]:
                raise ValueError("X_val and y_val must have same number of samples")
            validation_data = (X_val, y_val)
            logger.info(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")
        
        # Train
        try:
            history = model.fit(
                X_train, y_train,
                batch_size=self.config.batch_size,
                epochs=self.config.num_epochs,
                validation_data=validation_data,
                verbose=self.config.verbose
            )
            
            self.history = history.history
            logger.info("Training completed successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    @log_execution
    @validate_input
    def batch_train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        network_type: str = 'dense'
    ) -> keras.Model:
        """
        Train using batch processing for large datasets.
        
        Args:
            X_train: Training input data
            y_train: Training target data
            X_val: Validation input data
            y_val: Validation target data
            network_type: Type of network ('dense' or 'residual')
            
        Returns:
            Trained keras model
        """
        logger.info("Starting batch training")
        
        # Build network
        if network_type == 'dense':
            model = NetworkBuilder.build_dense_network(
                X_train.shape[1],
                y_train.shape[1],
                self.config
            )
        elif network_type == 'residual':
            model = NetworkBuilder.build_residual_network(
                X_train.shape[1],
                y_train.shape[1],
                self.config
            )
        else:
            raise ValueError(f"Unknown network_type: {network_type}")
        
        processor = BatchProcessor(self.config.batch_size, shuffle=True)
        
        # Training history
        train_losses = []
        val_losses = []
        
        try:
            for epoch in range(self.config.num_epochs):
                epoch_losses = []
                
                # Create batches
                batches = processor.create_multiple_batches(X_train, y_train)
                
                for batch_X, batch_y in batches:
                    loss = model.train_on_batch(batch_X, batch_y)
                    epoch_losses.append(loss)
                
                avg_train_loss = np.mean(epoch_losses)
                train_losses.append(avg_train_loss)
                
                # Validation
                if X_val is not None and y_val is not None:
                    val_loss = model.evaluate(X_val, y_val, verbose=0)
                    val_losses.append(val_loss)
                    
                    if (epoch + 1) % max(1, self.config.num_epochs // 10) == 0:
                        logger.info(
                            f"Epoch {epoch+1}/{self.config.num_epochs} - "
                            f"Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}"
                        )
                else:
                    if (epoch + 1) % max(1, self.config.num_epochs // 10) == 0:
                        logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - Loss: {avg_train_loss:.6f}")
            
            self.history = {
                'loss': train_losses,
                'val_loss': val_losses if val_losses else None
            }
            logger.info("Batch training completed successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error during batch training: {str(e)}")
            raise
    
    @log_execution
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history.
        
        Args:
            save_path: Path to save the figure
        """
        if not self.history or 'loss' not in self.history:
            logger.warning("No training history available")
            return
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(self.history['loss'], label='Training Loss')
            if self.history.get('val_loss'):
                ax.plot(self.history['val_loss'], label='Validation Loss')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training History')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save_path:
                fig.savefig(save_path)
                logger.info(f"Training history plot saved to {save_path}")
            
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error plotting training history: {str(e)}")
            raise


class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {}
        logger.info("PerformanceMonitor initialized")
    
    def record_metric(self, name: str, value: float) -> None:
        """
        Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
        """
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        logger.debug(f"Recorded metric {name}: {value:.6f}")
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all recorded metrics.
        
        Returns:
            Dictionary of metric summaries
        """
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        logger.info(f"Metrics summary: {summary}")
        return summary


# Example usage and demonstration
if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("FBSNNs Enhanced Module Started")
    logger.info("=" * 50)
    
    # Create configuration
    config = FBSDEConfig(
        T=1.0,
        num_time_steps=100,
        batch_size=32,
        num_epochs=10,
        learning_rate=0.001,
        num_hidden_layers=2,
        hidden_units=64
    )
    
    # Create solver
    solver = FBSNNSolver(config)
    
    # Simple SDE function for demonstration
    def simple_sde(num_samples, num_time_steps, dt, initial_condition=None):
        """Simple geometric Brownian motion."""
        if initial_condition is None:
            initial_condition = np.ones(1)
        
        paths = np.zeros((num_samples, num_time_steps + 1))
        paths[:, 0] = initial_condition[0]
        
        for t in range(num_time_steps):
            dW = np.random.normal(0, np.sqrt(dt), num_samples)
            paths[:, t+1] = paths[:, t] * np.exp((0.1 - 0.5 * 0.2**2) * dt + 0.2 * dW)
        
        return paths
    
    # Generate data
    X_data, time_steps = solver.generate_training_data(simple_sde, num_samples=500)
    
    logger.info("=" * 50)
    logger.info("FBSNNs Enhanced Module Demo Completed")
    logger.info("=" * 50)
