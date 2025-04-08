"""
Data loaders for Sudden Stratospheric Warming (SSW) datasets.
"""
import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Callable, Tuple, Union


class SSWConditionalDataLoader:
    """
    Loads a Sudden Stratospheric Warming (SSW) dataset and provides conditional data pairs.
    
    Each call to __next__ returns:
    - x_cond: Conditioning state sequence (batch_size, condition_steps*2, spatial_dim)
    - x_target: Target state (batch_size, 2, spatial_dim)
    """
    def __init__(self, data_file: str, batch_size: int, condition_steps: int = 2, 
                 timesteps: Optional[int] = None, dt: int = 1, normalize: bool = False):
        """
        Args:
            data_file: Path to the numpy file containing SSW data
            batch_size: Number of samples per batch
            condition_steps: Number of historical steps to use for conditioning
            timesteps: Maximum number of timesteps to use from the data
            dt: Time step size
            normalize: Whether to normalize the data
        """
        # Load data from file
        data = np.load(data_file)
        data = data.astype(np.float32)
        
        # Keep only up to 'timesteps' timepoints if specified
        if timesteps is not None and timesteps < data.shape[0]:
            data = data[:timesteps]
            print(f"Using first {timesteps} timesteps from dataset")
        else:
            print(f"Using all {data.shape[0]} timesteps from dataset")
            
        # Optionally normalize
        if normalize:
            self.mean = data.mean()
            self.std = data.std()
            data = (data - self.mean) / self.std
            print(f"Normalized data: mean={self.mean:.4f}, std={self.std:.4f}")
            
        # Store data and parameters
        self.data = data
        self.batch_size = batch_size
        self.condition_steps = condition_steps
        self.dt = dt
        self.spatial_dim = data.shape[1]
        
        # Need at least condition_steps*dt historical points and 1 future point
        self.min_index = condition_steps * dt
        self.max_index = data.shape[0] - dt - 1
        
        if self.min_index > self.max_index:
            raise ValueError(f"Not enough data points for {condition_steps=} and {dt=}")
            
        print(f"Using indices from {self.min_index} to {self.max_index}")
        print(f"Condition steps: {condition_steps}, dt: {dt}")
        print(f"Spatial dimension: {self.spatial_dim}")
        
    def __iter__(self):
        return self
        
    def __next__(self):
        # Pick random indices in valid range
        idx = np.random.randint(self.min_index, self.max_index + 1, self.batch_size)
        
        # Initialize arrays for the batch
        # Each conditioning state has 2 components (position and velocity)
        x_cond = np.zeros((self.batch_size, self.condition_steps * 2, self.spatial_dim), dtype=np.float32)
        # Target also has 2 components
        x_target = np.zeros((self.batch_size, 2, self.spatial_dim), dtype=np.float32)
        
        # For each sample in batch
        for b in range(self.batch_size):
            current_idx = idx[b]
            
            # Build conditioning sequence
            for c in range(self.condition_steps):
                history_idx = current_idx - (self.condition_steps - c) * self.dt
                
                # Position
                x_cond[b, c*2] = self.data[history_idx]
                # Velocity (approximated by difference)
                x_cond[b, c*2+1] = self.data[history_idx + 1] - self.data[history_idx]
            
            # Set target state
            # Position
            x_target[b, 0] = self.data[current_idx]
            # Velocity
            x_target[b, 1] = self.data[current_idx + self.dt] - self.data[current_idx]
        
        # Move to device
        return jax.device_put(x_cond), jax.device_put(x_target)
    
    def get_normalized_params(self):
        """Return normalization parameters if normalize=True was used"""
        return getattr(self, 'mean', 0.0), getattr(self, 'std', 1.0)