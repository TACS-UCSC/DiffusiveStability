import pickle
import numpy as np
import jax
import jax.numpy as jnp

class ksDataLoader:
    """
    Loads a dataset of shape [1024, large_num_timesteps].
    Truncates it to use only 'timesteps' columns for training.

    Each call to __next__ returns (x_t, x_tp) of shape (batch_size, 1024),
    where x_tp is the data at the same spatial dimension but dt steps in the future.
    """
    def __init__(self, pickle_file, batch_size, timesteps=50000, dt=1, normalize=False):
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)  # shape [1024, some_big_T]
        data = data.astype(np.float32)

        # Keep only up to 'timesteps' columns
        if timesteps > data.shape[1]:
            raise ValueError(f"Requested {timesteps=} exceeds available {data.shape[1]=}")
        data = data[:, :timesteps]

        # Optionally normalize
        if normalize:
            self.mean = data.mean()
            self.std = data.std()
            data = (data - self.mean) / self.std

        self.data = data
        self.batch_size = batch_size
        self.dt = dt
        # The largest index we can pick is timesteps - dt - 1
        self.max_index = data.shape[1] - dt
        if self.max_index < 1:
            raise ValueError("Not enough data points for the chosen dt.")

    def __iter__(self):
        return self

    def __next__(self):
        # Pick random indices in [0, max_index]
        idx = np.random.randint(0, self.max_index, self.batch_size)
        # x_t: shape [1024, batch_size]
        x_t = self.data[:, idx]
        # x_tp: shape [1024, batch_size]
        x_tp = self.data[:, idx + self.dt]

        # Transpose to (batch_size, 1024) and move to device
        return jax.device_put(x_t.T), jax.device_put(x_tp.T)

    def get_normalized_params(self):
        """Return normalization parameters if normalize=True was used"""
        return getattr(self, 'mean', 0.0), getattr(self, 'std', 1.0)


class ksDeltaDataLoader:
    """
    Loads a dataset of shape [1024, large_num_timesteps] and creates delta pairs.
    Each call to __next__ returns (x_t, delta_x) of shape (batch_size, 1024),
    where delta_x = (x_t+dt - x_t) * scale_factor.
    
    The scaling factor is applied to make small deltas more learnable by the neural network.
    """
    def __init__(self, pickle_file, batch_size, timesteps=50000, dt=1, scale_factor=1000.0):
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)  # shape [1024, some_big_T]
        data = data.astype(np.float32)
        
        # Keep only up to 'timesteps' columns
        if timesteps > data.shape[1]:
            raise ValueError(f"Requested {timesteps=} exceeds available {data.shape[1]=}")
        data = data[:, :timesteps]
        
        # Store original data and parameters
        self.data = data
        self.batch_size = batch_size
        self.dt = dt
        self.scale_factor = scale_factor
        
        # The largest index we can pick is timesteps - dt - 1
        self.max_index = data.shape[1] - dt
        if self.max_index < 1:
            raise ValueError("Not enough data points for the chosen dt.")
        
        # Calculate statistics of the deltas for reference (but don't apply normalization)
        sample_indices = np.random.randint(0, self.max_index, min(1000, self.max_index))
        sample_deltas = np.array([self.data[:, i + self.dt] - self.data[:, i] for i in sample_indices])
        self.delta_mean = float(sample_deltas.mean())
        self.delta_std = float(sample_deltas.std())
        self.delta_min = float(sample_deltas.min())
        self.delta_max = float(sample_deltas.max())
        
        print(f"Delta statistics before scaling:")
        print(f"  Mean: {self.delta_mean:.8f}")
        print(f"  Std:  {self.delta_std:.8f}")
        print(f"  Min:  {self.delta_min:.8f}")
        print(f"  Max:  {self.delta_max:.8f}")
        print(f"Applying scale factor: {self.scale_factor}")
        
        # Calculate statistics after scaling
        self.scaled_delta_mean = self.delta_mean * self.scale_factor
        self.scaled_delta_std = self.delta_std * self.scale_factor
        self.scaled_delta_min = self.delta_min * self.scale_factor
        self.scaled_delta_max = self.delta_max * self.scale_factor
        
        print(f"Delta statistics after scaling:")
        print(f"  Mean: {self.scaled_delta_mean:.4f}")
        print(f"  Std:  {self.scaled_delta_std:.4f}")
        print(f"  Min:  {self.scaled_delta_min:.4f}")
        print(f"  Max:  {self.scaled_delta_max:.4f}")
        
    def __iter__(self):
        return self
        
    def __next__(self):
        # Pick random indices in [0, max_index]
        idx = np.random.randint(0, self.max_index, self.batch_size)
        
        # Get states at time t
        x_t = self.data[:, idx]
        
        # Get states at time t+dt
        x_tp = self.data[:, idx + self.dt]
        
        # Calculate deltas and apply scaling
        delta_x = (x_tp - x_t) * self.scale_factor
        
        # Transpose to (batch_size, 1024) and move to device
        return jax.device_put(x_t.T), jax.device_put(delta_x.T)
        
    def get_scale_factor(self):
        """Return the scale factor used for deltas"""
        return self.scale_factor
    
    def unscale_delta(self, scaled_delta):
        """Convert a scaled delta back to its original scale"""
        return scaled_delta / self.scale_factor
        
class ksConditionalDataLoader:
    """
    Loads a dataset of shape [1024, large_num_timesteps].
    Each call to __next__ returns:
    - x_t(cond): A sequence of historical states with shape (condition_steps, 1024)
    - x_tp: The next state after the most recent historical state
    
    Historical states are arranged in reverse chronological order:
    x_t(cond)[0] is the most recent state, x_t(cond)[1] is one timestep before, etc.
    
    For training, can optionally add noise to conditioning data to simulate autoregressive prediction.
    """
    def __init__(self, pickle_file, batch_size, condition_steps=2, timesteps=50000, dt=1, 
                 normalize=False, condition_noise=0.0, condition_noise_schedule=None, 
                 start_sample_index=0):
        """
        Args:
            pickle_file: Path to the pickle file containing KS data
            batch_size: Number of samples per batch
            condition_steps: Number of historical steps to use for conditioning
            timesteps: Maximum number of timesteps to use from the data
            dt: Time step increment
            normalize: Whether to normalize the data
            condition_noise: Standard deviation of Gaussian noise to add to conditioning data (default: 0.0)
            condition_noise_schedule: Optional function mapping condition step index to noise level
                                     (overrides condition_noise if provided)
            start_sample_index: Index of first sample to use (default: 0) - discards earlier samples
        """
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)  # shape [1024, some_big_T]
        data = data.astype(np.float32)

        # Apply start_sample_index to discard initial samples
        if start_sample_index > 0:
            if start_sample_index >= data.shape[1]:
                raise ValueError(f"start_sample_index ({start_sample_index}) exceeds data length ({data.shape[1]})")
            
            original_length = data.shape[1]
            data = data[:, start_sample_index:]
            print(f"Discarding first {start_sample_index} samples. "
                  f"Data shape changed from [1024, {original_length}] to [1024, {data.shape[1]}]")

        # Keep only up to 'timesteps' columns after applying start_sample_index
        if timesteps > data.shape[1]:
            print(f"Warning: Requested {timesteps=} exceeds available {data.shape[1]=} after applying start_sample_index")
            print(f"Using all available {data.shape[1]} timesteps")
        else:
            data = data[:, :timesteps]
            print(f"Using {timesteps} timesteps after start_sample_index")

        # Optionally normalize
        if normalize:
            self.mean = data.mean()
            self.std = data.std()
            data = (data - self.mean) / self.std

        self.data = data
        self.batch_size = batch_size
        self.condition_steps = condition_steps
        self.dt = dt
        self.condition_noise = condition_noise
        self.condition_noise_schedule = condition_noise_schedule
        self.start_sample_index = start_sample_index
        
        # Need at least condition_steps*dt historical points and 1 future point
        self.min_index = (condition_steps - 1) * dt
        self.max_index = data.shape[1] - dt - 1
        
        if self.min_index > self.max_index:
            raise ValueError(f"Not enough data points for {condition_steps=} and {dt=}.")
            
        print(f"Using indices from {self.min_index} to {self.max_index}")
        print(f"Condition steps: {condition_steps}, dt: {dt}")
        
        if condition_noise > 0:
            print(f"Adding Gaussian noise with std={condition_noise} to conditioning data during training")
        if condition_noise_schedule is not None:
            print(f"Using custom noise schedule for conditioning data during training")

    def set_start_index(self, start_index):
        """
        Set the starting index for data extraction.
        
        Args:
            start_index: The index to start data extraction from
        """
        # Make sure index is within bounds
        if start_index < self.min_index:
            print(f"Warning: Requested start_index {start_index} is less than minimum allowed index {self.min_index}.")
            start_index = self.min_index
        
        if start_index > self.max_index:
            print(f"Warning: Requested start_index {start_index} exceeds maximum allowed index {self.max_index}.")
            start_index = self.max_index
        
        # Store the index for later use
        self.current_index = start_index
        print(f"Data loader start index set to {self.current_index}")
        
        # Return the object to allow method chaining
        return self

    def __iter__(self):
        return self

    def __next__(self):
        # Determine indices to use
        if hasattr(self, 'current_index'):
            # Use the stored index for the first element, random for others
            if self.batch_size == 1:
                idx = np.array([self.current_index])
            else:
                # For batch size > 1, use the set index for the first item,
                # random indices for the rest
                random_idx = np.random.randint(self.min_index, self.max_index + 1, self.batch_size - 1)
                idx = np.concatenate([[self.current_index], random_idx])
        else:
            # Original behavior - pick random indices
            idx = np.random.randint(self.min_index, self.max_index + 1, self.batch_size)
            self.current_index = idx[0]  # Store for reference
        
        # Initialize arrays for batch
        x_cond_batch = np.zeros((self.batch_size, self.condition_steps, self.data.shape[0]), dtype=np.float32)
        x_tp_batch = np.zeros((self.batch_size, self.data.shape[0]), dtype=np.float32)
        
        # For each sample in batch
        for b in range(self.batch_size):
            # Current index for this batch item
            current_idx = idx[b]
            
            # Get the condition sequence (in reverse chronological order)
            for c in range(self.condition_steps):
                history_idx = current_idx - c * self.dt
                x_cond_batch[b, c] = self.data[:, history_idx]
            
            # Get the future state
            x_tp_batch[b] = self.data[:, current_idx + self.dt]
        
        # Add noise to conditioning data if requested
        if self.condition_noise > 0 or self.condition_noise_schedule is not None:
            for c in range(self.condition_steps):
                # Determine noise level for this condition step
                if self.condition_noise_schedule is not None:
                    # Use custom noise schedule
                    noise_level = self.condition_noise_schedule(c)
                else:
                    # Use fixed noise level
                    noise_level = self.condition_noise
                
                # Generate and add noise
                if noise_level > 0:
                    noise = np.random.normal(0, noise_level, x_cond_batch[:, c, :].shape)
                    x_cond_batch[:, c, :] += noise
        
        # Transpose to put spatial dimension last and move to device
        # Result: x_cond shape: (batch_size, condition_steps, 1024)
        #         x_tp shape: (batch_size, 1024)
        return jax.device_put(x_cond_batch), jax.device_put(x_tp_batch)
    
    def get_normalized_params(self):
        """Return normalization parameters if normalize=True was used"""
        return getattr(self, 'mean', 0.0), getattr(self, 'std', 1.0)

class ksConditionalDeltaDataLoader:
    """
    Combines conditional history with delta prediction for Kuramoto-Sivashinsky equation.
    
    Returns:
    - x_cond: A sequence of historical states with shape (condition_steps, 1024)
    - delta_x: The scaled difference between future state and most recent state
    """
    def __init__(self, pickle_file, batch_size, condition_steps=2, timesteps=50000, dt=1, 
                 normalize=False, scale_factor=1000.0, condition_noise=0.0, 
                 condition_noise_schedule=None, start_sample_index=0):
        """
        Args:
            pickle_file: Path to the pickle file containing KS data
            batch_size: Number of samples per batch
            condition_steps: Number of historical steps to use for conditioning
            timesteps: Maximum number of timesteps to use from the data
            dt: Time step increment
            normalize: Whether to normalize the data
            scale_factor: Factor to scale delta values for better learning
            condition_noise: Standard deviation of noise to add to conditioning data
            condition_noise_schedule: Optional function mapping condition step to noise level
            start_sample_index: Index of first sample to use (discards earlier samples)
        """
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)  # shape [1024, some_big_T]
        data = data.astype(np.float32)

        # Apply start_sample_index to discard initial samples
        if start_sample_index > 0:
            if start_sample_index >= data.shape[1]:
                raise ValueError(f"start_sample_index ({start_sample_index}) exceeds data length ({data.shape[1]})")
            
            original_length = data.shape[1]
            data = data[:, start_sample_index:]
            print(f"Discarding first {start_sample_index} samples. "
                  f"Data shape changed from [1024, {original_length}] to [1024, {data.shape[1]}]")

        # Keep only up to 'timesteps' columns
        if timesteps > data.shape[1]:
            print(f"Warning: Requested {timesteps=} exceeds available {data.shape[1]=}")
            print(f"Using all available {data.shape[1]} timesteps")
        else:
            data = data[:, :timesteps]
            print(f"Using {timesteps} timesteps")

        # Optionally normalize
        if normalize:
            self.mean = data.mean()
            self.std = data.std()
            data = (data - self.mean) / self.std
            print(f"Normalized data: mean={self.mean:.4f}, std={self.std:.4f}")

        self.data = data
        self.batch_size = batch_size
        self.condition_steps = condition_steps
        self.dt = dt
        self.scale_factor = scale_factor
        self.condition_noise = condition_noise
        self.condition_noise_schedule = condition_noise_schedule
        self.start_sample_index = start_sample_index
        self.normalize = normalize
        
        # Need at least condition_steps*dt historical points and 1 future point
        self.min_index = (condition_steps - 1) * dt
        self.max_index = data.shape[1] - dt - 1
        
        if self.min_index > self.max_index:
            raise ValueError(f"Not enough data points for {condition_steps=} and {dt=}.")
            
        print(f"Using indices from {self.min_index} to {self.max_index}")
        print(f"Condition steps: {condition_steps}, dt: {dt}")
        
        # Calculate statistics of the deltas for reference
        sample_indices = np.random.randint(0, self.max_index, min(1000, self.max_index))
        sample_deltas = np.array([self.data[:, i + self.dt] - self.data[:, i] for i in sample_indices])
        self.delta_mean = float(sample_deltas.mean())
        self.delta_std = float(sample_deltas.std())
        self.delta_min = float(sample_deltas.min())
        self.delta_max = float(sample_deltas.max())
        
        print(f"Delta statistics before scaling:")
        print(f"  Mean: {self.delta_mean:.8f}")
        print(f"  Std:  {self.delta_std:.8f}")
        print(f"  Min:  {self.delta_min:.8f}")
        print(f"  Max:  {self.delta_max:.8f}")
        print(f"Applying scale factor: {self.scale_factor}")
        
        # Calculate statistics after scaling
        self.scaled_delta_mean = self.delta_mean * self.scale_factor
        self.scaled_delta_std = self.delta_std * self.scale_factor
        self.scaled_delta_min = self.delta_min * self.scale_factor
        self.scaled_delta_max = self.delta_max * self.scale_factor
        
        print(f"Delta statistics after scaling:")
        print(f"  Mean: {self.scaled_delta_mean:.4f}")
        print(f"  Std:  {self.scaled_delta_std:.4f}")
        print(f"  Min:  {self.scaled_delta_min:.4f}")
        print(f"  Max:  {self.scaled_delta_max:.4f}")
        
        if condition_noise > 0:
            print(f"Adding Gaussian noise with std={condition_noise} to conditioning data during training")
        if condition_noise_schedule is not None:
            print(f"Using custom noise schedule for conditioning data during training")

    def set_start_index(self, start_index):
        """Set the starting index for data extraction."""
        # Make sure index is within bounds
        if start_index < self.min_index:
            print(f"Warning: Requested start_index {start_index} is less than minimum allowed index {self.min_index}.")
            start_index = self.min_index
        
        if start_index > self.max_index:
            print(f"Warning: Requested start_index {start_index} exceeds maximum allowed index {self.max_index}.")
            start_index = self.max_index
        
        # Store the index for later use
        self.current_index = start_index
        print(f"Data loader start index set to {self.current_index}")
        
        # Return the object to allow method chaining
        return self

    def __iter__(self):
        return self

    def __next__(self):
        # Determine indices to use
        if hasattr(self, 'current_index'):
            # Use the stored index for the first element, random for others
            if self.batch_size == 1:
                idx = np.array([self.current_index])
            else:
                # For batch size > 1, use the set index for the first item,
                # random indices for the rest
                random_idx = np.random.randint(self.min_index, self.max_index + 1, self.batch_size - 1)
                idx = np.concatenate([[self.current_index], random_idx])
        else:
            # Original behavior - pick random indices
            idx = np.random.randint(self.min_index, self.max_index + 1, self.batch_size)
            self.current_index = idx[0]  # Store for reference
        
        # Initialize arrays for batch
        x_cond_batch = np.zeros((self.batch_size, self.condition_steps, self.data.shape[0]), dtype=np.float32)
        delta_batch = np.zeros((self.batch_size, self.data.shape[0]), dtype=np.float32)
        
        # For each sample in batch
        for b in range(self.batch_size):
            # Current index for this batch item
            current_idx = idx[b]
            
            # Get the condition sequence (in reverse chronological order)
            for c in range(self.condition_steps):
                history_idx = current_idx - c * self.dt
                x_cond_batch[b, c] = self.data[:, history_idx]
            
            # Get the future state and calculate delta
            most_recent_state = self.data[:, current_idx]
            future_state = self.data[:, current_idx + self.dt]
            delta_batch[b] = (future_state - most_recent_state) * self.scale_factor
        
        # Add noise to conditioning data if requested
        if self.condition_noise > 0 or self.condition_noise_schedule is not None:
            for c in range(self.condition_steps):
                # Determine noise level for this condition step
                if self.condition_noise_schedule is not None:
                    # Use custom noise schedule
                    noise_level = self.condition_noise_schedule(c)
                else:
                    # Use fixed noise level
                    noise_level = self.condition_noise
                
                # Generate and add noise
                if noise_level > 0:
                    noise = np.random.normal(0, noise_level, x_cond_batch[:, c, :].shape)
                    x_cond_batch[:, c, :] += noise
        
        # Move to device
        return jax.device_put(x_cond_batch), jax.device_put(delta_batch)
    
    def get_scale_factor(self):
        """Return the scale factor used for deltas"""
        return self.scale_factor
    
    def unscale_delta(self, scaled_delta):
        """Convert a scaled delta back to its original scale"""
        return scaled_delta / self.scale_factor
    
    def get_normalized_params(self):
        """Return normalization parameters if normalize=True was used"""
        return getattr(self, 'mean', 0.0), getattr(self, 'std', 1.0)

class KSSequenceDataLoader:
    """
    Dataloader for sequence-to-sequence modeling with the NoisyDataset.pkl file.
    
    Provides batches of sequential data with:
    - Input sequence: A sequence of partial samples for each timestep (one randomly 
      selected sample per timestep) with shape (batch_size, seq_len, spatial_dim)
    - Target sequence: The corresponding ground truth target data with 
      shape (batch_size, seq_len, spatial_dim)
    
    Memory-efficient implementation that loads data in smaller chunks as needed.
    """
    def __init__(self, dataset_file, batch_size=2, seq_len=100, dt=100, shuffle=True, seed=None):
        """
        Initialize the sequence dataloader.
        
        Args:
            dataset_file: Path to the NoisyDataset.pkl file
            batch_size: Number of sequences per batch
            seq_len: Length of sequences to generate
            dt: Time step increment between sequence elements (default: 100)
            shuffle: Whether to shuffle the starting indices
            seed: Random seed for reproducibility
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        self.dataset_file = dataset_file
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dt = dt          # New parameter for time step increment
        self.shuffle = shuffle
        
        # Open the file once to get metadata
        print(f"Loading dataset metadata from: {dataset_file}")
        with open(dataset_file, "rb") as f:
            # Load only the config and shape information, not the entire arrays
            data_dict = pickle.load(f)
            
            # Get the shapes of the data
            self.num_timesteps = data_dict["target_data"].shape[0]
            self.spatial_dim = data_dict["target_data"].shape[1]
            self.num_samples = data_dict["partial_samples"].shape[1]
            
            # Keep the config
            self.config = data_dict.get("config", {})
        
        # Calculate number of valid starting indices
        # Need enough timesteps for sequence_length * dt steps
        # The last valid index is: num_timesteps - (seq_len * dt)
        self.max_start_idx = self.num_timesteps - (seq_len * dt)
        
        # Validate parameters
        if self.max_start_idx < 0:
            raise ValueError(f"Dataset has {self.num_timesteps} timesteps, which is less than "
                             f"required {seq_len * dt} timesteps for sequence length {seq_len} with dt={dt}")
            
        # Generate all possible starting indices
        self.all_indices = np.arange(0, self.max_start_idx + 1, dtype=np.int32)
        
        # Shuffle indices if requested
        if self.shuffle:
            np.random.shuffle(self.all_indices)
        
        # Calculate number of complete batches available
        self.num_batches = len(self.all_indices) // batch_size
        
        if self.num_batches == 0:
            raise ValueError(f"Not enough data for even a single batch of size {batch_size} "
                            f"with sequence length {seq_len} and dt={dt}")
        
        print(f"Initialized KSSequenceDataLoader with:")
        print(f"  {self.num_timesteps} timesteps")
        print(f"  {self.num_samples} partial samples per timestep")
        print(f"  {self.spatial_dim} spatial dimensions")
        print(f"  {self.seq_len} sequence length")
        print(f"  dt={self.dt} timesteps between sequence elements")
        print(f"  {self.batch_size} batch size")
        print(f"  {self.num_batches} total batches available")
        print(f"  Shuffle: {self.shuffle}")
    
    def __iter__(self):
        # Shuffle indices at the start of each epoch if requested
        if self.shuffle:
            np.random.shuffle(self.all_indices)
        
        # Reset iteration state
        self.current_batch = 0
        return self
    
    def __next__(self):
        """Returns the next batch of sequential data."""
        if self.current_batch >= self.num_batches:
            # End of iteration
            raise StopIteration
        
        # Get the batch of starting indices
        batch_start_idx = self.current_batch * self.batch_size
        batch_end_idx = min((self.current_batch + 1) * self.batch_size, len(self.all_indices))
        batch_indices = self.all_indices[batch_start_idx:batch_end_idx]
        
        actual_batch_size = len(batch_indices)
        
        # Calculate the range of data we need to load
        # For each sequence, we need data from start_idx to start_idx + (seq_len-1)*dt
        min_idx = min(batch_indices)
        # The highest index we'll need is the last element of the last sequence
        max_idx = max(batch_indices) + (self.seq_len - 1) * self.dt + 1
        
        # Load only the data we need for this batch
        with open(self.dataset_file, "rb") as f:
            data_dict = pickle.load(f)
            
            # Get the relevant slices of data
            target_data_slice = data_dict["target_data"][min_idx:max_idx]
            partial_samples_slice = data_dict["partial_samples"][min_idx:max_idx]
        
        # Prepare batch arrays
        input_batch = np.zeros((actual_batch_size, self.seq_len, self.spatial_dim), dtype=np.float32)
        target_batch = np.zeros((actual_batch_size, self.seq_len, self.spatial_dim), dtype=np.float32)
        
        # Fill the batch with sequential data, using dt to space elements
        for i, start_idx in enumerate(batch_indices):
            # Calculate target sequence indices relative to start_idx
            # Using dt steps between elements
            seq_indices = [start_idx + j * self.dt for j in range(self.seq_len)]
            
            # Adjust to local indices (relative to min_idx where our slice starts)
            local_indices = [idx - min_idx for idx in seq_indices]
            
            # Extract the target sequence (ground truth)
            for j, local_idx in enumerate(local_indices):
                target_batch[i, j] = target_data_slice[local_idx]
            
            # For each timestep in the sequence, randomly select one partial sample
            for j, local_idx in enumerate(local_indices):
                # Random sample index for this timestep
                sample_idx = np.random.randint(0, self.num_samples)
                
                # Extract the randomly selected partial sample
                input_batch[i, j] = partial_samples_slice[local_idx, sample_idx]
        
        # Increment batch counter
        self.current_batch += 1
        
        # Return as JAX arrays on CPU to avoid GPU memory issues
        try:
            return jax.device_put(input_batch), jax.device_put(target_batch)
        except:
            # Fallback to CPU if we hit memory issues
            return jax.device_put(input_batch, jax.devices("cpu")[0]), jax.device_put(target_batch, jax.devices("cpu")[0])
    
    def __len__(self):
        """Returns the number of batches."""
        return self.num_batches