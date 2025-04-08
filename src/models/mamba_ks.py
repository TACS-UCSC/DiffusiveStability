import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial
import math

# Simplified Mamba block implementation

class SimpleMambaBlock(eqx.Module):
    """
    Simplified implementation of the Mamba block.
    
    Instead of using the full selective SSM mechanism, we use a simplified
    version with a 1D convolution followed by a GRU-like recurrent step.
    """
    # Input projections
    proj_in: eqx.nn.Linear   # Project input to hidden dimensions 
    
    # Convolution for local context
    conv1d: eqx.nn.Conv1d
    
    # SSM parameters
    A: jax.Array  # State transition matrix (diagonal)
    B: eqx.nn.Linear  # Input projection
    C: eqx.nn.Linear  # Output projection
    D: jax.Array  # Skip connection
    
    # Output projection
    proj_out: eqx.nn.Linear
    
    # Configuration
    hidden_dim: int
    state_dim: int
    expand: int
    expanded_dim: int
    
    def __init__(
        self,
        dim: int,
        state_dim: int = 16,
        expand: int = 2,
        kernel_size: int = 4,
        key: jax.random.PRNGKey = None,
    ):
        keys = jax.random.split(key, 6)
        
        self.hidden_dim = dim
        self.state_dim = state_dim
        self.expand = expand
        
        # Compute expanded dimension and store as attribute
        self.expanded_dim = dim * expand
        
        # Input projection
        self.proj_in = eqx.nn.Linear(
            in_features=dim,
            out_features=self.expanded_dim * 2,  # For gating
            key=keys[0]
        )
        
        # 1D convolution for local context
        self.conv1d = eqx.nn.Conv1d(
            in_channels=self.expanded_dim,
            out_channels=self.expanded_dim,
            kernel_size=kernel_size,
            padding='SAME',
            groups=self.expanded_dim,  # Depthwise convolution
            key=keys[1]
        )
        
        # SSM parameters
        # Initialize A as negative values on the diagonal
        A_diag = -jnp.exp(jnp.linspace(0, 2, state_dim))
        self.A = jnp.tile(A_diag, (self.expanded_dim, 1))
        
        # B and C as learnable projections
        self.B = eqx.nn.Linear(
            in_features=self.expanded_dim,
            out_features=state_dim * self.expanded_dim,
            key=keys[2]
        )
        
        self.C = eqx.nn.Linear(
            in_features=state_dim * self.expanded_dim,
            out_features=self.expanded_dim,
            key=keys[3]
        )
        
        # D as learnable skip connection (initialized to ones)
        self.D = jnp.ones((self.expanded_dim,))
        
        # Output projection
        self.proj_out = eqx.nn.Linear(
            in_features=self.expanded_dim,
            out_features=dim,
            key=keys[4]
        )
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Apply the Mamba block to a sequence.
        
        Args:
            x: Input tensor of shape (seq_len, hidden_dim)
            
        Returns:
            Output tensor of shape (seq_len, hidden_dim)
        """
        # Project and split for gating
        x_proj = jax.vmap(self.proj_in)(x)  # (seq_len, expanded_dim * 2)
        x_gate, x_proj = jnp.split(x_proj, 2, axis=-1)  # Each (seq_len, expanded_dim)
        
        # Apply convolution for local context
        # Use the correct shape for equinox's Conv1d which expects (channels, seq_len)
        seq_len = x_proj.shape[0]
        x_conv = x_proj.T  # (expanded_dim, seq_len)
        x_conv = self.conv1d(x_conv)  # (expanded_dim, seq_len)
        x_conv = x_conv.T  # (seq_len, expanded_dim)
        
        # Apply SiLU activation
        x_conv = jax.nn.silu(x_conv)
        
        # Discretized selective scan
        # For simplicity in this implementation, we use a recurrent loop
        # rather than the parallel scan algorithm from the paper
        def scan_fn(carry, x_t):
            # Unpack the current state
            h = carry  # (expanded_dim, state_dim)
            
            # Get input at this timestep
            u = x_t  # (expanded_dim,)
            
            # Reshape B and compute input projection
            B_t = jax.vmap(self.B)(jnp.ones((self.expanded_dim, 1)))
            B_t = B_t.reshape(self.expanded_dim, self.state_dim)
            
            # Update state with discretized state space model
            # h_next = Ah + Bu
            h_next = h * jnp.exp(self.A) + u[:, None] * B_t
            
            # Compute output
            # Reshape C and compute output projection
            C_t = jax.vmap(self.C)(h_next.reshape(self.expanded_dim, self.state_dim))
            y = C_t + u * self.D
            
            return h_next, y
        
        # Initialize state with zeros
        init_state = jnp.zeros((self.expanded_dim, self.state_dim))
        
        # Apply sequential scan over the sequence
        # This is inefficient but conceptually simpler - a real implementation
        # would use the parallel scan algorithm
        _, y = jax.lax.scan(
            scan_fn,
            init_state,
            x_conv
        )
        
        # Apply gating
        y = y * jax.nn.silu(x_gate)
        
        # Project back to hidden dimension
        output = jax.vmap(self.proj_out)(y)
        
        return output
    
    def init_state(self, batch_size=1):
        """Initialize recurrent state for inference."""
        return jnp.zeros((batch_size, self.expanded_dim, self.state_dim))
    
    def step(self, x_t, state):
        """Single step for autoregressive inference."""
        # Project input
        x_proj = self.proj_in(x_t)  # (expanded_dim * 2,)
        x_gate, x_proj = jnp.split(x_proj, 2)  # Each (expanded_dim,)
        
        # Update convolution state (simplified)
        x_conv = jax.nn.silu(x_proj)
        
        # Update SSM state
        B_t = self.B(x_conv.reshape(-1, 1)).reshape(self.expanded_dim, self.state_dim)
        new_state = state * jnp.exp(self.A) + x_conv[:, None] * B_t
        
        # Compute output
        C_t = self.C(new_state.reshape(-1, 1)).reshape(self.expanded_dim)
        y = C_t + x_conv * self.D
        
        # Apply gating
        y = y * jax.nn.silu(x_gate)
        
        # Project back to hidden dimension
        output = self.proj_out(y)
        
        return output, new_state


class ResidualBlock(eqx.Module):
    """
    Residual block with a Mamba layer and normalization.
    """
    norm: eqx.nn.LayerNorm
    mamba: SimpleMambaBlock
    
    def __init__(
        self,
        dim: int,
        state_dim: int = 16,
        expand: int = 2,
        kernel_size: int = 4,
        key: jax.random.PRNGKey = None,
    ):
        keys = jax.random.split(key, 2)
        self.norm = eqx.nn.LayerNorm((dim,))
        self.mamba = SimpleMambaBlock(
            dim=dim,
            state_dim=state_dim,
            expand=expand,
            kernel_size=kernel_size,
            key=keys[0]
        )
    
    def __call__(self, x: jax.Array) -> jax.Array:
        # Apply normalization
        h = jax.vmap(self.norm)(x)
        
        # Apply Mamba block
        h = self.mamba(h)
        
        # Residual connection
        return x + h
    
    def init_state(self, batch_size=1):
        """Initialize state for inference."""
        return self.mamba.init_state(batch_size)
    
    def step(self, x_t, state):
        """Single step for autoregressive inference."""
        # Apply normalization
        h = self.norm(x_t)
        
        # Apply Mamba step
        h, new_state = self.mamba.step(h, state)
        
        # Residual connection
        return x_t + h, new_state


class KSMambaEncoder(eqx.Module):
    """
    Encoder for KS data that projects spatial states to higher dimensions.
    
    Inputs: 
        - x: (seq_len, spatial_dim)
    Output: (seq_len, model_dim)
    """
    spatial_proj: eqx.nn.Linear
    layer_norm: eqx.nn.LayerNorm
    model_dim: int
    
    def __init__(self, 
                 spatial_dim: int,
                 model_dim: int,
                 key: jax.random.PRNGKey):
        keys = jax.random.split(key, 2)
        
        self.model_dim = model_dim
        self.spatial_proj = eqx.nn.Linear(
            in_features=spatial_dim,
            out_features=model_dim,
            key=keys[0]
        )
        self.layer_norm = eqx.nn.LayerNorm((model_dim,))
    
    def __call__(self, x: jax.Array) -> jax.Array:
        # x shape: (seq_len, spatial_dim)
        # Apply projection to each sequence element
        x = jax.vmap(self.spatial_proj)(x)  # (seq_len, model_dim)
        # Apply layer norm to each sequence element
        x = jax.vmap(self.layer_norm)(x)  # (seq_len, model_dim)
        return x


class KSMambaDecoder(eqx.Module):
    """
    Decoder for KS data that projects model dimension back to spatial dimension.
    
    Inputs: 
        - x: (seq_len, model_dim)
    Output: (seq_len, spatial_dim)
    """
    spatial_proj: eqx.nn.Linear
    
    def __init__(self, 
                 spatial_dim: int,
                 model_dim: int,
                 key: jax.random.PRNGKey):
        self.spatial_proj = eqx.nn.Linear(
            in_features=model_dim,
            out_features=spatial_dim,
            key=key
        )
    
    def __call__(self, x: jax.Array) -> jax.Array:
        # x shape: (seq_len, model_dim)
        # Apply projection to each sequence element
        return jax.vmap(self.spatial_proj)(x)  # (seq_len, spatial_dim)


class KSMambaModel(eqx.Module):
    """
    Mamba-based model for KS equation modeling.
    
    During training:
        Input: (seq_len, spatial_dim) → Output: (seq_len, spatial_dim)
    
    During inference (step-by-step):
        Input: (spatial_dim,) → Output: (spatial_dim,)
    """
    encoder: KSMambaEncoder
    blocks: list
    decoder: KSMambaDecoder
    model_dim: int
    
    def __init__(
        self,
        spatial_dim: int,
        key: jax.random.PRNGKey,
        model_dim: int = 128,
        num_layers: int = 4,
        state_dim: int = 16,
        expand: int = 2,
        kernel_size: int = 4,
    ):
        keys = jax.random.split(key, num_layers + 2)
        
        self.model_dim = model_dim
        
        # Input encoder
        self.encoder = KSMambaEncoder(
            spatial_dim=spatial_dim,
            model_dim=model_dim,
            key=keys[0]
        )
        
        # Mamba blocks
        self.blocks = []
        for i in range(num_layers):
            block = ResidualBlock(
                dim=model_dim,
                state_dim=state_dim,
                expand=expand,
                kernel_size=kernel_size,
                key=keys[i+1]
            )
            self.blocks.append(block)
        
        # Output decoder
        self.decoder = KSMambaDecoder(
            spatial_dim=spatial_dim,
            model_dim=model_dim,
            key=keys[-1]
        )
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """Process a full sequence during training.
        
        Args:
            x: Input sequence with shape (seq_len, spatial_dim)
            
        Returns:
            Output sequence with shape (seq_len, spatial_dim)
        """
        # Encode the input sequence
        hidden = self.encoder(x)
        
        # Apply Mamba blocks
        for block in self.blocks:
            hidden = block(hidden)
        
        # Decode to get predicted sequence
        return self.decoder(hidden)
    
    def init_cache(self, batch_size=1):
        """Initialize cache for autoregressive generation.
        
        Args:
            batch_size: Batch size for inference (typically 1)
            
        Returns:
            List of states for each block
        """
        return [block.init_state(batch_size) for block in self.blocks]
    
    def generate_step(self, x: jax.Array, cache) -> tuple:
        """Process a single step during autoregressive inference.
        
        Args:
            x: Current input with shape (spatial_dim,)
            cache: Cache of states for each block
            
        Returns:
            Tuple of (output, updated_cache)
        """
        # Encode the input
        hidden = self.encoder(x[None, :])[0]  # (model_dim,)
        
        # Apply Mamba blocks in autoregressive mode
        new_cache = []
        for i, block in enumerate(self.blocks):
            hidden, new_state = block.step(hidden, cache[i])
            new_cache.append(new_state)
        
        # Decode to get predicted output
        output = self.decoder(hidden[None, :])[0]  # (spatial_dim,)
        
        return output, new_cache
        
    def rollout(self, initial_state, num_steps):
        """Generate a sequence starting from an initial state.
        
        Args:
            initial_state: Starting state with shape (spatial_dim,)
            num_steps: Number of steps to predict
            
        Returns:
            Predicted sequence with shape (num_steps+1, spatial_dim)
            where the first step is the initial state
        """
        # Initialize cache for autoregressive generation
        cache = self.init_cache()
        
        # Initialize output sequence with the initial state
        sequence = [initial_state]
        current_state = initial_state
        
        # Generate steps autoregressively
        for _ in range(num_steps):
            next_state, cache = self.generate_step(current_state, cache)
            sequence.append(next_state)
            current_state = next_state
        
        return jnp.stack(sequence)