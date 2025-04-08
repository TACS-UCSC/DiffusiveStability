import jax
import jax.numpy as jnp
import equinox as eqx
from src.models.layers import *
from typing import List, Tuple
from src.models.blocks import create_block
from src.models.utils import cast_eqx_layer

class MambaSeqToSeq(eqx.Module):
    """
    A sequence-to-sequence model based on Mamba state space blocks.
    
    This model processes sequential data using Mamba SSM blocks for both
    full sequence processing (during training) and autoregressive generation
    (during inference). The architecture consists of an input embedding layer,
    a stack of Mamba blocks with residual connections, and an output projection.
    
    The model supports two modes of operation:
    1. Training mode: Process full sequences at once via __call__ method
    2. Inference mode: Generate outputs step-by-step via generate_step method
    
    Attributes:
        embedding: Linear layer that maps input spatial dimension to hidden dimension
        blocks: List of Mamba blocks with residual connections
        head: Linear layer that maps hidden dimension back to spatial dimension
        dtype: Data type for model parameters and computations
    
    Parameters:
        spatial_dim (int): Dimension of input and output vectors
        hidden_dim (int): Dimension of hidden representations
        n_layers (int): Number of Mamba blocks in the stack
        dtype (jnp.dtype): Data type for model parameters and computations
        key (jax.random.PRNGKey): Random key for parameter initialization
        **block_kwargs: Additional arguments passed to create_block function
            These can include state_dim, kernel_size, etc. See blocks.create_block
            for all available parameters.
    
    Example:
        >>> model = MambaSeqToSeq(
        ...     spatial_dim=10,
        ...     hidden_dim=32,
        ...     n_layers=2,
        ...     state_dim=16,
        ...     kernel_size=4,
        ...     dtype=jnp.float32,
        ...     key=jax.random.PRNGKey(0)
        ... )
        >>> 
        >>> # Training mode with batch
        >>> x = jnp.zeros((16, 10))  # (seq_len, spatial_dim)
        >>> y = model(x)  # (seq_len, spatial_dim)
        >>> 
        >>> # Inference mode
        >>> cache = model.init_cache()
        >>> x0 = jnp.zeros((10,))  # (spatial_dim,)
        >>> y0, cache = model.generate_step(x0, cache)  # (spatial_dim,), updated cache
    """
    embedding: eqx.nn.Linear
    blocks: List[eqx.Module]
    head: eqx.nn.Linear
    dtype: jnp.dtype = jnp.float32

    def __init__(
        self,
        spatial_dim: int,
        hidden_dim: int,
        n_layers: int,
        dtype: jnp.dtype = jnp.float32,
        key: jax.random.PRNGKey = None,
        **block_kwargs
    ):
        super().__init__()
        self.dtype = dtype
        
        keys = jax.random.split(key, n_layers + 2)
        
        # Input embedding
        self.embedding = cast_eqx_layer(
            eqx.nn.Linear(spatial_dim, hidden_dim, key=keys[0]), 
            dtype=self.dtype
        )
        
        # Mamba blocks
        self.blocks = []
        for i in range(n_layers):
            block_key = keys[i + 1]
            block = create_block(
                dim=hidden_dim,
                layer_idx=i,
                dtype=self.dtype,
                key=block_key,
                **block_kwargs
            )
            self.blocks.append(block)
        
        # Output head
        self.head = cast_eqx_layer(
            eqx.nn.Linear(hidden_dim, spatial_dim, key=keys[-1]),
            dtype=self.dtype
        )
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """Training mode: process full sequences."""
        # x shape: (seq_len, spatial_dim)
        
        # Embed input
        x = jax.vmap(self.embedding)(x)  # (seq_len, hidden_dim)
        
        # Pass through blocks
        res = None
        for block in self.blocks:
            x, res = block(x, res)
        
        # Project to output space
        x = jax.vmap(self.head)(x)  # (seq_len, spatial_dim)
        
        return x
    
    def init_cache(self) -> List:
        """Initialize cache for autoregressive generation."""
        n_layers = len(self.blocks)
        cache = [None] * n_layers
        
        for i, block in enumerate(self.blocks):
            # Get dimensions from the block
            state_dim = block.mixer.state_dim
            inner_dim = block.mixer.conv1d.out_channels
            kernel_size = block.mixer.conv1d.kernel_size
            
            # Conv state - shape (inner_dim, kernel_size)
            conv_state = jnp.zeros((inner_dim, kernel_size))
            
            # SSM state - shape (inner_dim, state_dim)
            ssm_state = jnp.zeros((inner_dim, state_dim))
            
            cache[i] = (conv_state, ssm_state)
        
        return cache
    
    def generate_step(
        self, 
        x: jax.Array, 
        cache: List
    ) -> Tuple[jax.Array, List]:
        """Inference mode: generate one step at a time."""
        # x shape: (spatial_dim,)
        
        # Embed input
        x = self.embedding(x)  # (hidden_dim,)
        
        # Pass through blocks
        res = None
        for block in self.blocks:
            x, res = block.generate_step(x, res, cache)
        
        # Project to output space
        x = self.head(x)  # (spatial_dim,)
        
        return x, cache

class ksModel1d(eqx.Module):
    encoder: Encoder
    transformerBlocks: list[TransformerBlock]
    decoder: eqx.nn.Linear
    marginal_prob_std: callable = None

    def __init__(self,
                 spatial_dim: int,
                 hidden_dim: int,
                 num_heads: int,
                 phead_scale: int,
                 num_layers: int,
                 key: jax.random.PRNGKey,
                 marginal_prob_std: callable = None):
        keys = jax.random.split(key, num_layers+2)
        self.transformerBlocks = [
            TransformerBlock(
                spatial_dim=spatial_dim,
                hidden_dim=hidden_dim, 
                num_heads = num_heads,
                phead_scale = phead_scale,
                key=keys[i]) 
                for i in range(num_layers)]
        self.encoder = Encoder(spatial_dim, hidden_dim, keys[-2])
        self.decoder = eqx.nn.Linear(in_features=hidden_dim, 
                                     out_features=1,
                                     use_bias = False,
                                     key = keys[-1])
        self.marginal_prob_std = marginal_prob_std
    def __call__(self, noisy_state: jax.Array, cond_state: jax.Array, t: float, key) -> jax.Array:
            
            x = self.encoder(noisy_state=noisy_state,
                             cond_state=cond_state,
                             t = t)
            for layer in self.transformerBlocks:
                x = layer(x)
            x = jax.vmap(self.decoder)(x)
            x = jnp.squeeze(x, axis=-1)
            if self.marginal_prob_std is not None:
                _, std_t = self.marginal_prob_std(0, t)
                x = x / (std_t + 1e-6)  # small epsilon for safety

            return x
    


class ksModelConditional1d(eqx.Module):
    encoder: ConditionalEncoder
    transformerBlocks: list[TransformerBlock]
    decoder: eqx.nn.Linear
    marginal_prob_std: callable = None

    def __init__(self,
                spatial_dim: int,
                hidden_dim: int,
                num_heads: int,
                phead_scale: int,
                num_layers: int,
                key: jax.random.PRNGKey,
                marginal_prob_std: callable = None,
                cond_states: int = 2):
        keys = jax.random.split(key, num_layers+2)
        self.transformerBlocks = [
            TransformerBlock(
                spatial_dim=spatial_dim,
                hidden_dim=hidden_dim, 
                num_heads = num_heads,
                phead_scale = phead_scale,
                key=keys[i]) 
                for i in range(num_layers)]
        self.encoder = ConditionalEncoder(spatial_dim, hidden_dim, keys[-2], cond_states)
        self.decoder = eqx.nn.Linear(in_features=hidden_dim, 
                                     out_features=1,
                                     use_bias = False,
                                     key = keys[-1])
        self.marginal_prob_std = marginal_prob_std
    def __call__(self, noisy_state: jax.Array, cond_state: jax.Array, t: float, key) -> jax.Array:
            
            x = self.encoder(noisy_state=noisy_state,
                             cond_state=cond_state,
                             t = t)
            for layer in self.transformerBlocks:
                x = layer(x)
            x = jax.vmap(self.decoder)(x)
            x = jnp.squeeze(x, axis=-1)
            if self.marginal_prob_std is not None:
                _, std_t = self.marginal_prob_std(0, t)
                x = x / (std_t + 1e-6)  # small epsilon for safety

            return x

    


        
