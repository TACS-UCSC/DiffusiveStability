import jax
import jax.numpy as jnp
import equinox as eqx
from src.models.layers import *

class SSWConditionalModel(eqx.Module):
    encoder: Encoder
    transformerBlocks: list[TransformerBlock]
    decoder: eqx.nn.Linear
    marginal_prob_std: callable = None
    cond_states: int = 2

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
        
        # Encoder handles space + time dimensions
        # Inputs are stacked as [batch, seq_len, spatial_dim]
        self.encoder = Encoder(spatial_dim, hidden_dim, keys[0])
        
        # Transformer blocks for processing
        self.transformerBlocks = [
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                phead_scale=phead_scale,
                key=keys[i+1]
            ) 
            for i in range(num_layers)
        ]
        
        # Final projection to output space
        self.decoder = eqx.nn.Linear(hidden_dim, spatial_dim, key=keys[-1])
        
        # Store conditioning steps
        self.cond_states = cond_states
        
        # Store noise scheduler if provided
        self.marginal_prob_std = marginal_prob_std

    def __call__(self, x, cond_state, t, key=None):
        """
        Forward pass for the SSW model with conditioning.
        
        Args:
            x: Target state to denoise of shape [batch_size, spatial_dim]
            cond_state: Conditioning states of shape [batch_size, condition_steps*2, spatial_dim]
            t: Diffusion time t (between 0 and 1)
            key: Optional PRNG key for any sampling operations
            
        Returns:
            Score function output (gradient of log probability) of shape [batch_size, spatial_dim]
        """
        batch_size = x.shape[0]
        
        # Scale x if marginal_prob_std is provided
        if self.marginal_prob_std:
            std = self.marginal_prob_std(t)
            x = x / std[:, None]  # Scale each row by the corresponding std
            
        # Add time embedding to each spatial position
        t_emb = get_timestep_embedding(t, x.shape[-1])
        t_emb = jnp.repeat(t_emb[:, None, :], x.shape[-1], axis=1)
        
        # Combine the noisy x with time embedding
        x = x + t_emb
        
        # Reshape to [batch, 1, spatial_dim] for encoder
        x = x.reshape((batch_size, 1, -1))
        
        # If conditioning info is provided, concatenate it with the input
        if cond_state is not None:
            # Conditional info should have shape [batch, C, spatial_dim]
            x = jnp.concatenate([cond_state, x], axis=1)
            
        # Encode the input sequence
        h = self.encoder(x)
        
        # Apply transformer blocks
        for block in self.transformerBlocks:
            h = block(h)
            
        # Extract the last token's representation (corresponding to the noisy x)
        # The conditioning states are at positions 0 to cond_states-1
        h_x = h[:, -1, :]
        
        # Project to the output space
        out = self.decoder(h_x)
        
        return out