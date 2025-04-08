import jax
import jax.numpy as jnp
import equinox as eqx
from src.models.layers import *

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

    


        
