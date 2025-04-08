import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial

def compute_continuous_diffusion_loss(model, vpsde, cond_batch, target_batch, t_batch, key):
    """
    Computes the average denoising score-matching loss for continuous diffusion.
    This is the standard score-matching approach.
    
    Args:
        model: ksModel1d instance.
        vpsde: VPSDE instance.
        cond_batch: x(t) with shape (batch_size, spatial_dim).
        target_batch: x(t+1) or delta_x with shape (batch_size, spatial_dim).
        t_batch: Noise-level times, shape (batch_size,).
        key: JAX random PRNGKey or a batch of keys with shape (batch_size, 2).

    Returns:
        A scalar loss averaged over the entire batch.
    """
    batch_size = cond_batch.shape[0]
    
    # Check if key is a single key or already a batch of keys
    if key.ndim == 1:
        # Single key, split it into batch_size keys
        rngs = jax.random.split(key, batch_size)
    elif key.ndim == 2 and key.shape[0] == batch_size:
        # Already have a batch of keys
        rngs = key
    else:
        raise ValueError(f"Unexpected key shape: {key.shape}. Expected (2,) or ({batch_size}, 2)")
    
    def per_example_loss(x_cond, x_target, t, subkey):
        """
        1) Forward-sample x_target -> noisy sample.
        2) Compute the target score.
        3) Compute score_pred from the model using x_cond as conditioning.
        4) Return weighted MSE loss over the spatial dimension.
        """
        # Generate noisy sample
        skey1, skey2 = jax.random.split(subkey)
        y = vpsde.forward_sample(x_target, t, skey1)
        mean, std = vpsde.marginal_prob(x_target, t)
        target_score = -(y-mean) / std
        # SNR-based weighting
        # Check if the SDE has the alpha method (VPSDE) or not (VESDE)
        weight = 1
        """Optional, removed stability normalization.
        if hasattr(vpsde, 'alpha'):
            alpha = vpsde.alpha(t)
            snr = (alpha**2) / (std**2 + 1e-6)
        else:
            # For VESDE or other SDEs that don't have alpha
            snr = 1.0 / (std**2 + 1e-6)
        
        weight = snr / (1 + snr)"""

        
        # Get model prediction
        score_pred = model(y, x_cond, t, subkey)  # shape (spatial_dim,)

        # Weighted MSE loss using Song's approach: weight * E[(std * score_pred - scaled_score)²]
        # This is numerically stable compared to directly using true_score = scaled_score/std
        return weight * jnp.mean((std*score_pred - target_score) ** 2)

    # Vectorize over the batch dimension
    losses = jax.vmap(per_example_loss)(
        cond_batch, target_batch, t_batch, rngs
    )  # shape = (batch_size,)

    return jnp.mean(losses)


def compute_lazy_diffusion_loss(model, vpsde, cond_batch, target_batch, t_batch, key):
    """
    Computes a direct prediction loss for a lazy diffusion approach.
    Instead of denoising gradually, this directly predicts the target from noise.
    
    Args:
        model: ksModel1d instance.
        vpsde: VPSDE instance.
        cond_batch: x(t) with shape (batch_size, spatial_dim).
        target_batch: x(t+1) or delta_x with shape (batch_size, spatial_dim).
        t_batch: Noise-level times (fixed at t=1.0 for lazy diffusion).
        key: JAX random PRNGKey or a batch of keys with shape (batch_size, 2).

    Returns:
        A scalar loss averaged over the entire batch.
    """
    batch_size = cond_batch.shape[0]
    
    # Check if key is a single key or already a batch of keys
    if key.ndim == 1:
        # Single key, split it into batch_size keys
        rngs = jax.random.split(key, batch_size)
    elif key.ndim == 2 and key.shape[0] == batch_size:
        # Already have a batch of keys
        rngs = key
    else:
        raise ValueError(f"Unexpected key shape: {key.shape}. Expected (2,) or ({batch_size}, 2)")
    
    def per_example_loss(x_cond, x_target, t, subkey):
        """
        1) Generate a pure noise sample
        2) Use the model to predict the target directly from noise
        3) Return MSE loss over the spatial dimension
        """
        # Generate pure noise sample (fixed at t=1.0 for maximum noise)
        noise = jax.random.normal(subkey, shape=x_target.shape)
        
        # Get model prediction directly from noise
        # The model should learn to map from noise to target in one step
        pred = model(noise, x_cond, jnp.ones_like(t), subkey)  # shape (spatial_dim,)

        # MSE loss between prediction and target
        return jnp.mean((pred - x_target) ** 2)

    # Vectorize over the batch dimension
    losses = jax.vmap(per_example_loss)(
        cond_batch, target_batch, t_batch, rngs
    )  # shape = (batch_size,)

    return jnp.mean(losses)

def get_loss_fn(diffusion_type):
    """
    Returns the appropriate loss function based on the specified type.
    
    Args:
        diffusion_type: String identifier for the loss function ("cont_diffusion" or "lazy_diffusion")
        
    Returns:
        The corresponding loss function
    """
    if diffusion_type == "cont_diffusion":
        return compute_continuous_diffusion_loss
    elif diffusion_type == "lazy_diffusion":
        return compute_lazy_diffusion_loss
    else:
        raise ValueError(f"Unknown diffusion type: {diffusion_type}")


@eqx.filter_jit  
def compute_loss_with_params(model, vpsde, batch, key, loss_fn, tau_batch=1):
    """
    Compute loss for optimization with parameters.
    Supports multiple diffusion times per data point.
    
    Args:
        model: Complete model with parameters.
        vpsde: VPSDE or VESDE instance.
        batch: Tuple of (cond_batch, target_batch).
        key: JAX random key.
        loss_fn: Loss function to use.
        tau_batch: Number of diffusion times to sample per data point.
        
    Returns:
        Scalar loss value.
    """
    cond_batch, target_batch = batch
    
    # Get batch size
    batch_size = cond_batch.shape[0]
    
    # Generate tau_batch different diffusion time values
    key_time, key_noise = jax.random.split(key)
    
    if loss_fn.__name__ == "compute_lazy_diffusion_loss":
        # For lazy diffusion, use fixed t=1.0 (maximum noise)
        t_values = jnp.ones((tau_batch,))
    else:
        # For continuous diffusion, sample random time values
        t_values = jax.random.uniform(
            key_time, 
            shape=(tau_batch,), 
            minval=1e-4, 
            maxval=1.0
        )
        # Apply importance sampling if using power schedule
        if hasattr(vpsde, 'schedule_type') and vpsde.schedule_type == "power" and hasattr(vpsde, 'power'):
            t_values = 1 - (1 - t_values)**(1/vpsde.power) #importance sampling
    
    # Reshape inputs to create copies for each diffusion time
    # From [batch_size, ...] to [batch_size * tau_batch, ...]
    
    # First, tile the data to have tau_batch copies of each item
    # [batch_size, ...] -> [batch_size, tau_batch, ...]
    cond_expanded = jnp.repeat(cond_batch[:, None], tau_batch, axis=1)
    target_expanded = jnp.repeat(target_batch[:, None], tau_batch, axis=1)
    
    # Then reshape to [batch_size * tau_batch, ...]
    cond_expanded = cond_expanded.reshape(batch_size * tau_batch, *cond_batch.shape[1:])
    target_expanded = target_expanded.reshape(batch_size * tau_batch, *target_batch.shape[1:])
    
    # Repeat each time value for each batch item
    # [tau_batch] -> [batch_size * tau_batch]
    t_batch = jnp.repeat(t_values, batch_size)
    
    # Generate noise keys for each expanded sample
    noise_keys = jax.random.split(key_noise, batch_size * tau_batch)
    
    # Compute and return loss using the full model directly
    return loss_fn(model, vpsde, cond_expanded, target_expanded, t_batch, noise_keys)