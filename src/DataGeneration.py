import os
import sys
import time
import argparse
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from tqdm.auto import tqdm

from utils import load_config, create_sde, load_model_from_checkpoint
from DataLoaders import ksConditionalDataLoader


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate partially denoised samples using trained model")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the generated dataset")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of noisy samples per data point")
    parser.add_argument("--denoising_steps", type=int, default=500, help="Number of denoising steps (fewer = noisier)")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step size for denoising")
    parser.add_argument("--max_batches", type=int, default=None, help="Maximum number of batches to process (None = all)")
    parser.add_argument("--start_idx", type=int, default=None, help="Start from specific data index (overrides config)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--disable_progress", action="store_true", help="Disable progress bars for non-interactive environments")
    return parser.parse_args()


@eqx.filter_jit
def _generate_single_sample(model, sde, cond_data, target_data, subkey, num_steps, dt):
    """JIT-compiled function to generate a single partially denoised sample."""
    # Generate pure noise sample
    noise_key, sampling_key = jax.random.split(subkey)
    initial_noise = sde.prior_sampling(noise_key, target_data.shape)
    
    # Set up time steps for partial denoising
    time_steps = jnp.linspace(1.0, 1.0 - (num_steps * dt), num_steps)
    time_step_pairs = jnp.stack([time_steps[:-1], time_steps[1:]], axis=1)
    
    # Define the Euler-Maruyama step function
    def euler_maruyama_scan_step(carry, time_step):
        xt, key, cond_state = carry
        t_cur, t_next = time_step

        # Generate new key for each step
        key, score_key, noise_key = jax.random.split(key, 3)
        
        # Compute score
        score = model(xt, cond_state, t_cur, score_key)
        
        # Compute drift
        drift = -sde.drift(xt, t_cur) - sde.diffusion(xt, t_cur)**2 * score
        diffusion = sde.diffusion(xt, t_cur)

        # Compute time step
        dt = t_next - t_cur

        # Generate random noise
        noise = jax.random.normal(noise_key, shape=xt.shape)

        # Update using Euler-Maruyama
        next_xt = xt + drift * dt + diffusion * jnp.sqrt(jnp.abs(dt)) * noise * jnp.sign(dt)

        return (next_xt, key, cond_state), next_xt
    
    # Initialize carry
    initial_carry = (initial_noise, sampling_key, cond_data)
    
    # Run the scan
    (partially_denoised, _, _), _ = jax.lax.scan(
        euler_maruyama_scan_step, initial_carry, time_step_pairs
    )
    
    return partially_denoised


@eqx.filter_jit
def _generate_batch_samples(model, sde, cond_data, target_data, keys, num_steps, dt):
    """JIT-compiled function to generate multiple samples for a batch using vmap."""
    generate_vmap = jax.vmap(lambda k: _generate_single_sample(
        model, sde, cond_data, target_data, k, num_steps, dt
    ))
    return generate_vmap(keys)


def generate_partial_samples(model, sde, cond_data, target_data, key, num_samples=500, num_steps=500, dt=0.001, show_progress=False):
    """
    Generate partially denoised samples for a single batch.
    
    Args:
        model: Trained diffusion model
        sde: SDE instance
        cond_data: Conditional data with shape (condition_steps, spatial_dim)
        target_data: Target data with shape (spatial_dim,)
        key: JAX random key
        num_samples: Number of samples to generate
        num_steps: Number of denoising steps (fewer = noisier)
        dt: Time step size
        show_progress: Whether to show a progress bar for large sample generation
        
    Returns:
        Array of partially denoised samples with shape (num_samples, spatial_dim)
    """
    # For large number of samples with progress bar
    if show_progress and num_samples > 50:
        print(f"Generating {num_samples} samples for this batch...")
        
        # Generate in smaller batches with progress bar
        batch_size = 50  # Process in batches of 50 for progress display
        num_batches = (num_samples + batch_size - 1) // batch_size
        all_samples = []
        
        with tqdm(total=num_samples, desc="Generating samples", unit="sample") as pbar:
            for i in range(num_batches):
                # Determine batch size for this iteration
                current_batch_size = min(batch_size, num_samples - i * batch_size)
                
                # Split keys for this batch
                batch_keys = jax.random.split(key, current_batch_size + 1)
                key, sample_keys = batch_keys[0], batch_keys[1:]
                
                # JIT-compiled batch generation
                batch_samples = _generate_batch_samples(
                    model, sde, cond_data, target_data, sample_keys, num_steps, dt
                )
                
                all_samples.append(batch_samples)
                pbar.update(current_batch_size)
        
        # Combine batches
        samples = jnp.concatenate(all_samples, axis=0)
    else:
        # Generate all samples at once - faster for no progress tracking
        sample_keys = jax.random.split(key, num_samples)
        samples = _generate_batch_samples(
            model, sde, cond_data, target_data, sample_keys, num_steps, dt
        )
    
    return samples


def process_dataset(config, model, sde, args):
    """
    Process the entire dataset to generate partially denoised samples.
    
    Args:
        config: Experiment configuration
        model: Trained model
        sde: SDE instance
        args: Command-line arguments
        
    Returns:
        Dictionary with original data and generated samples
    """
    # Create data loader with batch_size=1
    data_loader = ksConditionalDataLoader(
        pickle_file=config.get("data_file", "data/KSData/KS_1024_500.pkl"),
        batch_size=1,
        condition_steps=config.get("condition_steps", 2),
        timesteps=config.get("timesteps", 160001),
        dt=config.get("dt", 1),
        normalize=config.get("normalize", False),
        condition_noise=0.0,  # Always use clean conditioning for generation
        start_sample_index=config.get("start_sample_index", 0)
    )
    
    # Initialize random key
    key = jax.random.PRNGKey(config.get("seed", 42))
    
    # Initialize storage for results
    all_cond_data = []
    all_target_data = []
    all_partial_samples = []
    
    # Determine available index range
    min_index = data_loader.min_index
    max_index = data_loader.max_index
    
    # Determine how many batches to process
    if args.max_batches:
        max_index = min(max_index, min_index + args.max_batches - 1)
    
    batch_count = 0
    
    # Calculate number of indices to process
    total_indices = max_index - min_index + 1
    
    print(f"Starting dataset generation...")
    print(f"Processing indices from {min_index} to {max_index}")
    print(f"Generating {args.num_samples} partially denoised samples per index")
    print(f"Using {args.denoising_steps} denoising steps (partial denoising)")
    
    start_time = time.time()
    
    # Create progress bar
    pbar = tqdm(total=total_indices, desc="Generating samples", 
                unit="batch", ncols=100, position=0, leave=True)
    
    # Explicitly iterate through each index
    for idx in range(min_index, max_index + 1):
        # Set the data loader to this specific index
        data_loader.set_start_index(idx)
        
        # Get the batch at this index
        batch = next(iter(data_loader))
        
        cond_batch, target_batch = batch
        
        # Get the first (and only) item in the batch
        cond_data = cond_batch[0]  # Shape: (condition_steps, spatial_dim)
        target_data = target_batch[0]  # Shape: (spatial_dim,)
        
        # Update the key
        key, sample_key = jax.random.split(key)
        
        # Generate partially denoised samples
        partial_samples = generate_partial_samples(
            model, sde, cond_data, target_data, sample_key,
            num_samples=args.num_samples,
            num_steps=args.denoising_steps,
            dt=args.dt,
            show_progress=(batch_count == 0)  # Only show detailed progress for first batch
        )
        
        # Store data
        all_cond_data.append(cond_data)
        all_target_data.append(target_data)
        all_partial_samples.append(partial_samples)
        
        batch_count += 1
        
        # Update progress bar
        elapsed = time.time() - start_time
        pbar.update(1)
        pbar.set_postfix({
            "index": idx, 
            "elapsed": f"{elapsed:.2f}s",
            "samples": batch_count * args.num_samples
        })
    
    # Close progress bar
    pbar.close()
    
    # Convert to arrays
    all_cond_data = jnp.stack(all_cond_data)
    all_target_data = jnp.stack(all_target_data)
    all_partial_samples = jnp.stack(all_partial_samples)
    
    print(f"Dataset generation complete. Generated data shapes:")
    print(f"  Conditional data: {all_cond_data.shape}")
    print(f"  Target data: {all_target_data.shape}")
    print(f"  Partial samples: {all_partial_samples.shape}")
    
    return {
        "cond_data": all_cond_data,
        "target_data": all_target_data,
        "partial_samples": all_partial_samples,
        "config": {
            "num_samples": args.num_samples,
            "denoising_steps": args.denoising_steps,
            "dt": args.dt,
            "original_config": config,
            "index_range": [min_index, max_index]
        }
    }

def main():
    """Main function to orchestrate the dataset generation process."""
    args = parse_args()
    
    # Configure tqdm to disable if requested
    if args.disable_progress:
        tqdm.__init__ = lambda *args, **kwargs: None
        tqdm.update = lambda *args, **kwargs: None
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override config with command-line arguments if provided
    if args.seed is not None:
        print(f"Overriding config seed ({config.get('seed', 42)}) with command-line seed ({args.seed})")
        config["seed"] = args.seed
    
    if args.start_idx is not None:
        print(f"Overriding config start_sample_index ({config.get('start_sample_index', 0)}) with command-line start_idx ({args.start_idx})")
        config["start_sample_index"] = args.start_idx
    
    # Create SDE
    sde = create_sde(config)
    
    # Load model
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model_key = jax.random.PRNGKey(config.get("seed", 42))
    model = load_model_from_checkpoint(config, args.checkpoint, model_key)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Process dataset
    start_time = time.time()
    results = process_dataset(config, model, sde, args)
    total_time = time.time() - start_time
    
    # Save results
    print(f"Saving generated dataset to: {args.output_file}")
    with open(args.output_file, "wb") as f:
        pickle.dump(results, f)
    
    # Print summary
    print("\nDataset generation complete!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Generated {len(results['target_data'])} data points")
    print(f"Each with {args.num_samples} partially denoised samples")
    print(f"Total samples: {len(results['target_data']) * args.num_samples}")


if __name__ == "__main__":
    main()
