import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize partially denoised samples dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset file")
    parser.add_argument("--batch_idx", type=int, default=None, help="Specific batch index to visualize (random if not specified)")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Directory to save visualizations")
    parser.add_argument("--num_visualizations", type=int, default=1, help="Number of random batches to visualize")
    return parser.parse_args()


def visualize_batch(batch_idx, cond_data, target_data, partial_samples, output_dir, config=None):
    """Visualize a single batch with true target and noisy samples."""
    # Get data for this batch
    cond_batch = cond_data[batch_idx]  # Shape: [condition_steps, spatial_dim]
    target = target_data[batch_idx]    # Shape: [spatial_dim]
    
    # Get 3 random samples from the partial samples
    num_samples = partial_samples.shape[1]
    sample_indices = np.random.choice(num_samples, size=3, replace=False)
    samples = partial_samples[batch_idx, sample_indices]  # Shape: [3, spatial_dim]
    
    # Create x-axis (spatial dimension)
    spatial_dim = target.shape[0]
    x = np.arange(spatial_dim)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Batch {batch_idx}: True State and Noisy Samples", fontsize=16)
    
    # Plot the most recent conditional state (x_t)
    axes[0, 0].plot(x, cond_batch[0], 'b-', linewidth=1.5)
    axes[0, 0].set_title("Most Recent State (x_t)")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot the true target state (x_{t+dt})
    axes[0, 1].plot(x, target, 'g-', linewidth=1.5)
    axes[0, 1].set_title("True Target State (x_{t+dt})")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot the noisy samples
    colors = ['r', 'm', 'c']
    for i, (idx, sample) in enumerate(zip(sample_indices, samples)):
        axes[1, i % 2].plot(x, sample, f'{colors[i]}-', linewidth=1.0, alpha=0.8)
        axes[1, i % 2].set_title(f"Noisy Sample {idx+1}")
        axes[1, i % 2].grid(True, alpha=0.3)
    
    # Add denoising info if available
    if config:
        steps = config.get("denoising_steps", "Unknown")
        dt = config.get("dt", "Unknown")
        fig.text(0.5, 0.01, f"Denoising: {steps} steps with dt={dt}", ha='center', fontsize=12)
    
    # Set x label for bottom plots
    for ax in axes[1, :]:
        ax.set_xlabel("Spatial Dimension")
    
    # Set y label for left plots
    for ax in axes[:, 0]:
        ax.set_ylabel("Value")
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"batch_{batch_idx}_visualization.png"))
    plt.close()
    
    print(f"Saved visualization for batch {batch_idx} to {output_dir}")


def main():
    """Main function to visualize dataset."""
    args = parse_args()
    
    # Load dataset
    print(f"Loading dataset from: {args.dataset}")
    with open(args.dataset, "rb") as f:
        data = pickle.load(f)
    
    # Extract components
    cond_data = data["cond_data"]
    target_data = data["target_data"]
    partial_samples = data["partial_samples"]
    config = data.get("config", {})
    
    print(f"Dataset contains {len(target_data)} data points")
    print(f"Each with {partial_samples.shape[1]} partially denoised samples")
    print(f"Spatial dimension: {target_data.shape[1]}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle specific batch index
    if args.batch_idx is not None:
        visualize_batch(args.batch_idx, cond_data, target_data, 
                       partial_samples, args.output_dir, config)
    else:
        # Select random batches to visualize
        num_batches = len(target_data)
        batch_indices = np.random.choice(num_batches, 
                                         size=min(args.num_visualizations, num_batches), 
                                         replace=False)
        
        for batch_idx in batch_indices:
            visualize_batch(batch_idx, cond_data, target_data, 
                           partial_samples, args.output_dir, config)


if __name__ == "__main__":
    main()
