#!/usr/bin/env python
# Script for rolling out predictions with trained Mamba KS model

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import equinox as eqx

from src.models.mamba_ks import KSMambaModel
from src.datasets.ks_dataloaders import KSSequenceDataLoader
from src.utils.visualization import plot_rollout, plot_comparison


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Rollout predictions with trained Mamba KS model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--steps", type=int, default=1000, help="Number of rollout steps")
    parser.add_argument("--output", type=str, default="rollout", help="Output directory for results")
    parser.add_argument("--animate", action="store_true", help="Create animation of the rollout")
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def load_checkpoint(checkpoint_path):
    """Load model from checkpoint."""
    with open(checkpoint_path, "rb") as f:
        checkpoint = eqx.serialization.from_bytes(f.read())
    
    return checkpoint["model"]


def get_initial_state(config):
    """Get initial state from dataset for rollout."""
    # Create a data loader with batch size 1
    data_loader = KSSequenceDataLoader(
        dataset_file=config["test_dataset"],
        batch_size=1,
        seq_len=1,
        dt=1,
        shuffle=False
    )
    
    # Get the first batch
    (inputs, _) = next(iter(data_loader))
    
    # Extract initial state (first element from first sequence)
    initial_state = inputs[0, 0]
    
    return initial_state


def save_rollout_data(rollout_data, output_dir):
    """Save rollout data to NPY file."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "rollout_data.npy"), np.array(rollout_data))
    print(f"Rollout data saved to {os.path.join(output_dir, 'rollout_data.npy')}")


def create_animation(rollout_data, output_dir):
    """Create animation of the rollout."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Initial plot
    line, = ax.plot(rollout_data[0])
    
    # Set title and labels
    ax.set_title('KS Equation Rollout')
    ax.set_xlabel('Spatial dimension')
    ax.set_ylabel('Value')
    
    # Set y-axis limits based on data range
    y_min = np.min(rollout_data) - 0.1
    y_max = np.max(rollout_data) + 0.1
    ax.set_ylim(y_min, y_max)
    
    # Function to update plot for each frame
    def update(frame):
        line.set_ydata(rollout_data[frame])
        ax.set_title(f'KS Equation Rollout - Step {frame}')
        return [line]
    
    # Create animation
    anim = FuncAnimation(
        fig, update, frames=len(rollout_data), 
        interval=50, blit=True
    )
    
    # Save animation
    anim.save(os.path.join(output_dir, 'rollout_animation.mp4'), 
              writer='ffmpeg', dpi=100, fps=30)
    
    plt.close()
    print(f"Animation saved to {os.path.join(output_dir, 'rollout_animation.mp4')}")


def main():
    """Main rollout function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model from checkpoint
    model = load_checkpoint(args.checkpoint)
    print(f"Model loaded from {args.checkpoint}")
    
    # Get initial state
    initial_state = get_initial_state(config)
    print(f"Initial state shape: {initial_state.shape}")
    
    # Perform rollout
    print(f"Performing rollout for {args.steps} steps...")
    rollout_data = model.rollout(initial_state, args.steps)
    print(f"Rollout completed. Shape: {rollout_data.shape}")
    
    # Save rollout data
    save_rollout_data(rollout_data, output_dir)
    
    # Create static visualization
    fig = plot_rollout(rollout_data.astype(np.float32))
    fig.savefig(os.path.join(output_dir, "rollout_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap visualization saved to {os.path.join(output_dir, 'rollout_heatmap.png')}")
    
    # Plot first, middle, and last frames
    indices = [0, args.steps // 2, args.steps]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, idx in enumerate(indices):
        axes[i].plot(rollout_data[idx])
        axes[i].set_title(f"Step {idx}")
        axes[i].set_xlabel("Spatial dimension")
        if i == 0:
            axes[i].set_ylabel("Value")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rollout_snapshots.png"), dpi=300)
    plt.close()
    print(f"Snapshot visualization saved to {os.path.join(output_dir, 'rollout_snapshots.png')}")
    
    # Create animation if requested
    if args.animate:
        print("Creating animation...")
        create_animation(rollout_data, output_dir)


if __name__ == "__main__":
    main()