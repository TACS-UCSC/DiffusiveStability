#!/usr/bin/env python
# Script for rolling out predictions with trained MambaSeqToSeq model

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import equinox as eqx

from src.models.ks import MambaSeqToSeq
from src.datasets.ks_dataloaders import KSSequenceDataLoader
from src.utils.visualization import plot_rollout, plot_comparison


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run rollout with MambaSeqToSeq model")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (optional)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset file")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to roll out")
    parser.add_argument("--output_dir", type=str, default="results/mamba_seq2seq", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_model(model_path: str) -> MambaSeqToSeq:
    """Load model from file."""
    with open(model_path, "rb") as f:
        model = eqx.serialization.from_bytes(f.read())
    return model


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def get_initial_state(dataset_path: str, seq_len: int, dt: int) -> Tuple[jax.Array, jax.Array]:
    """Get initial state and ground truth sequence from dataset."""
    # Create a data loader with batch size 1
    data_loader = KSSequenceDataLoader(
        dataset_file=dataset_path,
        batch_size=1,
        seq_len=seq_len,
        dt=dt,
        shuffle=False
    )
    
    # Get the first batch
    inputs, targets = next(iter(data_loader))
    
    # Extract initial state (the first element of the first sequence)
    initial_input = inputs[0]
    ground_truth = targets[0]
    
    return initial_input, ground_truth


def autoregressive_rollout(model: MambaSeqToSeq, initial_state: jax.Array, 
                          num_steps: int, spatial_dim: int) -> jax.Array:
    """Generate a sequence by rolling out the model autoregressively.
    
    Args:
        model: The MambaSeqToSeq model
        initial_state: Initial sequence with shape (seq_len, spatial_dim)
        num_steps: Number of steps to predict
        spatial_dim: Spatial dimension
        
    Returns:
        Predicted sequence with shape (seq_len + num_steps, spatial_dim)
    """
    # Initialize cache for the model
    cache = model.init_cache()
    
    # Start with the initial input sequence
    seq_len = initial_state.shape[0]
    sequence = [state for state in initial_state]
    
    # Get the last state from the input sequence
    current_state = initial_state[-1]
    
    # Generate steps autoregressively
    for _ in range(num_steps):
        # Generate next state
        next_state, cache = model.generate_step(current_state, cache)
        
        # Add to sequence
        sequence.append(next_state)
        
        # Update current state
        current_state = next_state
    
    # Stack sequence into a single array
    return jnp.stack(sequence)


def generate_plots(rollout_data: jax.Array, ground_truth: jax.Array, output_dir: str):
    """Generate and save visualizations of the rollout."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy for plotting
    rollout_np = np.array(rollout_data)
    
    # Plot heatmap of the entire rollout
    fig = plot_rollout(rollout_np)
    plt.savefig(os.path.join(output_dir, "rollout_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved to {os.path.join(output_dir, 'rollout_heatmap.png')}")
    
    # If ground truth is available, plot comparison
    if ground_truth is not None:
        # Use the common length for comparison
        common_length = min(rollout_np.shape[0], ground_truth.shape[0])
        fig = plot_comparison(
            rollout_np[:common_length], 
            np.array(ground_truth[:common_length]), 
            title1="Model Prediction", 
            title2="Ground Truth"
        )
        plt.savefig(os.path.join(output_dir, "prediction_vs_ground_truth.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Comparison plot saved to {os.path.join(output_dir, 'prediction_vs_ground_truth.png')}")
    
    # Plot snapshots at different timesteps
    if ground_truth is not None:
        num_steps = rollout_np.shape[0]
        indices = [0, num_steps // 4, num_steps // 2, 3 * num_steps // 4, num_steps - 1]
        fig, axes = plt.subplots(len(indices), 2, figsize=(12, 3 * len(indices)))
        
        for i, idx in enumerate(indices):
            if idx < rollout_np.shape[0] and idx < ground_truth.shape[0]:
                axes[i, 0].plot(rollout_np[idx])
                axes[i, 0].set_title(f"Prediction at step {idx}")
                
                axes[i, 1].plot(ground_truth[idx])
                axes[i, 1].set_title(f"Ground Truth at step {idx}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "snapshots.png"), dpi=300)
        plt.close(fig)
        print(f"Snapshot comparison saved to {os.path.join(output_dir, 'snapshots.png')}")
    
    # Save the raw data for further analysis
    np.save(os.path.join(output_dir, "rollout_data.npy"), rollout_np)
    print(f"Rollout data saved to {os.path.join(output_dir, 'rollout_data.npy')}")
    
    if ground_truth is not None:
        np.save(os.path.join(output_dir, "ground_truth.npy"), np.array(ground_truth))
        print(f"Ground truth data saved to {os.path.join(output_dir, 'ground_truth.npy')}")


def main():
    """Main function for model rollout."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    key = jax.random.PRNGKey(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    
    # Load configuration
    config = None
    if args.config:
        print(f"Loading configuration from {args.config}...")
        config = load_config(args.config)
    else:
        # Try to load config from the same directory as the model
        config_path = os.path.join(os.path.dirname(args.model), "model_config.json")
        if os.path.exists(config_path):
            print(f"Loading configuration from {config_path}...")
            config = load_config(config_path)
        else:
            print("No configuration file found. Using default values.")
            config = {
                "spatial_dim": 1024,
                "seq_len": 100,
                "dt": 100
            }
    
    # Determine dataset path
    dataset_path = args.dataset
    if dataset_path is None and config and "train_dataset" in config:
        dataset_path = config["train_dataset"]
    
    # Get initial state and ground truth if dataset is available
    initial_input = None
    ground_truth = None
    if dataset_path and os.path.exists(dataset_path):
        print(f"Loading initial state from {dataset_path}...")
        seq_len = config.get("seq_len", 100)
        dt = config.get("dt", 100)
        initial_input, ground_truth = get_initial_state(dataset_path, seq_len, dt)
    else:
        print("No dataset specified or found. Generating random initial state...")
        spatial_dim = config.get("spatial_dim", 1024)
        seq_len = config.get("seq_len", 100)
        key, subkey = jax.random.split(key)
        initial_input = jax.random.normal(subkey, (seq_len, spatial_dim))
    
    # Run rollout
    print(f"Performing rollout for {args.steps} steps...")
    rollout_data = autoregressive_rollout(
        model=model,
        initial_state=initial_input,
        num_steps=args.steps,
        spatial_dim=config.get("spatial_dim", 1024)
    )
    print(f"Rollout complete. Shape: {rollout_data.shape}")
    
    # Generate plots
    print("Generating visualizations...")
    generate_plots(rollout_data, ground_truth, args.output_dir)
    
    print(f"Rollout results saved to {args.output_dir}")


if __name__ == "__main__":
    main()