#!/usr/bin/env python
# Training script for MambaSeqToSeq model using the KSSequenceDataLoader for denoising tasks

import os
import sys
import json
import time
import argparse
from functools import partial
from typing import Dict, Any, Optional, Tuple

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from tqdm import tqdm

from src.models.ks import MambaSeqToSeq
from src.datasets.ks_dataloaders import KSSequenceDataLoader
from src.utils.experiment import setup_experiment_directories


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MambaSeqToSeq model for KS denoising")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for resuming training")
    parser.add_argument("--output_dir", type=str, default="experiments/mamba_seq2seq", help="Output directory")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def create_model(config: Dict[str, Any], key: jax.random.PRNGKey) -> MambaSeqToSeq:
    """Create a new MambaSeqToSeq model based on configuration."""
    return MambaSeqToSeq(
        spatial_dim=config.get("spatial_dim", 1024),
        hidden_dim=config.get("hidden_dim", 256),
        n_layers=config.get("n_layers", 4),
        state_dim=config.get("state_dim", 16),
        kernel_size=config.get("kernel_size", 4),
        expand=config.get("expand", 2),
        dt_rank=config.get("dt_rank", "auto"),
        dtype=jnp.float32,
        key=key
    )


def create_optimizer(config: Dict[str, Any]) -> optax.GradientTransformation:
    """Create optimizer based on configuration."""
    learning_rate = config.get("learning_rate", 1e-4)
    weight_decay = config.get("weight_decay", 1e-5)
    
    # Create AdamW optimizer
    return optax.adamw(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        b1=config.get("beta1", 0.9),
        b2=config.get("beta2", 0.999),
        eps=config.get("epsilon", 1e-8)
    )


def create_data_loaders(config: Dict[str, Any]) -> Tuple[KSSequenceDataLoader, Optional[KSSequenceDataLoader]]:
    """Create data loaders based on configuration."""
    # Training data loader
    train_loader = KSSequenceDataLoader(
        dataset_file=config["train_dataset"],
        batch_size=config["batch_size"],
        seq_len=config.get("seq_len", 100),
        dt=config.get("dt", 100),
        shuffle=True,
        seed=config.get("data_seed", 42)
    )
    
    # Validation data loader (optional)
    val_loader = None
    if "val_dataset" in config:
        val_loader = KSSequenceDataLoader(
            dataset_file=config["val_dataset"],
            batch_size=config["batch_size"],
            seq_len=config.get("seq_len", 100),
            dt=config.get("dt", 100),
            shuffle=False,
            seed=config.get("data_seed", 42)
        )
    
    return train_loader, val_loader


def load_checkpoint(checkpoint_path: str) -> Tuple[MambaSeqToSeq, optax.OptState, int]:
    """Load model and optimizer state from checkpoint."""
    # Load checkpoint using tree_deserialise_leaves
    checkpoint = eqx.tree_deserialise_leaves(checkpoint_path)
    
    model = checkpoint["model"]
    opt_state = checkpoint["opt_state"]
    step = checkpoint["step"]
    
    print(f"Loaded checkpoint from {checkpoint_path} at step {step}")
    return model, opt_state, step


def save_checkpoint(model: MambaSeqToSeq, opt_state: optax.OptState, step: int, 
                   output_dir: str, config: Dict[str, Any]) -> str:
    """Save model, optimizer state, and config to checkpoint."""
    # Create checkpoint directory if it doesn't exist
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Create checkpoint dictionary
    checkpoint = {
        "model": model,
        "opt_state": opt_state,
        "step": step,
        "config": config
    }
    
    # Use eqx.tree_serialise_leaves for checkpointing
    checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_{step}.eqx")
    eqx.tree_serialise_leaves(checkpoint_path, checkpoint)
    
    # Also save the latest model separately
    model_path = os.path.join(output_dir, "model.eqx")
    eqx.tree_serialise_leaves(model_path, model)
    
    # Save config for easy model recreation
    config_path = os.path.join(output_dir, "model_config.json")
    model_config = {
        "spatial_dim": config.get("spatial_dim", 1024),
        "hidden_dim": config.get("hidden_dim", 256),
        "n_layers": config.get("n_layers", 4),
        "state_dim": config.get("state_dim", 16),
        "kernel_size": config.get("kernel_size", 4),
        "expand": config.get("expand", 2),
        "dt_rank": config.get("dt_rank", "auto"),
    }
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    
    print(f"Checkpoint saved to {checkpoint_path}")
    print(f"Model saved to {model_path}")
    print(f"Model config saved to {config_path}")
    
    return checkpoint_path


@eqx.filter_jit
def compute_loss(model: MambaSeqToSeq, inputs: jax.Array, targets: jax.Array) -> jax.Array:
    """Compute MSE loss between model outputs and targets.
    
    Args:
        model: The MambaSeqToSeq model
        inputs: Input noisy sequences with shape (batch_size, seq_len, spatial_dim)
        targets: Target clean sequences with shape (batch_size, seq_len, spatial_dim)
        
    Returns:
        MSE loss value
    """
    # Transpose batch and sequence dimensions for vmap compatibility
    # From (batch_size, seq_len, spatial_dim) to (seq_len, batch_size, spatial_dim)
    inputs = jnp.transpose(inputs, (1, 0, 2))
    targets = jnp.transpose(targets, (1, 0, 2))
    
    # Apply model to each sequence element across the batch
    # This vmaps over the batch dimension (dim 1)
    predictions = jax.vmap(model, in_axes=1, out_axes=1)(inputs)
    
    # Transpose back to original shape
    # From (seq_len, batch_size, spatial_dim) to (batch_size, seq_len, spatial_dim)
    predictions = jnp.transpose(predictions, (1, 0, 2))
    targets = jnp.transpose(targets, (1, 0, 2))
    
    # Compute MSE loss
    return jnp.mean((predictions - targets) ** 2)


@eqx.filter_jit
def train_step(model: MambaSeqToSeq, opt_state: optax.OptState, 
              inputs: jax.Array, targets: jax.Array, optimizer: optax.GradientTransformation) -> Tuple[MambaSeqToSeq, optax.OptState, jax.Array]:
    """Perform a single training step.
    
    Args:
        model: The MambaSeqToSeq model
        opt_state: Current optimizer state
        inputs: Input noisy sequences
        targets: Target clean sequences
        optimizer: The optimizer to use
        
    Returns:
        Updated model, optimizer state, and loss value
    """
    # Define loss function for this batch
    def loss_fn(model):
        return compute_loss(model, inputs, targets)
    
    # Compute loss and gradients
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    
    # Update parameters
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    
    return new_model, new_opt_state, loss


@eqx.filter_jit
def evaluate_batch(model: MambaSeqToSeq, inputs: jax.Array, targets: jax.Array) -> jax.Array:
    """Evaluate model on a single batch."""
    return compute_loss(model, inputs, targets)


def evaluate(model: MambaSeqToSeq, dataloader: KSSequenceDataLoader) -> float:
    """Evaluate model on the entire dataset."""
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        inputs, targets = batch
        loss = evaluate_batch(model, inputs, targets)
        total_loss += loss
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup experiment directories
    exp_config = {
        "experiment_name": config.get("experiment_name", "mamba_seq2seq"),
        "output_dir": args.output_dir
    }
    paths = setup_experiment_directories(exp_config)
    output_dir = paths["exp_dir"]
    
    # Save the full config for reference
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Set random seed for reproducibility
    seed = config.get("seed", 42)
    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config)
    
    # Initialize or load model
    print("Initializing model...")
    if args.checkpoint:
        model, opt_state, start_step = load_checkpoint(args.checkpoint)
    else:
        model = create_model(config, model_key)
        optimizer = create_optimizer(config)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        start_step = 0
    
    # Create optimizer if not loaded from checkpoint
    if args.checkpoint is None:
        optimizer = create_optimizer(config)
    else:
        optimizer = create_optimizer(config)  # Recreate optimizer with potentially updated config
    
    # Training parameters
    num_epochs = config.get("num_epochs", 100)
    checkpoint_freq = config.get("checkpoint_freq", 10)
    eval_freq = config.get("eval_freq", 1)
    
    # Calculate steps per epoch
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    
    # Initialize TensorBoard writer if available
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        use_tensorboard = True
    except ImportError:
        use_tensorboard = False
        print("TensorBoard not available. Training metrics will be logged to text file only.")
    
    # Setup logging
    log_file = os.path.join(output_dir, "training_log.txt")
    with open(log_file, "a") as f:
        f.write(f"Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total epochs: {num_epochs}, Steps per epoch: {steps_per_epoch}\n")
        f.write(f"Model configuration: {json.dumps(config, indent=2)}\n\n")
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    global_step = start_step
    
    for epoch in range(start_step // steps_per_epoch, num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        # Training loop
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                # Get inputs and targets from batch
                inputs, targets = batch
                
                # Perform one training step
                model, opt_state, loss = train_step(model, opt_state, inputs, targets, optimizer)
                
                # Update metrics
                epoch_loss += loss
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({"loss": float(loss)})
                
                # Log metrics
                if use_tensorboard:
                    writer.add_scalar("training/batch_loss", float(loss), global_step)
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / steps_per_epoch
        
        # Evaluate on validation set if available
        val_loss = None
        if val_loader is not None and (epoch + 1) % eval_freq == 0:
            val_loss = evaluate(model, val_loader)
            print(f"Epoch {epoch+1}/{num_epochs} - Validation loss: {val_loss:.6f}")
            
            # Log validation metrics
            if use_tensorboard:
                writer.add_scalar("validation/loss", float(val_loss), global_step)
        
        # Log epoch results
        epoch_time = time.time() - epoch_start_time
        log_message = f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s - Training loss: {avg_epoch_loss:.6f}"
        if val_loss is not None:
            log_message += f" - Validation loss: {val_loss:.6f}"
        print(log_message)
        
        with open(log_file, "a") as f:
            f.write(log_message + "\n")
        
        # Log to TensorBoard
        if use_tensorboard:
            writer.add_scalar("training/epoch_loss", float(avg_epoch_loss), epoch + 1)
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0 or epoch == num_epochs - 1:
            save_checkpoint(model, opt_state, global_step, output_dir, config)
    
    # Save final model
    final_checkpoint_path = save_checkpoint(model, opt_state, global_step, output_dir, config)
    print(f"Training completed. Final model saved to {output_dir}")
    
    # Close TensorBoard writer if used
    if use_tensorboard:
        writer.close()


if __name__ == "__main__":
    main()