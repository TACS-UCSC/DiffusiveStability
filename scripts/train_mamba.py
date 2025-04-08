#!/usr/bin/env python
# Training script for Mamba KS model

import os
import sys
import argparse
import json
import time
from functools import partial

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from tqdm import tqdm

from src.models.mamba_ks import KSMambaModel
from src.datasets.ks_dataloaders import KSSequenceDataLoader
from src.utils.experiment import setup_experiment_directories


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Mamba model for KS equation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for resuming training")
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def create_model(config, key):
    """Create a new model based on configuration."""
    return KSMambaModel(
        spatial_dim=config["spatial_dim"],
        key=key,
        model_dim=config["model_dim"],
        num_layers=config["num_layers"],
        state_dim=config["state_dim"]
    )


def create_optimizer(config):
    """Create optimizer based on configuration."""
    if config["optimizer"] == "adam":
        return optax.adam(
            learning_rate=config["learning_rate"],
            b1=config.get("beta1", 0.9),
            b2=config.get("beta2", 0.999)
        )
    elif config["optimizer"] == "adamw":
        return optax.adamw(
            learning_rate=config["learning_rate"],
            b1=config.get("beta1", 0.9),
            b2=config.get("beta2", 0.999),
            weight_decay=config.get("weight_decay", 1e-4)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")


def load_checkpoint(checkpoint_path, model, opt_state):
    """Load model and optimizer state from checkpoint."""
    with open(checkpoint_path, "rb") as f:
        checkpoint = eqx.serialization.from_bytes(f.read())
    
    loaded_model = checkpoint["model"]
    loaded_opt_state = checkpoint["opt_state"]
    step = checkpoint["step"]
    
    return loaded_model, loaded_opt_state, step


@partial(jax.jit, static_argnums=(3,))
def compute_loss(model, inputs, targets, loss_type="mse"):
    """Compute loss function for training."""
    predictions = jax.vmap(model)(inputs)
    
    if loss_type == "mse":
        # Mean squared error
        loss = jnp.mean((predictions - targets) ** 2)
    elif loss_type == "mae":
        # Mean absolute error
        loss = jnp.mean(jnp.abs(predictions - targets))
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    return loss


@partial(jax.jit, static_argnums=(3,))
def train_step(model, opt_state, batch, loss_type="mse"):
    """Perform a single training step."""
    inputs, targets = batch
    
    # Define loss function for this batch
    def loss_fn(model):
        return compute_loss(model, inputs, targets, loss_type)
    
    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(model)
    
    # Update parameters
    updates, new_opt_state = optax.apply_updates(opt_state, grads)
    new_model = eqx.apply_updates(model, updates)
    
    return new_model, new_opt_state, loss


def save_checkpoint(model, opt_state, step, save_dir):
    """Save model and optimizer state to checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        "model": model,
        "opt_state": opt_state,
        "step": step
    }
    
    checkpoint_path = os.path.join(save_dir, f"checkpoint_{step}.eqx")
    with open(checkpoint_path, "wb") as f:
        f.write(eqx.serialization.to_bytes(checkpoint))
    
    print(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def evaluate(model, dataloader, loss_type="mse"):
    """Evaluate model on validation set."""
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        inputs, targets = batch
        loss = compute_loss(model, inputs, targets, loss_type)
        total_loss += loss
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create experiment directory
    experiment_dir = setup_experiment_directories({"experiment_name": "mamba_ks"})["exp_dir"]
    
    # Save configuration
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(config.get("seed", 42))
    key, model_key = jax.random.split(key)
    
    # Create model
    model = create_model(config, model_key)
    
    # Create optimizer
    optimizer = create_optimizer(config)
    
    # Initialize optimizer state
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    
    # Resume from checkpoint if provided
    start_step = 0
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        model, opt_state, start_step = load_checkpoint(args.checkpoint, model, opt_state)
        print(f"Resuming training from step {start_step}")
    
    # Create data loaders
    train_loader = KSSequenceDataLoader(
        dataset_file=config["train_dataset"],
        batch_size=config["batch_size"],
        seq_len=config["seq_len"],
        dt=config["dt"],
        shuffle=True,
        seed=config.get("data_seed", 42)
    )
    
    # Check if validation set is specified
    val_loader = None
    if "val_dataset" in config:
        val_loader = KSSequenceDataLoader(
            dataset_file=config["val_dataset"],
            batch_size=config["batch_size"],
            seq_len=config["seq_len"],
            dt=config["dt"],
            shuffle=False,
            seed=config.get("data_seed", 42)
        )
    
    # Training loop
    num_epochs = config["num_epochs"]
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    
    # Determine checkpoint and evaluation frequency
    checkpoint_freq = config.get("checkpoint_freq", steps_per_epoch)
    eval_freq = config.get("eval_freq", steps_per_epoch)
    
    # Logging setup
    log_file = os.path.join(experiment_dir, "training_log.txt")
    with open(log_file, "a") as f:
        f.write(f"Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total epochs: {num_epochs}, Steps per epoch: {steps_per_epoch}\n")
        f.write(f"Config: {json.dumps(config, indent=2)}\n\n")
    
    # Training progress
    step = start_step
    for epoch in range(start_step // steps_per_epoch, num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        # Iterate through batches
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Update model and optimizer
            model, opt_state, loss = train_step(model, opt_state, batch, config.get("loss_type", "mse"))
            
            # Accumulate loss
            epoch_loss += loss
            step += 1
            
            # Evaluate periodically
            if val_loader is not None and step % eval_freq == 0:
                val_loss = evaluate(model, val_loader, config.get("loss_type", "mse"))
                print(f"Step {step}/{total_steps} - Validation loss: {val_loss:.6f}")
                with open(log_file, "a") as f:
                    f.write(f"Step {step}/{total_steps} - Validation loss: {val_loss:.6f}\n")
            
            # Save checkpoint periodically
            if step % checkpoint_freq == 0:
                save_checkpoint(model, opt_state, step, experiment_dir)
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s - Avg. loss: {avg_loss:.6f}")
        
        # Log epoch summary
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s - Avg. loss: {avg_loss:.6f}\n")
    
    # Final checkpoint
    final_checkpoint_path = save_checkpoint(model, opt_state, step, experiment_dir)
    print(f"Training completed. Final checkpoint saved to {final_checkpoint_path}")


if __name__ == "__main__":
    main()