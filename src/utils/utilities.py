"""
Utility functions for diffusion models.
"""
import os
import json
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Union, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.sdes.diffusion import VPSDE, VESDE, subVPSDE


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load experiment configuration from a JSON file.
    
    Args:
        config_path: Path to the config JSON file
        
    Returns:
        Dictionary containing experiment configuration
    """
    try:
        with open(config_path, 'r') as f:
            # Parse JSON with support for comments
            content = f.read()
            # Remove comments
            content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
            config = json.loads(content)
        return config
    except Exception as e:
        raise ValueError(f"Error loading config from {config_path}: {e}")


def setup_experiment_directories(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Create necessary directories for the experiment.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary with paths to various directories
    """
    exp_name = config["experiment_name"]
    base_dir = config.get("output_dir", "experiments")
    exp_dir = os.path.join(base_dir, exp_name)
    
    # Create main directories
    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    plots_dir = os.path.join(exp_dir, "plots")
    results_dir = os.path.join(exp_dir, "results")
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    paths = {
        "exp_dir": exp_dir,
        "checkpoints_dir": checkpoints_dir,
        "plots_dir": plots_dir,
        "results_dir": results_dir
    }
    
    # Save config to experiment directory
    with open(os.path.join(exp_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    return paths


def create_sde(config: Dict[str, Any]):
    """
    Create an SDE instance based on the given configuration.
    
    Args:
        config: Dictionary containing SDE configuration parameters
        
    Returns:
        An SDE instance
    """
    sde_class = config.get("sde_class", "VPSDE")
    schedule_type = config.get("schedule_type", "linear")
    beta_min = config.get("beta_min", 0.1)
    beta_max = config.get("beta_max", 20.0)
    gamma = config.get("gamma", 1.0)
    power = config.get("power", 2.0)
    
    if sde_class == "VPSDE":
        sde = VPSDE(
            beta_min=beta_min, 
            beta_max=beta_max,
            schedule_type=schedule_type,
            gamma=gamma,
            power=power
        )
    elif sde_class == "VESDE":
        sde = VESDE(
            sigma_min=beta_min, 
            sigma_max=beta_max, 
            power=power
        )
    elif sde_class == "subVPSDE":
        sde = subVPSDE(
            beta_min=beta_min, 
            beta_max=beta_max, 
            schedule_type=schedule_type,
            gamma=gamma, 
            power=power
        )
    else:
        raise ValueError(f"Unknown SDE class: {sde_class}")
    
    return sde


def save_loss_plot(loss_history, paths, config):
    """
    Save a plot of the training loss history.
    
    Args:
        loss_history: List of loss values
        paths: Dictionary with experiment paths
        config: Experiment configuration
    """
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(len(loss_history)), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    
    # Add configuration details to title
    title = f"Loss History - {config.get('diffusion_type', 'cont_diffusion')}"
    title += f" ({config.get('data_mode', 'direct')} mode)"
    title += f"\n{config.get('schedule_type', 'linear')} schedule "
    title += f"(β={config.get('beta_min', 0.1)}-{config.get('beta_max', 20.0)}, γ={config.get('gamma', 1.0)})"
    
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    
    # Save plot
    plot_path = os.path.join(paths["plots_dir"], "loss_history.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss plot saved to {plot_path}")