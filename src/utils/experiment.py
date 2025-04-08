"""
Experiment utilities for diffusion models.
"""
import os
import json
from typing import Dict, Any, Optional, Union, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.ks import ksModel1d, ksModelConditional1d
from src.models.ssw import SSWConditionalModel
from src.sdes.diffusion import VPSDE, VESDE, subVPSDE


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


def create_model(config, key, dataset="ks"):
    """
    Create a model instance based on the configuration.
    
    Args:
        config: Experiment configuration
        key: JAX random key
        dataset: Dataset type (ks or ssw)
        
    Returns:
        A model instance
    """
    data_mode = config.get("data_mode", "direct")
    
    if dataset == "ks":
        if data_mode.startswith("conditional"):
            # Use conditional model for all conditional modes
            return ksModelConditional1d(
                spatial_dim=1024,
                hidden_dim=config.get("hidden_dim", 64),
                num_heads=config.get("num_heads", 8),
                phead_scale=1,
                num_layers=config.get("num_layers", 4),
                key=key,
                marginal_prob_std=None,
                cond_states=config.get("condition_steps", 2)
            )
        else:
            # Use standard model for direct or delta modes
            return ksModel1d(
                spatial_dim=1024,
                hidden_dim=config.get("hidden_dim", 64),
                num_heads=config.get("num_heads", 8),
                num_layers=config.get("num_layers", 4),
                key=key,
                marginal_prob_std=None
            )
    elif dataset == "ssw":
        # Use SSW model for SSW dataset
        return SSWConditionalModel(
            spatial_dim=config.get("spatial_dim", 75),
            hidden_dim=config.get("hidden_dim", 128),
            num_heads=config.get("num_heads", 8),
            phead_scale=1,
            num_layers=config.get("num_layers", 4),
            key=key,
            marginal_prob_std=None,
            cond_states=config.get("condition_steps", 2)
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_model_from_checkpoint(config, checkpoint_path, key, dataset="ks"):
    """
    Load a model from a checkpoint file.
    
    Args:
        config: Experiment configuration
        checkpoint_path: Path to the checkpoint file
        key: JAX random key
        dataset: Dataset type (ks or ssw)
        
    Returns:
        The loaded model
    """
    # Create empty model with the same architecture
    model = create_model(config, key, dataset)
    
    # Load weights from checkpoint
    try:
        model = eqx.tree_deserialise_leaves(checkpoint_path, model)
        print(f"Loaded model weights from {checkpoint_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {checkpoint_path}: {e}")
        print("Using untrained model - results will be random")
        return model