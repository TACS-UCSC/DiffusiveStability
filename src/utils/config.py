"""
Configuration utilities for diffusion models.
"""
import json
import re
import os
from typing import Dict, Any, Optional

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


def create_sde(config):
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