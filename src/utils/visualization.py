"""
Visualization utilities for diffusion models.
"""
import os
import numpy as np
import matplotlib.pyplot as plt


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