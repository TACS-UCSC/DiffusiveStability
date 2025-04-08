# ode_solvers.py - Implementation of probability flow ODE solvers
import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial

def score_to_ode_fn(score_fn, sde):
    """Convert a score function to the probability flow ODE function"""
    def ode_fn(t, y):
        # Get drift and diffusion coefficients from SDE
        drift = sde.drift(y, t)
        diffusion = sde.diffusion(y, t)
        
        # Compute score using the model
        score = score_fn(t, y)
        
        # Probability flow ODE: dx/dt = f(x,t) - 0.5 * g(t)^2 * score
        dydt = drift - 0.5 * diffusion**2 * score
        
        return dydt
    
    return ode_fn

@eqx.filter_jit
def solve_ode(ode_fn, y0, t_range, steps, method="rk4"):
    """Solve ODE using RK4 with JAX's scan for efficiency"""
    t0, t1 = t_range
    dt = (t1 - t0) / steps
    
    # Generate time points (excluding the final point)
    ts = jnp.linspace(t0, t1, steps)
    
    # Define a single RK4 step for the scan operation
    def rk4_scan_step(y, t):
        # Standard RK4 update
        k1 = ode_fn(t, y)
        k2 = ode_fn(t + dt/2, y + dt*k1/2)
        k3 = ode_fn(t + dt/2, y + dt*k2/2)
        k4 = ode_fn(t + dt, y + dt*k3)
        
        next_y = y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        return next_y, None
    
    # Use JAX's scan for efficient iteration
    final_y, _ = jax.lax.scan(rk4_scan_step, y0, ts)
    return final_y

@eqx.filter_jit
def probability_flow_sampling(model, sde, cond_data, x_t, key, config):
    """Generate a sample using probability flow ODE with efficient JAX implementation"""
    # Get solver configuration
    solver_config = config.get("ode_solver_config", {})
    steps = solver_config.get("steps", 100)
    
    # Define score function with fixed conditional data
    def score_fn(t, x):
        return model(x, cond_data, t, key)
    
    # Convert to ODE function
    ode_fn = score_to_ode_fn(score_fn, sde)
    
    # Solve from t=1 (noise) to t=0 (clean data)
    return solve_ode(ode_fn, x_t, (1.0, 0.0), steps)
