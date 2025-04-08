import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial
import numpy as np

def get_solver(config, model, sde):
    """Select solver based on configuration"""
    solver_type = config.get("solver_type", "sde")
    
    if solver_type == "ode_flow":
        from src.solvers.ode_solvers import probability_flow_sampler
        return probability_flow_sampler(model, sde, config)
    else:
        # Use existing SDE-based solver
        return reverseSDE_solve

@eqx.filter_jit
def euler_maruyama_step(x, score_fn, sde, t, dt, key):
    """
    Single step of the Euler-Maruyama algorithm for SDE integration.
    
    Args:
        x: Current state
        score_fn: Score function (gradient of log likelihood)
        sde: SDE object with drift and diffusion methods
        t: Current time
        dt: Time step size
        key: Random key
        
    Returns:
        Updated state
    """
    drift = -sde.drift(x, t) - sde.diffusion(x, t)**2 * score_fn(x, t, key)
    diffusion = sde.diffusion(x, t)
    
    noise = jax.random.normal(key, shape=x.shape)
    x_next = x + drift * dt + diffusion * jnp.sqrt(jnp.abs(dt)) * noise
    
    return x_next


def generate_time_steps(schedule_type, num_steps=1000, t_min=1e-5, t_max=1.0, power=2):
    """
    Generate time steps for SDE integration according to specified schedule.
    
    Args:
        schedule_type: Type of schedule ('uniform', 'quadratic', 'power')
        num_steps: Number of time steps
        t_min: Minimum time value (typically close to 0)
        t_max: Maximum time value (typically 1.0)
        power: Power for the power schedule
    
    Returns:
        Array of time steps in descending order (from t_max to t_min)
    """
    if schedule_type == 'uniform':
        # Uniform time steps
        return jnp.linspace(t_max, t_min, num_steps)
    
    elif schedule_type == 'quadratic':
        # Quadratic schedule - more steps near t=0
        t = jnp.linspace(0, 1, num_steps)
        t = t_max * (1 - t**2) + t_min * t**2
        return t_max - t + t_min
    
    elif schedule_type == 'power':
        # Power schedule with adjustable power
        t = jnp.linspace(0, 1, num_steps)
        t = t_max * (1 - t**power) + t_min * t**power
        return t_max - t + t_min
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def create_predictor_fn(score_fn, sde, num_steps=1000, schedule_type='uniform'):
    """
    Create a predictor function that integrates the SDE backward in time.
    
    Args:
        score_fn: Score function (gradient of log likelihood)
        sde: SDE object with drift and diffusion methods
        num_steps: Number of integration steps
        schedule_type: Type of time step schedule
    
    Returns:
        Predictor function that takes (x_t, t, key) and returns x_0
    """
    # Generate time steps
    time_steps = generate_time_steps(schedule_type, num_steps)
    
    @jax.jit
    def predictor(x_t, t, key):
        """
        Integrate SDE backward from time t to time 0.
        
        Args:
            x_t: Current state at time t
            t: Current time
            key: Random key
            
        Returns:
            Predicted state at time 0
        """
        # Find the closest time step to t
        idx = jnp.argmin(jnp.abs(time_steps - t))
        
        # Initialize state
        x = x_t
        
        # Loop over time steps from idx to end
        for i in range(idx, len(time_steps) - 1):
            t_cur = time_steps[i]
            t_next = time_steps[i + 1]
            dt = t_next - t_cur
            
            # Generate a new key
            key, step_key = jax.random.split(key)
            
            # Take one step
            x = euler_maruyama_step(x, score_fn, sde, t_cur, dt, step_key)
        
        return x
    
    return predictor


def create_conditional_sampler(model, sde, num_steps=1000, schedule_type='uniform'):
    """
    Create a conditional sampler function for the diffusion model.
    
    Args:
        model: The diffusion model
        sde: SDE object
        num_steps: Number of integration steps
        schedule_type: Type of time step schedule
        
    Returns:
        Sampler function that takes conditional data and returns a sample
    """
    # Generate time steps
    time_steps = generate_time_steps(schedule_type, num_steps)
    
    # Create time step pairs for scan
    time_step_pairs = jnp.stack([time_steps[:-1], time_steps[1:]], axis=1)
    
    @eqx.filter_jit
    def conditional_score_fn(x, cond_data, t, key):
        """
        Compute the score function using the conditional model.
        
        Args:
            x: Current state
            cond_data: Conditional data
            t: Current time
            key: Random key
            
        Returns:
            Score value
        """
        return model(x, cond_data, t, key)
    
    def sample_fn(cond_data, key):
        """
        Generate a sample conditioned on cond_data.
        
        Args:
            cond_data: Conditional data
            key: Random key
            
        Returns:
            Generated sample
        """
        # Generate initial noise
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, shape=cond_data[0].shape)
        
        # Define a single Euler-Maruyama step for the scan function
        def euler_maruyama_scan_step(carry, time_step):
            xt, key, cond_state = carry
            t_cur, t_next = time_step

            # Generate new key for each step
            key, score_key, noise_key = jax.random.split(key, 3)
            
            # Compute score
            score = conditional_score_fn(xt, cond_state, t_cur, score_key)
            
            # Compute drift using the learned score function
            drift = -sde.drift(xt, t_cur) - sde.diffusion(xt, t_cur)**2 * score
            diffusion = sde.diffusion(xt, t_cur)

            # Compute time step
            dt = t_next - t_cur

            # Generate random noise
            noise = jax.random.normal(noise_key, shape=xt.shape)

            # Update using Euler-Maruyama
            next_xt = xt + drift * dt + diffusion * jnp.sqrt(jnp.abs(dt)) * noise

            return (next_xt, key, cond_state), next_xt
        
        # Initialize carry
        initial_carry = (x, key, cond_data)
        
        # Run the scan function
        (final_x, _, _), trajectory = jax.lax.scan(
            euler_maruyama_scan_step, initial_carry, time_step_pairs
        )
        
        return final_x, trajectory
    
    return sample_fn


def create_autoregressive_predictor(model, sde, num_steps=1000, schedule_type='uniform'):
    """
    Create an autoregressive predictor for the conditional diffusion model.
    
    Args:
        model: Conditional diffusion model
        sde: SDE object
        num_steps: Number of SDE integration steps
        schedule_type: Type of time step schedule
        
    Returns:
        Autoregressive predictor function
    """
    # Generate time steps
    time_steps = generate_time_steps(schedule_type, num_steps)
    
    # Create time step pairs for scan
    time_step_pairs = jnp.stack([time_steps[:-1], time_steps[1:]], axis=1)
    
    @eqx.filter_jit
    def predict_next_step(cond_window, key):
        """
        Predict the next step given the conditional window.
        
        Args:
            cond_window: Conditional window data
            key: Random key
            
        Returns:
            Predicted next step
        """
        # Generate noisy initial sample
        key, noise_key = jax.random.split(key)
        initial_sample = sde.forward_sample(cond_window[0], 1.0, noise_key)
        
        # Define the Euler-Maruyama step function
        def euler_maruyama_scan_step(carry, time_step):
            xt, key, cond_state = carry
            t_cur, t_next = time_step

            # Generate new key for each step
            key, score_key, noise_key = jax.random.split(key, 3)
            
            # Compute score
            score = model(xt, cond_state, t_cur, score_key)
            
            # Compute drift
            drift = -sde.drift(xt, t_cur) - sde.diffusion(xt, t_cur)**2 * score
            diffusion = sde.diffusion(xt, t_cur)

            # Compute time step
            dt = t_next - t_cur

            # Generate random noise
            noise = jax.random.normal(noise_key, shape=xt.shape)

            # Update using Euler-Maruyama
            next_xt = xt + drift * dt + diffusion * jnp.sqrt(jnp.abs(dt)) * noise

            return (next_xt, key, cond_state), next_xt
        
        # Initialize carry
        initial_carry = (initial_sample, key, cond_window)
        
        # Run the scan
        (final_sample, key, _), _ = jax.lax.scan(
            euler_maruyama_scan_step, initial_carry, time_step_pairs
        )
        
        return final_sample, key
    
    def predict_n_steps(cond_data, n_steps, key=None):
        """
        Predict n steps ahead from the conditional data.
        
        Args:
            cond_data: Initial conditional data
            n_steps: Number of steps to predict
            key: Random key (optional)
            
        Returns:
            Array of predictions including the initial data
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # Initialize with conditional data
        cond_window = cond_data[0]  # Shape [cond_steps, spatial_dim]
        all_steps = jnp.stack([cond_window[i] for i in range(cond_window.shape[0])], axis=0)
        
        # Loop through and predict each step
        for i in range(n_steps):
            # Predict next step
            next_step, key = predict_next_step(cond_window, key)
            
            # Update conditional window (roll forward)
            if cond_window.shape[0] > 1:
                # For multi-step conditioning
                new_cond = jnp.concatenate([
                    jnp.expand_dims(next_step, axis=0),
                    cond_window[:-1]
                ], axis=0)
            else:
                # For single-step conditioning
                new_cond = jnp.expand_dims(next_step, axis=0)
                
            cond_window = new_cond
            
            # Add to collection of steps
            all_steps = jnp.concatenate([all_steps, jnp.expand_dims(next_step, axis=0)], axis=0)
        
        return all_steps
    
    return predict_n_steps

def euler_maruyama_solver(model_fn, y0, ts, rngkey):
    """
    Euler-Maruyama solver for stochastic differential equations.
    
    Args:
        model_fn: Function that computes the drift and diffusion.
        y0: Initial state.
        ts: Time points.
        rngkey: JAX random key.
        
    Returns:
        Solution trajectory.
    """
    def step_fn(y, args):
        t, dt, key = args
        drift, diffusion = model_fn(t, y)
        noise = jax.random.normal(key, y.shape)
        y_next = y + drift * dt + diffusion * jnp.sqrt(dt) * noise
        return y_next, y_next
    
    # Create time steps and key splits
    ts_pairs = jnp.stack([ts[:-1], ts[1:]], axis=1)
    dts = ts_pairs[:, 1] - ts_pairs[:, 0]
    keys = jax.random.split(rngkey, len(dts))
    
    args = (ts_pairs[:, 0], dts, keys)
    _, ys = jax.lax.scan(step_fn, y0, args)
    
    # Include initial state
    ys = jnp.concatenate([y0[None], ys], axis=0)
    return ys

def reverseSDE_solve(solver, sde, score_fn, y, t0, dt, timesteps, rngkey, return_all=False):
    """
    Solve the reverse SDE to generate samples.
    
    Args:
        solver: Solver function to use (e.g., euler_maruyama_solver).
        sde: SDE object with drift and diffusion methods.
        score_fn: Score function.
        y: Initial state.
        t0: Initial time.
        dt: Time step size.
        timesteps: Number of time steps.
        rngkey: JAX random key.
        return_all: Whether to return all intermediate states.
        
    Returns:
        Final state or trajectory of states.
    """
    def drift_diffusion_fn(t, y):
        """Combined drift and diffusion function for the solver."""
        # Get the score for the current state and time
        key = jax.random.fold_in(rngkey, int(t * 1000))
        score = score_fn(t, y)
        
        # Compute the reverse SDE drift (using score)
        drift = -sde.drift(y, t) - sde.diffusion(y, t)**2 * score
        diffusion = sde.diffusion(y, t)
        
        return drift, diffusion
    
    # Generate time sequence
    ts = jnp.linspace(t0, t0 - timesteps * dt, timesteps + 1)
    
    # Solve the SDE
    traj = solver(drift_diffusion_fn, y, ts, rngkey)
    
    if return_all:
        return traj
    else:
        return traj[-1]  # Return only the final state