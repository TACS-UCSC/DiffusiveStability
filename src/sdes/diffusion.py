import jax.numpy as jnp
import jax
import numpy as np
import equinox as eqx


class VPSDE(eqx.Module):
    beta_min: float
    beta_max: float
    T: float = 1.0
    schedule_type: str = "linear"
    gamma: float = 1.0
    power: float = 3.1
    
    def __init__(self, beta_min=0.1, beta_max=25.0, T=1.0, schedule_type="linear", gamma=1.0, power=3.1):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.schedule_type = schedule_type
        self.gamma = gamma
        self.power = power

    def beta(self, t):
        """Beta schedule function."""
        if self.schedule_type == "linear":
            return self.beta_min + t * (self.beta_max - self.beta_min)
        elif self.schedule_type == "cosine":
            cos_t = jnp.cos((t * jnp.pi) / 2)
            normalized = 1 - (cos_t ** 2)  # Goes from 0 to 1
            return self.beta_min + normalized * (self.beta_max - self.beta_min)
        elif self.schedule_type == "power":
            return self.beta_min + (self.beta_max - self.beta_min) * (t ** self.power)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def beta_integral(self, t):
        """Analytical integral of beta from 0 to t."""
        if self.schedule_type == "linear":
            # For linear schedule: ∫(beta_min + s*(beta_max-beta_min))ds from 0 to t
            return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        
        elif self.schedule_type == "cosine":
            # For cosine schedule: ∫(beta_min + (1-cos²(πs/2))*(beta_max-beta_min))ds from 0 to t
            return self.beta_min * t + (self.beta_max - self.beta_min) * (t - (2/jnp.pi) * jnp.sin(jnp.pi * t / 2))
        elif self.schedule_type == "power":
            # Integral of t^p from 0 to t
            return self.beta_min * t + (self.beta_max - self.beta_min) * (t ** (self.power + 1)) / (self.power + 1)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def alpha(self, t):
        """Alpha function: exp(-0.5 * beta_integral)."""
        return jnp.exp(-0.5 * self.beta_integral(t))

    def drift(self, x, t):
        """Drift term for the SDE."""
        return -0.5 * self.beta(t) * x

    def diffusion(self, x, t):
        """Diffusion coefficient for the SDE."""
        return jnp.sqrt(self.beta(t)) * self.gamma

    def marginal_prob(self, x, t):
        """Marginal probability parameters for data at time t."""
        mean = self.alpha(t) * x
        #std = jnp.max(jnp.sqrt(1.0 - self.alpha(t)**2) * self.gamma, jnp.sqrt(1e-6))
        std = jnp.sqrt(1.0 - self.alpha(t)**2) * self.gamma
        return mean, std
    
    def marginal_prob_std(self, t):
        """Get standard deviation for marginal probability distribution."""
        return jnp.sqrt(1.0 - self.alpha(t)**2) * self.gamma

    def prior_sampling(self, rng, shape):
        """Sample from the prior distribution."""
        _, std = self.marginal_prob(jnp.zeros(shape), self.T)
        return jax.random.normal(rng, shape)*std

    def forward_sample(self, x, t, rng):
        """
        Generate noisy data at time t for a batch of clean data x.
        
        Args:
            x: Clean input data.
            t: Noise level, can be a scalar or a batch of times.
            rng: JAX random key.
        Returns:
            Noisy data sample.
        """
        mean, std = self.marginal_prob(x, t)
        noise = jax.random.normal(rng, shape=x.shape)
        return mean + std * noise

class VESDE(eqx.Module):
    """
    Instead of controlling noise through a beta parameter like VPSDE,
    this directly controls the variance through sigma parameters.
    """
    sigma_min: float
    sigma_max: float
    T: float = 1.0
    schedule_type: str = "exponential"
    gamma: float = 1.0
    power: float = 1.0  
    
    def __init__(self, sigma_min=0.01, sigma_max=50.0, T=1.0, schedule_type="exponential", gamma=1.0, power=1.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.T = T
        self.schedule_type = schedule_type
        self.gamma = gamma
        self.power = power
    
    def sigma(self, t):

        if self.schedule_type == "exponential": #standard from paper
            return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        elif self.schedule_type == "linear":
            return self.sigma_min + t * (self.sigma_max - self.sigma_min)
        elif self.schedule_type == "power":
            return self.sigma_min * (self.sigma_max / self.sigma_min) ** (t ** self.power)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def drift(self, x, t):
        """Drift term for the SDE - zero for VESDE."""
        return jnp.zeros_like(x)
    
    def diffusion(self, x, t):
        """Diffusion coefficient for the SDE.
        
        The coefficient follows the formula from the VESDE paper:
        sigma(t) * sqrt(2 * log(sigma_max/aigma_min))
        """
        sigma_t = self.sigma(t)
        diffusion_coeff = sigma_t * jnp.sqrt(2 * (jnp.log(self.sigma_max) - jnp.log(self.sigma_min)))
        return diffusion_coeff * self.gamma
    
    def marginal_prob(self, x, t):
        """Marginal probability parameters for data at time t.
        
        For VESDE, the mean stays the same and the std grows according to sigma(t).
        """
        mean = x  # Mean is unmodified in VESDE
        std = self.sigma(t) * self.gamma
        return mean, std
    
    def marginal_prob_std(self, t):
        """Get standard deviation for marginal probability distribution."""
        return self.sigma(t) * self.gamma
    
    def prior_sampling(self, rng, shape):
        """Sample from the prior distribution.
        
        For VESDE, the prior is a Gaussian with std = sigma_max.
        """
        return jax.random.normal(rng, shape) * self.sigma_max * self.gamma
    
    def forward_sample(self, x, t, rng):
        """
        Generate noisy data at time t for a batch of clean data x.
        
        Args:
            x: Clean input data.
            t: Noise level, can be a scalar or a batch of times.
            rng: JAX random key.
        Returns:
            Noisy data sample.
        """
        mean, std = self.marginal_prob(x, t)
        noise = jax.random.normal(rng, shape=x.shape)
        return mean + std * noise
        
class subVPSDE(VPSDE):
    """
    A variant of VPSDE that uses the drift of a subVP SDE.
    This is used for certain types of variance-preserving diffusion.
    """
    
    def drift(self, x, t):
        """Drift term for the subVP SDE."""
        # This is the specific form of the drift for a subVP SDE
        # It differs from VPSDE in the coefficient
        drift = -0.5 * self.beta(t) * x / (1 - self.alpha(t)**2)
        return drift