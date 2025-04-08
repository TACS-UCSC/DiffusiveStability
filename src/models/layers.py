import jax
import jax.numpy as jnp
import equinox as eqx

    
class GaussianFourierProjection(eqx.Module):
    """ Gaussian random features for encoding time steps.
        in: time
        out: time embedding
        () -> (embed_dim,)    
    """
    W: jax.Array
    scale: float = 30.0
    
    def __init__(self, embed_dim: int, key: jax.random.PRNGKey, scale: float = 30.0):
        self.W = jax.random.normal(key, (embed_dim // 2,)) * scale
        self.scale = scale
        
    def __call__(self, t: float , key = None):
        # Stop gradient on weights during forward pass
        W = jax.lax.stop_gradient(self.W)
        x_proj = t * W * 2 * jnp.pi 
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)])
    
    

class TransformerBlock(eqx.Module):
    
    layer_norm1: eqx.nn.LayerNorm
    layer_norm2: eqx.nn.LayerNorm
    
    # GLU feedforward components
    ff_gate_proj: eqx.nn.Linear
    ff_out_proj: eqx.nn.Linear
    
    # Attention components
    query_proj: eqx.nn.Linear
    key_proj: eqx.nn.Linear
    value_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear
    
    # Configuration parameters
    num_heads: int
    hidden_dim: int
    head_dim: int
 
    def __init__(self,
                 spatial_dim: int,
                 hidden_dim: int,
                 num_heads: int,
                 phead_scale: int,
                 key: jax.random.PRNGKey):
    
        k1, k2, k3, k4, k5a, k5b = jax.random.split(key, 6)
        self.layer_norm1 = eqx.nn.LayerNorm((spatial_dim, hidden_dim))
        self.layer_norm2 = eqx.nn.LayerNorm((spatial_dim, hidden_dim))
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // phead_scale
        
        # Create projection matrices for attention
        self.query_proj = eqx.nn.Linear(hidden_dim, num_heads * self.head_dim, key=k1)
        self.key_proj = eqx.nn.Linear(hidden_dim, num_heads * self.head_dim, key=k2)
        self.value_proj = eqx.nn.Linear(hidden_dim, num_heads * self.head_dim, key=k3)
        self.output_proj = eqx.nn.Linear(num_heads * self.head_dim, hidden_dim, key=k4)
        
        # Create GLU feedforward projections
        # Double width for GLU (values and gates)
        ff_inner_dim = hidden_dim * 4
        self.ff_gate_proj = eqx.nn.Linear(hidden_dim, ff_inner_dim * 2, key=k5a)
        self.ff_out_proj = eqx.nn.Linear(ff_inner_dim, hidden_dim, key=k5b)
                            
    def _apply_attention(self, x):
        # Input shape: (spatial_dim, hidden_dim)
        seq_len = x.shape[0]  # spatial_dim
        
        # Apply projections and reshape for attention
        # Output shape will be (seq_len, num_heads, head_dim)
        query = jax.vmap(self.query_proj)(x).reshape(seq_len, self.num_heads, self.head_dim)
        key = jax.vmap(self.key_proj)(x).reshape(seq_len, self.num_heads, self.head_dim)
        value = jax.vmap(self.value_proj)(x).reshape(seq_len, self.num_heads, self.head_dim)
        
        # Transpose to get shapes required by jax.nn.dot_product_attention
        # Shape becomes (num_heads, seq_len, head_dim)
        query = jnp.transpose(query, (1, 0, 2))
        key = jnp.transpose(key, (1, 0, 2))
        value = jnp.transpose(value, (1, 0, 2))
        
        # Cast to bfloat16 for cuDNN compatibility
        # (bf16 usually works better than fp16 for training)
        """query = query.astype(jnp.bfloat16)
        key = key.astype(jnp.bfloat16)
        value = value.astype(jnp.bfloat16)"""
        
        # Apply attention with CuDNN backend for flash attention
        # Output shape: (num_heads, seq_len, head_dim)
        attention_output = jax.nn.dot_product_attention(
            query=query,
            key=key,
            value=value,
            scale=1.0 / (self.head_dim ** 0.5),  # Standard scaling
            implementation='xla'  # Use CuDNN backend for flash attention
        )
        
        # Cast back to float32
        """attention_output = attention_output.astype(jnp.float32)"""
        
        # Transpose back to (seq_len, num_heads, head_dim)
        attention_output = jnp.transpose(attention_output, (1, 0, 2))
        
        # Reshape to (seq_len, num_heads * head_dim)
        attention_output = attention_output.reshape(seq_len, self.num_heads * self.head_dim)
        
        # Apply output projection
        # Output shape: (seq_len, hidden_dim)
        output = jax.vmap(self.output_proj)(attention_output)
        
        return output
    
    def __call__(self,
                 x,
                 key = None):
        # Apply attention with residual connection and layer norm
        x = x + self._apply_attention(x)
        x = self.layer_norm1(x)
        
        # Apply GLU feedforward network
        # 1. Project to higher dimension with gate values
        ff_output = jax.vmap(self.ff_gate_proj)(x)
        # 2. Apply GLU activation (splits input and applies gating)
        ff_output = jax.nn.glu(ff_output, axis=-1)
        # 3. Project back to hidden dimension
        ff_output = jax.vmap(self.ff_out_proj)(ff_output)
        
        # Apply residual connection and layer norm
        x = x + ff_output
        x = self.layer_norm2(x)
        
        return x

    
class Encoder(eqx.Module):
    """Encoder for KS diffusion model that projects states and time to higher dimensions.
    Inputs: 
        - noisy_state: (spatial_dim,)      # x(τ+1)
        - cond_state: (spatial_dim,)       # x(τ)
        - t: scalar                        # diffusion time
    Output: (spatial_dim, hidden_dim)
    """
    noisy_conv: eqx.nn.Conv1d
    cond_conv: eqx.nn.Conv1d
    noisy_tok: eqx.nn.MLP
    cond_tok: eqx.nn.MLP

    pos_embed: eqx.nn.RotaryPositionalEmbedding
    time_proj: GaussianFourierProjection
    time_mlp: eqx.nn.Linear
    
    layer_norm: eqx.nn.LayerNorm
    spatial_dim: int
    hidden_dim: int
    
    def __init__(self, 
                 spatial_dim: int,
                 hidden_dim: int,
                 key: jax.random.PRNGKey):
        k1, k2, k3, k4, k5, k6= jax.random.split(key, 6)
        
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim

        self.noisy_conv = eqx.nn.Conv1d(
            in_channels=1, out_channels=hidden_dim//4,
            kernel_size=1, stride=1, padding=0, key=k1
        )
        self.cond_conv = eqx.nn.Conv1d(
            in_channels=1, out_channels=hidden_dim//4,
            kernel_size=1, stride=1, padding=0, key=k2
        )
        
        self.noisy_tok = eqx.nn.MLP(
            in_size=hidden_dim//4,
            out_size=hidden_dim//4,
            width_size=hidden_dim//4 * 2,  # e.g. 2× bigger hidden layer
            depth=2,
            activation=jax.nn.gelu,
            key=k3
        )
        self.cond_tok = eqx.nn.MLP(
            in_size=hidden_dim//4,
            out_size=hidden_dim//4,
            width_size=hidden_dim//4 * 2,  # e.g. 2× bigger hidden layer
            depth=2,
            activation=jax.nn.gelu,
            key=k4
        )

        self.pos_embed = eqx.nn.RotaryPositionalEmbedding(hidden_dim//4)
        
        time_embed_dim = spatial_dim * hidden_dim//4
        self.time_proj = GaussianFourierProjection(time_embed_dim, key=k5)
        self.time_mlp = eqx.nn.Linear(time_embed_dim, time_embed_dim, key=k6)
        
        # LayerNorm needs to match the full shape (spatial_dim, hidden_dim)
        self.layer_norm = eqx.nn.LayerNorm((spatial_dim, hidden_dim))
    
    def __call__(self, noisy_state: jax.Array, cond_state: jax.Array, t: float, key = None) -> jax.Array:

        ns = noisy_state[None, :]  #(1, spatial_dim)
        cs = cond_state[None, :]   #(1, spatial_dim)

        noisy_proj = self.noisy_conv(ns)  # (hidden_dim//4, spatial_dim)
        cond_proj  = self.cond_conv(cs)   # (hidden_dim//4, spatial_dim)
        
        noisy_proj = noisy_proj.T  # (spatial_dim, hidden_dim//4)
        cond_proj  = cond_proj.T

        noisy_emb = jax.vmap(self.noisy_tok)(noisy_proj)  # same shape
        cond_emb  = jax.vmap(self.cond_tok)(cond_proj)

        p_emb = self.pos_embed(cond_emb)
        time_emb = self.time_proj(t)
        time_emb = self.time_mlp(time_emb)
        time_emb = time_emb.reshape(self.spatial_dim, self.hidden_dim//4)
        
        x = jnp.concatenate([noisy_emb, cond_emb, p_emb, time_emb], axis=-1)
        x = self.layer_norm(x)
        
        return x # Shape: (spatial_dim, hidden_dim)
    
class ConditionalEncoder(eqx.Module):
    """Encoder for diffusion model that projects multiple conditional states and time."""
    noisy_conv: eqx.nn.Conv1d
    cond_conv: eqx.nn.Conv1d
    noisy_tok: eqx.nn.MLP
    cond_tok: eqx.nn.MLP

    pos_embed: eqx.nn.RotaryPositionalEmbedding
    
    # Time projection components
    time_proj: GaussianFourierProjection
    time_conv: eqx.nn.Conv1d
    
    # Conditional time projection components
    cond_time_proj: GaussianFourierProjection
    cond_time_conv: eqx.nn.Conv1d
    
    layer_norm: eqx.nn.LayerNorm
    spatial_dim: int
    hidden_dim: int
    
    def __init__(self, 
                 spatial_dim: int,
                 hidden_dim: int,
                 key: jax.random.PRNGKey,
                 cond_states: int = 2):
        k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)
        
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim

        self.noisy_conv = eqx.nn.Conv1d(
            in_channels=1, out_channels=hidden_dim//4,
            kernel_size=1, stride=1, padding=0, key=k1
        )
        self.cond_conv = eqx.nn.Conv1d(
            in_channels=cond_states, out_channels=hidden_dim//4,
            kernel_size=1, stride=1, padding=0, key=k2
        )
        
        self.noisy_tok = eqx.nn.MLP(
            in_size=hidden_dim//4,
            out_size=hidden_dim//4,
            width_size=hidden_dim//4 * 2,
            depth=2,
            activation=jax.nn.gelu,
            key=k3
        )
        self.cond_tok = eqx.nn.MLP(
            in_size=hidden_dim//4,
            out_size=hidden_dim//4,
            width_size=hidden_dim//4 * 2,
            depth=2,
            activation=jax.nn.gelu,
            key=k4
        )

        self.pos_embed = eqx.nn.RotaryPositionalEmbedding(hidden_dim//4)
        
        # Time projection for diffusion time t
        self.time_proj = GaussianFourierProjection(spatial_dim, key=k5)
        # Convolutional layer to project time features to hidden dimension
        self.time_conv = eqx.nn.Conv1d(
            in_channels=1, out_channels=hidden_dim//8,
            kernel_size=1, stride=1, padding=0, key=k6
        )
        
        # Time projection for conditional states
        self.cond_time_proj = GaussianFourierProjection(spatial_dim, key=k7)
        # Convolutional layer to project conditional time features to hidden dimension
        self.cond_time_conv = eqx.nn.Conv1d(
            in_channels=cond_states, out_channels=hidden_dim//8,
            kernel_size=1, stride=1, padding=0, key=k8
        )
        
        # LayerNorm needs to match the full shape (spatial_dim, hidden_dim)
        self.layer_norm = eqx.nn.LayerNorm((spatial_dim, hidden_dim))
    
    def __call__(self, noisy_state: jax.Array, cond_state: jax.Array, t: float, key = None) -> jax.Array:
        ns = noisy_state[None, :]  # (1, spatial_dim)
        # cond_state is (cond_states, spatial_dim)
        
        # Process noisy state
        noisy_proj = self.noisy_conv(ns)  # (hidden_dim//4, spatial_dim)
        noisy_proj = noisy_proj.T  # (spatial_dim, hidden_dim//4)
        noisy_emb = jax.vmap(self.noisy_tok)(noisy_proj)  # (spatial_dim, hidden_dim//4)
        
        # Process conditional states
        cond_proj = self.cond_conv(cond_state)  # (hidden_dim//4, spatial_dim)
        cond_proj = cond_proj.T  # (spatial_dim, hidden_dim//4)
        cond_emb = jax.vmap(self.cond_tok)(cond_proj)  # (spatial_dim, hidden_dim//4)
        
        # Generate spatial positional embeddings
        p_emb = self.pos_embed(cond_emb)  # (spatial_dim, hidden_dim//4)
        
        # Generate time indices for conditional states
        num_cond_states = cond_state.shape[0]
        cond_state_range = jnp.arange(0, -1*num_cond_states, -1)  # [0, -1, -2, ...]
        
        # --- New diffusion time embedding approach ---
        # Project diffusion time t to (spatial_dim,)
        time_proj = self.time_proj(t)  # () -> (spatial_dim,)
        # Reshape to (1, spatial_dim) for convolution
        time_proj = time_proj[None, :]  # (1, spatial_dim)
        # Apply convolution to get (hidden_dim//8, spatial_dim)
        time_emb = self.time_conv(time_proj)  # (hidden_dim//8, spatial_dim)
        # Transpose to (spatial_dim, hidden_dim//8)
        time_emb = time_emb.T  # (spatial_dim, hidden_dim//8)
        
        # --- New conditional time embedding approach ---
        # Project each conditional time to (spatial_dim,)
        cond_time_projs = jax.vmap(self.cond_time_proj)(cond_state_range)  # (cond_states, spatial_dim)
        # Apply convolution to get (hidden_dim//8, spatial_dim)
        cond_time_emb = self.cond_time_conv(cond_time_projs)  # (hidden_dim//8, spatial_dim)
        # Transpose to (spatial_dim, hidden_dim//8)
        cond_time_emb = cond_time_emb.T  # (spatial_dim, hidden_dim//8)
        
        # Concatenate all embeddings
        # hidden_dim//4 + hidden_dim//4 + hidden_dim//4 + hidden_dim//8 + hidden_dim//8 = hidden_dim
        x = jnp.concatenate([noisy_emb, cond_emb, p_emb, cond_time_emb, time_emb], axis=-1)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        return x  # Shape: (spatial_dim, hidden_dim)


"""class ConditionalEncoder(eqx.Module):
    #Encoder for diffusion model that projects multiple conditional states and time.
    noisy_conv: eqx.nn.Conv1d
    cond_conv: eqx.nn.Conv1d
    noisy_tok: eqx.nn.MLP
    cond_tok: eqx.nn.MLP

    pos_embed: eqx.nn.RotaryPositionalEmbedding
    time_proj: GaussianFourierProjection
    cond_time_proj: GaussianFourierProjection
    time_mlp: eqx.nn.Linear
    cond_time_mlp: eqx.nn.Linear
    
    layer_norm: eqx.nn.LayerNorm
    spatial_dim: int
    hidden_dim: int
    
    def __init__(self, 
                 spatial_dim: int,
                 hidden_dim: int,
                 key: jax.random.PRNGKey,
                 cond_states: int = 2):
        k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)
        
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim

        self.noisy_conv = eqx.nn.Conv1d(
            in_channels=1, out_channels=hidden_dim//4,
            kernel_size=1, stride=1, padding=0, key=k1
        )
        self.cond_conv = eqx.nn.Conv1d(
            in_channels=cond_states, out_channels=hidden_dim//4,
            kernel_size=1, stride=1, padding=0, key=k2
        )
        
        self.noisy_tok = eqx.nn.MLP(
            in_size=hidden_dim//4,
            out_size=hidden_dim//4,
            width_size=hidden_dim//4 * 2,
            depth=2,
            activation=jax.nn.gelu,
            key=k3
        )
        self.cond_tok = eqx.nn.MLP(
            in_size=hidden_dim//4,
            out_size=hidden_dim//4,
            width_size=hidden_dim//4 * 2,
            depth=2,
            activation=jax.nn.gelu,
            key=k4
        )

        self.pos_embed = eqx.nn.RotaryPositionalEmbedding(hidden_dim//4)
        
        # Adjust dimensions to ensure they add up correctly when concatenated
        time_embed_dim = spatial_dim * hidden_dim//8
        self.time_proj = GaussianFourierProjection(time_embed_dim, key=k5)
        self.cond_time_proj = GaussianFourierProjection(hidden_dim//8, key=k6)  # Simpler dimension
        self.time_mlp = eqx.nn.Linear(time_embed_dim, time_embed_dim, key=k7)
        self.cond_time_mlp = eqx.nn.Linear(hidden_dim//8, hidden_dim//8, key=k8)
        
        # LayerNorm needs to match the full shape (spatial_dim, hidden_dim)
        self.layer_norm = eqx.nn.LayerNorm((spatial_dim, hidden_dim))
    
    def __call__(self, noisy_state: jax.Array, cond_state: jax.Array, t: float, key = None) -> jax.Array:
        ns = noisy_state[None, :]  # (1, spatial_dim)
        # cond_state is (cond_states, spatial_dim)
        
        # Generate temporal indices for conditional states
        cond_state_range = jnp.arange(0, -1*cond_state.shape[0], -1)
        
        # Process noisy state
        noisy_proj = self.noisy_conv(ns)  # (hidden_dim//4, spatial_dim)
        noisy_proj = noisy_proj.T  # (spatial_dim, hidden_dim//4)
        noisy_emb = jax.vmap(self.noisy_tok)(noisy_proj)  # (spatial_dim, hidden_dim//4)
        
        # Process conditional states
        cond_proj = self.cond_conv(cond_state)  # (hidden_dim//4, spatial_dim)
        cond_proj = cond_proj.T  # (spatial_dim, hidden_dim//4)
        cond_emb = jax.vmap(self.cond_tok)(cond_proj)  # (spatial_dim, hidden_dim//4)
        
        # Generate spatial positional embeddings
        p_emb = self.pos_embed(cond_emb)  # (spatial_dim, hidden_dim//4)
        
        # Generate diffusion time embedding
        time_emb = self.time_proj(t)  # (time_embed_dim,)
        time_emb = self.time_mlp(time_emb)  # (time_embed_dim,)
        time_emb = time_emb.reshape(self.spatial_dim, self.hidden_dim//8)  # (spatial_dim, hidden_dim//8)
        
        # Generate and process temporal embeddings for each conditional state
        cond_time_embs = jax.vmap(self.cond_time_proj)(cond_state_range)  # (cond_states, hidden_dim//8)
        cond_time_embs = jax.vmap(self.cond_time_mlp)(cond_time_embs)  # (cond_states, hidden_dim//8)
        
        # Combine temporal embeddings (averaging is simple, but you could use more sophisticated approaches)
        cond_time_emb = jnp.mean(cond_time_embs, axis=0)  # (hidden_dim//8,)
        
        # Expand conditional time embedding to match spatial dimension
        cond_time_emb = jnp.broadcast_to(cond_time_emb, (self.spatial_dim, self.hidden_dim//8))
        
        # Concatenate all embeddings (ensure dimensions add up to hidden_dim)
        # hidden_dim//4 + hidden_dim//4 + hidden_dim//4 + hidden_dim//8 + hidden_dim//8 = hidden_dim
        x = jnp.concatenate([noisy_emb, cond_emb, p_emb, cond_time_emb, time_emb], axis=-1)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        return x  # Shape: (spatial_dim, hidden_dim)
    
    """
