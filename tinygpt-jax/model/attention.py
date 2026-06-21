"""
Multi-head causal self-attention for TinyGPT-JAX.

Attention is the core mechanism that lets each token "look at" other
tokens to gather context. Causal masking ensures the model only sees
past tokens (required for autoregressive generation).
"""

import flax.linen as nn
import jax.numpy as jnp


class MultiHeadAttention(nn.Module):
    """
    Scaled dot-product multi-head self-attention with causal masking.

    Each head learns a different "view" of the relationships between tokens:
        Head 1 might focus on grammar
        Head 2 might track entity references ("he" → "Rohit")
        Head 3 might capture local word patterns

    Query / Key / Value analogy (LinkedIn search):
        Query  = your search term       ("AI Engineer")
        Key    = profile headline       (what each word "advertises")
        Value  = profile content        (what gets returned after matching)
    """

    d_model: int
    n_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, *, train: bool = True
    ) -> jnp.ndarray:
        """
        Args:
            x: Input tensor, shape (batch, seq_len, d_model).
            train: Whether to apply attention dropout.

        Returns:
            Attention output, shape (batch, seq_len, d_model).
        """
        batch_size, seq_len, _ = x.shape
        head_dim = self.d_model // self.n_heads

        # Project input into Query, Key, Value — each (batch, seq, d_model)
        q = nn.Dense(self.d_model, name="query")(x)
        k = nn.Dense(self.d_model, name="key")(x)
        v = nn.Dense(self.d_model, name="value")(x)

        # Reshape into multiple heads: (batch, n_heads, seq_len, head_dim)
        q = q.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(0, 2, 1, 3)

        # Attention scores: (batch, n_heads, seq_len, seq_len)
        # Each row answers: "how much should token i attend to token j?"
        scale = jnp.sqrt(head_dim).astype(x.dtype)
        attn_scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale

        # Causal mask: prevent attending to future tokens
        # Lower-triangular matrix — position i can only see positions ≤ i
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        attn_scores = jnp.where(
            causal_mask == 0,
            jnp.finfo(x.dtype).min,  # Masked positions → -inf before softmax
            attn_scores,
        )

        # Softmax converts scores to probabilities (each row sums to 1)
        attn_weights = nn.softmax(attn_scores, axis=-1)
        attn_weights = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(
            attn_weights
        )

        # Weighted sum of values: (batch, n_heads, seq_len, head_dim)
        attn_output = jnp.matmul(attn_weights, v)

        # Merge heads back: (batch, seq_len, d_model)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.d_model
        )

        # Final linear projection mixes information across heads
        return nn.Dense(self.d_model, name="output")(attn_output)
