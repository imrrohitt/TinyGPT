"""
Transformer block for TinyGPT-JAX.

One block = Attention → Feed Forward, each wrapped in residual
connections and layer normalization (Pre-LN style, used by GPT-2).

Stack multiple blocks and each one refines the model's understanding.
"""

import flax.linen as nn
import jax.numpy as jnp

from model.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network (FFN).

    Applied independently to each token position:
        Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model)

    Think of attention as "gather context" and FFN as "process it".
    """

    d_model: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        x = nn.Dense(self.d_ff)(x)
        x = nn.gelu(x)  # Smooth activation — better than ReLU for language
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(self.d_model)(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single transformer "thinking step".

    Architecture (Pre-LN, GPT-style):
        x → LayerNorm → Attention → + residual
          → LayerNorm → FFN       → + residual

    Residual connections let gradients flow through deep stacks,
    which is why we can train 4+ layers effectively.
    """

    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        # --- Self-attention sub-layer ---
        residual = x
        x = nn.LayerNorm()(x)
        x = MultiHeadAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate,
        )(x, train=train)
        x = residual + x  # Residual: preserve original signal

        # --- Feed-forward sub-layer ---
        residual = x
        x = nn.LayerNorm()(x)
        x = FeedForward(
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout_rate=self.dropout_rate,
        )(x, train=train)
        x = residual + x

        return x
