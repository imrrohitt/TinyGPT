"""
Token and positional embeddings for TinyGPT-JAX.

Embeddings convert discrete token ids into continuous vectors that
capture semantic relationships. Positional embeddings tell the model
where each token sits in the sequence (order matters!).
"""

import flax.linen as nn
import jax.numpy as jnp


class TokenEmbedding(nn.Module):
    """
    Lookup table: token_id → d_model-dimensional vector.

    Example internal table (simplified to 2D for illustration):

        "Dog"   → [0.9, 0.8]
        "Cat"   → [0.8, 0.7]
        "Car"   → [0.1, 0.2]

    After training, similar words end up with similar vectors.
    """

    vocab_size: int
    d_model: int

    @nn.compact
    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            token_ids: Integer array of shape (batch, seq_len).

        Returns:
            Embeddings of shape (batch, seq_len, d_model).
        """
        # Each token id indexes into a learned (vocab_size, d_model) table
        return nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)(
            token_ids
        )


class PositionalEmbedding(nn.Module):
    """
    Learned positional embeddings added to token embeddings.

    Solves the ordering problem:
        "Dog bites man" vs "Man bites dog"
    Same words, different meaning — position vectors disambiguate.
    """

    block_size: int
    d_model: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Add position information to token embeddings.

        Args:
            x: Token embeddings, shape (batch, seq_len, d_model).

        Returns:
            x + position embeddings, same shape.
        """
        seq_len = x.shape[1]

        # Learned position table: one vector per position 0 … block_size-1
        positions = nn.Embed(num_embeddings=self.block_size, features=self.d_model)(
            jnp.arange(seq_len)
        )

        # Broadcast positions across the batch dimension and add
        return x + positions[None, :, :]


class GPTEmbeddings(nn.Module):
    """
    Combines token embeddings + positional embeddings + dropout.

    This is the entry point of the GPT model — raw token ids go in,
    contextualized vectors come out (before any transformer blocks).
    """

    vocab_size: int
    d_model: int
    block_size: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, token_ids: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
        """
        Args:
            token_ids: (batch, seq_len) integer token ids.
            train: Whether to apply dropout (True during training).

        Returns:
            (batch, seq_len, d_model) embedded representations.
        """
        x = TokenEmbedding(self.vocab_size, self.d_model)(token_ids)
        x = PositionalEmbedding(self.block_size, self.d_model)(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        return x
