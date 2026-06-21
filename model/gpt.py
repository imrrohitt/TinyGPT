"""
Full GPT model for TinyGPT-JAX.

Combines embeddings → N transformer blocks → output logits.

The output logits are unnormalized scores over the vocabulary.
After softmax they become probabilities for the next token.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from model.transformer import TransformerBlock


class GPT(nn.Module):
    """
    A minimal GPT-style autoregressive language model.

    Forward pass:
        token_ids → Embeddings → [TransformerBlock × n_layers]
                  → LayerNorm → Linear (tied to embeddings) → logits

    Weight tying: the output projection shares weights with the
    token embedding table, halving embedding parameters and often
    improving generalization on small models.
    """

    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    block_size: int
    dropout_rate: float = 0.1

    def setup(self):
        # Token embedding table — also reused as the output projection (weight tying)
        self.token_embedding = nn.Embed(self.vocab_size, self.d_model)

        # Learned positional embeddings: one vector per position 0 … block_size-1
        self.pos_embedding = nn.Embed(self.block_size, self.d_model)

        # Stack of transformer blocks
        self.blocks = [
            TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                name=f"block_{i}",
            )
            for i in range(self.n_layers)
        ]

        self.final_ln = nn.LayerNorm(name="final_ln")
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(
        self, token_ids: jnp.ndarray, *, train: bool = True
    ) -> jnp.ndarray:
        """
        Args:
            token_ids: (batch, seq_len) integer token ids.
            train: Enable dropout when True.

        Returns:
            logits: (batch, seq_len, vocab_size) raw scores per token position.
        """
        batch_size, seq_len = token_ids.shape

        # Step 1: Token embeddings — discrete ids → continuous vectors
        x = self.token_embedding(token_ids)

        # Step 2: Add positional information so word order matters
        positions = jnp.arange(seq_len)
        x = x + self.pos_embedding(positions)[None, :, :]

        # Dropout on embeddings (regularization during training)
        x = self.dropout(x, deterministic=not train)

        # Step 3: Pass through each transformer block
        for block in self.blocks:
            x = block(x, train=train)

        # Step 4: Final normalization before predicting next token
        x = self.final_ln(x)

        # Step 5: Project to vocabulary logits via weight tying
        # Same matrix used for input embedding and output projection
        logits = x @ self.token_embedding.embedding.T

        return logits


def count_parameters(params) -> int:
    """
    Count total trainable parameters in the model.

    Useful for verifying we're near the ~1M parameter target.
    """
    return sum(p.size for p in jax.tree_util.tree_leaves(params))
