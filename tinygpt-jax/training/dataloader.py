"""
Data loading utilities for TinyGPT-JAX.

Converts flat token arrays into batched (input, target) pairs
for next-token prediction training.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import jax.numpy as jnp


class DataLoader:
    """
    Samples random contiguous sequences from a flat token array.

    For autoregressive training:
        input  = tokens[t : t+block_size]
        target = tokens[t+1 : t+block_size+1]

    Each target token is what the model should predict given all prior tokens.
    """

    def __init__(
        self,
        data: np.ndarray,
        block_size: int,
        batch_size: int,
        seed: int = 42,
    ):
        """
        Args:
            data: 1D array of token ids (train or val split).
            block_size: Sequence length for each training example.
            batch_size: Number of sequences per batch.
            seed: RNG seed for reproducible batch sampling.
        """
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        # Need at least block_size + 1 tokens to form input/target pair
        if len(data) < block_size + 1:
            raise ValueError(
                f"Need at least {block_size + 1} tokens, got {len(data)}"
            )

    def __len__(self) -> int:
        """Approximate number of batches per epoch."""
        max_start = len(self.data) - self.block_size - 1
        return max(1, max_start // self.batch_size)

    def get_batch(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Sample one batch of (inputs, targets).

        Returns:
            inputs:  (batch_size, block_size) token ids
            targets: (batch_size, block_size) next-token labels
        """
        max_start = len(self.data) - self.block_size - 1
        starts = self.rng.integers(0, max_start + 1, size=self.batch_size)

        input_batch = np.stack(
            [self.data[s : s + self.block_size] for s in starts]
        )
        target_batch = np.stack(
            [self.data[s + 1 : s + self.block_size + 1] for s in starts]
        )

        return jnp.array(input_batch), jnp.array(target_batch)

    def iterate(self, num_batches: int) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Yield num_batches consecutive random batches."""
        for _ in range(num_batches):
            yield self.get_batch()
