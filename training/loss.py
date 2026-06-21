"""
Cross-entropy loss measures how far the model's predicted token
distribution is from the true next token. Lower loss = better predictions.
"""

import jax
import jax.numpy as jnp


def cross_entropy_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """
    Compute mean cross-entropy loss for next-token prediction.

    For each position, the model outputs logits (raw scores) over the
    entire vocabulary. We compare these to the true next token using
    softmax + negative log-likelihood.

    Example:
        Truth:   "I love AI"
        Model predicts at "love" position: AI=80%, pizza=10%, ...
        Low loss if AI has highest probability.

    Args:
        logits:  (batch, seq_len, vocab_size) raw model outputs.
        targets: (batch, seq_len) integer token ids (ground truth).

    Returns:
        Scalar mean loss across all positions in the batch.
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Flatten batch and sequence dimensions for efficient computation
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # log_softmax + pick the log-probability of the correct token
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    target_log_probs = log_probs[jnp.arange(logits_flat.shape[0]), targets_flat]

    # Negative log-likelihood: penalize low probability on correct token
    loss = -jnp.mean(target_log_probs)
    return loss


def compute_perplexity(loss: float) -> float:
    """
    Perplexity = exp(loss).

    Interpretation: on average, the model is as uncertain as if it were
    choosing uniformly among exp(loss) tokens. Lower perplexity is better.
    """
    import math
    return math.exp(loss)
