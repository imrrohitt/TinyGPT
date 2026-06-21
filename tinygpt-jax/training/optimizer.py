"""
Optimizer setup for TinyGPT-JAX.

AdamW adapts the learning rate per parameter and applies weight decay.
Think of gradients as "which direction is downhill" — the optimizer
actually takes the step.
"""

import optax


def create_optimizer(
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float = 1.0,
) -> optax.GradientTransformation:
    """
    Build an AdamW optimizer with gradient clipping.

    Pipeline:
        1. clip_by_global_norm — prevent exploding gradients
        2. adamw — adaptive learning rate + L2 weight decay

    Args:
        learning_rate: Peak step size for weight updates.
        weight_decay: L2 penalty strength (keeps weights small).
        grad_clip_norm: Maximum allowed gradient norm before clipping.

    Returns:
        An optax optimizer ready to use with optax.apply_updates().
    """
    return optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    )
