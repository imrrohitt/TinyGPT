"""
Production-grade optimizer setup for TinyGPT-JAX.

Uses the same recipe as modern LLM pretraining (GPT-2/3, LLaMA style):
  1. Linear LR warmup
  2. Cosine decay to a floor LR
  3. AdamW with decoupled weight decay
  4. Weight decay masking (skip biases, LayerNorm, embeddings)
  5. Global gradient clipping
"""

from __future__ import annotations

from typing import Any

import jax
import optax


# Parameter path substrings that should NOT receive weight decay.
_NO_DECAY_KEYS = ("bias", "LayerNorm", "layernorm", "ln", "Embed", "embedding")


def create_lr_schedule(
    peak_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
) -> optax.Schedule:
    """
    Linear warmup followed by cosine decay.

    Warmup stabilizes early training when gradients are noisy.
    Cosine decay gently reduces LR so the model settles into a good minimum.
    """
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=max_steps,
        end_value=min_lr,
    )


def create_weight_decay_mask(params: Any) -> Any:
    """
    Build a boolean mask matching `params`.

    True  → apply weight decay (typically weight matrices)
    False → skip weight decay (biases, LayerNorm, embeddings)

    Decaying LayerNorm scale or embedding tables hurts small language models.
    """
    def _should_decay(path, _value) -> bool:
        path_str = "/".join(getattr(k, "key", str(k)) for k in path)
        return not any(key in path_str for key in _NO_DECAY_KEYS)

    return jax.tree_util.tree_map_with_path(_should_decay, params)


def count_decayed_params(params: Any, mask: Any) -> tuple[int, int]:
    """Return (decayed_param_count, total_param_count) for logging."""
    leaves = jax.tree_util.tree_leaves(params)
    mask_leaves = jax.tree_util.tree_leaves(mask)
    total = sum(p.size for p in leaves)
    decayed = sum(p.size for p, m in zip(leaves, mask_leaves) if m)
    return decayed, total


def create_optimizer(
    params: Any,
    *,
    learning_rate: float,
    min_learning_rate: float,
    weight_decay: float,
    max_steps: int,
    warmup_steps: int,
    grad_clip_norm: float,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
) -> tuple[optax.GradientTransformation, optax.Schedule]:
    """
    Build a full AdamW optimizer chain used in production LLM training.

    Pipeline:
        gradients → global norm clip → AdamW (with LR schedule + masked decay)

    Args:
        params: Model parameters (used to build the weight-decay mask).
        learning_rate: Peak LR after warmup.
        min_learning_rate: Floor LR at end of cosine decay.
        weight_decay: L2 penalty on decay-eligible weights only.
        max_steps: Total training steps (defines schedule length).
        warmup_steps: Linear warmup duration.
        grad_clip_norm: Max global gradient norm before clipping.
        beta1: Adam first moment decay (default 0.9).
        beta2: Adam second moment decay (default 0.95, common in LLMs).
        eps: Adam epsilon for numerical stability.

    Returns:
        (optimizer, lr_schedule) — use schedule(step) to log current LR.
    """
    lr_schedule = create_lr_schedule(
        peak_lr=learning_rate,
        min_lr=min_learning_rate,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
    )
    wd_mask = create_weight_decay_mask(params)

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=weight_decay,
            b1=beta1,
            b2=beta2,
            eps=eps,
            mask=wd_mask,
        ),
    )
    return optimizer, lr_schedule
