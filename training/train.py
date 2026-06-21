"""
Training loop for TinyGPT-JAX.

Run:
    python -m training.train

This script:
    1. Loads processed data and vocabulary
    2. Initializes the GPT model (~1M parameters)
    3. Trains with next-token prediction loss
    4. Saves checkpoints periodically
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (  # noqa: E402
    CHECKPOINT_DIR,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    VOCAB_PATH,
    model_config,
    train_config,
)
from model.gpt import GPT, count_parameters  # noqa: E402
from tokenizer.tokenizer import Tokenizer  # noqa: E402
from training.dataloader import DataLoader  # noqa: E402
from training.loss import cross_entropy_loss  # noqa: E402
from training.optimizer import (  # noqa: E402
    count_decayed_params,
    create_optimizer,
    create_weight_decay_mask,
)


def create_train_state(rng, model: GPT):
    """
    Initialize model parameters.

    Returns a tuple: (state dict, model) where state holds params, opt_state, step.
    """
    dummy_input = jnp.ones((1, model_config.block_size), dtype=jnp.int32)
    variables = model.init(rng, dummy_input, train=True)
    params = variables["params"]

    state = {
        "params": params,
        "opt_state": None,
        "step": 0,
    }
    return state, model


def make_train_step(model: GPT, optimizer):
    """Build a JIT-compiled train step closed over the model architecture."""

    @jax.jit
    def train_step(state, batch_inputs, batch_targets, dropout_rng):
        def loss_fn(params):
            logits = model.apply(
                {"params": params},
                batch_inputs,
                train=True,
                rngs={"dropout": dropout_rng},
            )
            return cross_entropy_loss(logits, batch_targets)

        loss, grads = jax.value_and_grad(loss_fn)(state["params"])
        updates, new_opt_state = optimizer.update(
            grads, state["opt_state"], state["params"]
        )
        new_params = optax.apply_updates(state["params"], updates)

        new_state = {
            "params": new_params,
            "opt_state": new_opt_state,
            "step": state["step"] + 1,
        }
        return new_state, loss

    return train_step


def make_eval_step(model: GPT):
    """Build a JIT-compiled eval step closed over the model architecture."""

    @jax.jit
    def eval_step(params, batch_inputs, batch_targets):
        logits = model.apply({"params": params}, batch_inputs, train=False)
        return cross_entropy_loss(logits, batch_targets)

    return eval_step


def main() -> None:
    print("=" * 60)
    print("TinyGPT-JAX Training")
    print("=" * 60)

    # --- Load data ---
    if not VOCAB_PATH.exists():
        print("Processed data not found. Run: python -m tokenizer.build_vocab")
        sys.exit(1)

    tokenizer = Tokenizer.load(VOCAB_PATH)
    train_data = np.load(TRAIN_DATA_PATH)
    val_data = np.load(VAL_DATA_PATH)

    # Update vocab size in config from actual vocabulary
    model_config.vocab_size = tokenizer.vocab_size

    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    print(f"Train tokens:    {len(train_data)}")
    print(f"Val tokens:      {len(val_data)}")

    # --- Initialize model ---
    rng = jax.random.PRNGKey(train_config.seed)
    init_rng, dropout_rng = jax.random.split(rng)

    model = GPT(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        n_layers=model_config.n_layers,
        n_heads=model_config.n_heads,
        d_ff=model_config.d_ff,
        block_size=model_config.block_size,
        dropout_rate=model_config.dropout_rate,
    )

    # Init params first — optimizer needs them for weight-decay masking
    state, model = create_train_state(init_rng, model)
    optimizer, lr_schedule = create_optimizer(
        state["params"],
        learning_rate=train_config.learning_rate,
        min_learning_rate=train_config.min_learning_rate,
        weight_decay=train_config.weight_decay,
        max_steps=train_config.max_steps,
        warmup_steps=train_config.warmup_steps,
        grad_clip_norm=train_config.grad_clip_norm,
        beta1=train_config.beta1,
        beta2=train_config.beta2,
    )
    state["opt_state"] = optimizer.init(state["params"])

    wd_mask = create_weight_decay_mask(state["params"])
    decayed, total = count_decayed_params(state["params"], wd_mask)
    train_step = make_train_step(model, optimizer)
    eval_step = make_eval_step(model)
    n_params = count_parameters(state["params"])
    print(f"\nModel parameters: {n_params:,} (~{n_params / 1e6:.2f}M)")
    print(
        f"Optimizer: AdamW (β1={train_config.beta1}, β2={train_config.beta2}) "
        f"+ warmup({train_config.warmup_steps}) → cosine decay"
    )
    print(
        f"  LR: {train_config.learning_rate} → {train_config.min_learning_rate} "
        f"| weight_decay={train_config.weight_decay} on {decayed:,}/{total:,} params"
    )

    # --- Data loaders ---
    train_loader = DataLoader(
        train_data,
        block_size=model_config.block_size,
        batch_size=train_config.batch_size,
        seed=train_config.seed,
    )
    val_loader = DataLoader(
        val_data,
        block_size=model_config.block_size,
        batch_size=train_config.batch_size,
        seed=train_config.seed + 1,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {train_config.max_steps} steps...")
    print(f"  batch_size={train_config.batch_size}, block_size={model_config.block_size}")
    print(f"  n_layers={model_config.n_layers}, grad_clip={train_config.grad_clip_norm}")
    print()

    # --- Training loop ---
    best_val_loss = float("inf")
    best_step = 0
    start_time = time.time()

    pbar = tqdm(range(1, train_config.max_steps + 1), desc="Training")
    for step in pbar:
        dropout_rng, step_rng = jax.random.split(dropout_rng)
        inputs, targets = train_loader.get_batch()
        state, loss = train_step(state, inputs, targets, step_rng)

        if step % 100 == 0:
            current_lr = float(lr_schedule(step))
            pbar.set_postfix(loss=f"{float(loss):.4f}", lr=f"{current_lr:.2e}")

        if step % train_config.eval_every == 0:
            total_loss = 0.0
            for inputs, targets in val_loader.iterate(10):
                total_loss += float(eval_step(state["params"], inputs, targets))
            val_loss = total_loss / 10
            current_lr = float(lr_schedule(step))
            pbar.write(
                f"  Step {step:5d} | train_loss={float(loss):.4f} "
                f"| val_loss={val_loss:.4f} | lr={current_lr:.2e}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                best_dir = CHECKPOINT_DIR / "best"
                checkpoints.save_checkpoint(
                    ckpt_dir=str(best_dir),
                    target={"params": state["params"], "step": step},
                    step=step,
                    keep=1,
                    overwrite=True,
                )
                pbar.write(f"  ★ New best val_loss={val_loss:.4f} — saved to {best_dir}")

        if step % train_config.checkpoint_every == 0:
            checkpoints.save_checkpoint(
                ckpt_dir=str(CHECKPOINT_DIR),
                target={"params": state["params"], "step": step},
                step=step,
                keep=5,
                overwrite=True,
            )
            pbar.write(f"  Saved checkpoint at step {step}")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best validation loss: {best_val_loss:.4f} (step {best_step})")

    # Save final checkpoint only if we didn't just save one at max_steps
    if train_config.max_steps % train_config.checkpoint_every != 0:
        checkpoints.save_checkpoint(
            ckpt_dir=str(CHECKPOINT_DIR),
            target={"params": state["params"], "step": train_config.max_steps},
            step=train_config.max_steps,
            keep=5,
            overwrite=True,
        )
        print(f"Final checkpoint saved to {CHECKPOINT_DIR}")
    print("\nGenerate text with: python -m inference.generate")


if __name__ == "__main__":
    main()
