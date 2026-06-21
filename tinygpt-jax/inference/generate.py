"""
Autoregressive text generation for TinyGPT-JAX.

Run:
    python -m inference.generate --prompt "The capital of France is"

The model generates text one token at a time:
    1. Feed current sequence to the model
    2. Take logits at the last position
    3. Sample the next token (temperature controls randomness)
    4. Append token and repeat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from flax.training import checkpoints

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CHECKPOINT_DIR, VOCAB_PATH, model_config  # noqa: E402
from model.gpt import GPT  # noqa: E402
from tokenizer.tokenizer import Tokenizer  # noqa: E402


def sample_next_token(
    logits: jnp.ndarray,
    rng: jax.random.PRNGKey,
    temperature: float = 1.0,
) -> int:
    """
    Sample one token from the model's output logits.

    Args:
        logits: (vocab_size,) scores for the next token.
        rng: JAX random key for sampling.
        temperature: Higher = more random, lower = more deterministic.
                     temperature=0 → always pick argmax (greedy).

    Returns:
        Sampled token id (integer).
    """
    if temperature <= 0:
        return int(jnp.argmax(logits))

    # Scale logits by temperature before softmax
    scaled_logits = logits / temperature
    probs = jax.nn.softmax(scaled_logits, axis=-1)
    return int(jax.random.categorical(rng, jnp.log(probs + 1e-10)))


def make_forward_last_logits(model: GPT):
    """Return a JIT function that outputs logits for the last token position."""

    @jax.jit
    def forward_last_logits(params, token_ids: jnp.ndarray) -> jnp.ndarray:
        logits = model.apply({"params": params}, token_ids, train=False)
        return logits[0, -1, :]  # (vocab_size,)

    return forward_last_logits


def generate(
    prompt: str,
    max_new_tokens: int = 30,
    temperature: float = 0.8,
    checkpoint_step: int | None = None,
) -> str:
    """
    Generate text autoregressively from a prompt string.

    Args:
        prompt: Starting text (e.g. "Rohit works as").
        max_new_tokens: How many tokens to generate after the prompt.
        temperature: Sampling temperature (0 = greedy, 1 = default).
        checkpoint_step: Load a specific checkpoint step (latest if None).

    Returns:
        Full generated string (prompt + continuation).
    """
    tokenizer = Tokenizer.load(VOCAB_PATH)
    model_config.vocab_size = tokenizer.vocab_size

    model = GPT(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        n_layers=model_config.n_layers,
        n_heads=model_config.n_heads,
        d_ff=model_config.d_ff,
        block_size=model_config.block_size,
        dropout_rate=model_config.dropout_rate,
    )

    # Load latest or specified checkpoint (need param template for Orbax restore)
    if not CHECKPOINT_DIR.exists() or not any(
        p.name.startswith("checkpoint_") for p in CHECKPOINT_DIR.iterdir()
    ):
        print("No checkpoint found. Train first: python -m training.train")
        sys.exit(1)

    init_rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, model_config.block_size), dtype=jnp.int32)
    template = model.init(init_rng, dummy_input, train=False)
    restored = checkpoints.restore_checkpoint(
        ckpt_dir=str(CHECKPOINT_DIR),
        target={"params": template["params"], "step": 0},
        step=checkpoint_step,
    )
    params = restored["params"]
    step = restored.get("step", "?")
    print(f"Loaded checkpoint (step {step})")

    # Encode prompt to token ids
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        print("Empty prompt after tokenization.")
        sys.exit(1)

    rng = jax.random.PRNGKey(0)

    forward_last_logits = make_forward_last_logits(model)

    # Generate one token at a time
    for _ in range(max_new_tokens):
        # Truncate to block_size if prompt grows too long
        context = token_ids[-model_config.block_size :]
        input_ids = jnp.array([context], dtype=jnp.int32)

        logits = forward_last_logits(params, input_ids)
        rng, sample_rng = jax.random.split(rng)
        next_token = sample_next_token(logits, sample_rng, temperature)

        token_ids.append(next_token)

        # Stop if we hit padding (shouldn't happen, but safe guard)
        if next_token == tokenizer.pad_id:
            break

    return tokenizer.decode(token_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with TinyGPT-JAX")
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Text prompt to continue",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=30,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0=greedy, higher=more random)",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Checkpoint step to load (default: latest)",
    )
    args = parser.parse_args()

    print(f"Prompt: {args.prompt!r}\n")
    output = generate(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        checkpoint_step=args.checkpoint,
    )
    print(f"Generated:\n{output}")


if __name__ == "__main__":
    main()
