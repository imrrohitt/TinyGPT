"""
Central configuration for TinyGPT-JAX.

All hyperparameters live here so you can tune the ~1M parameter model
without hunting through the codebase.
"""

from dataclasses import dataclass
from pathlib import Path


# Project root: tinygpt-jax/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "stories.txt"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VOCAB_PATH = PROCESSED_DIR / "vocab.pkl"
TRAIN_DATA_PATH = PROCESSED_DIR / "train.npy"
VAL_DATA_PATH = PROCESSED_DIR / "val.npy"

# Checkpoint directory
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


@dataclass
class ModelConfig:
    """
    GPT architecture settings targeting ~1M parameters.

    Parameter count scales roughly with:
        vocab_size * d_model          (token embeddings, tied to output)
      + block_size * d_model        (positional embeddings)
      + n_layers * (4*d_model^2 + 2*d_model*d_ff)  (attention + FFN)
    """

    # Vocabulary size is set dynamically after building vocab.
    # Default placeholder; overwritten at training time.
    vocab_size: int = 512

    # Embedding dimension — each token becomes a vector of this size.
    d_model: int = 128

    # Number of transformer blocks stacked on top of each other.
    n_layers: int = 4

    # Number of attention heads per block (d_model must be divisible by n_heads).
    n_heads: int = 4

    # Feed-forward hidden dimension (typically 4x d_model).
    d_ff: int = 512

    # Maximum sequence length the model can process at once.
    # Kept modest so the demo corpus (small stories.txt) can form valid batches.
    block_size: int = 32

    # Dropout rate for regularization during training.
    dropout_rate: float = 0.1


@dataclass
class TrainConfig:
    """Training loop settings."""

    # Mini-batch size — how many sequences per gradient step.
    batch_size: int = 32

    # Total number of gradient update steps.
    max_steps: int = 10_000

    # Peak learning rate (reached after warmup).
    learning_rate: float = 3e-4

    # Floor LR at the end of cosine decay.
    min_learning_rate: float = 3e-5

    # Linear warmup steps before cosine decay begins.
    warmup_steps: int = 200

    # AdamW weight decay — applied only to weight matrices (not bias/LN/embed).
    weight_decay: float = 0.1

    # Adam beta1 / beta2 — (0.9, 0.95) is standard for LLM pretraining.
    beta1: float = 0.9
    beta2: float = 0.95

    # Gradient clipping norm — prevents exploding gradients.
    grad_clip_norm: float = 1.0

    # Save a checkpoint every N steps.
    checkpoint_every: int = 1000

    # Evaluate on validation set every N steps.
    eval_every: int = 500

    # Random seed for reproducibility.
    seed: int = 42

    # Fraction of data used for validation.
    val_split: float = 0.1


# Default configs used across the project.
model_config = ModelConfig()
train_config = TrainConfig()
