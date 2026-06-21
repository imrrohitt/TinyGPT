"""
Build vocabulary and pre-process raw text into training arrays.

Run this script once before training:

    python -m tokenizer.build_vocab

It reads data/raw/stories.txt, builds a word vocabulary,
and saves tokenized sequences to data/processed/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running as a script: python tokenizer/build_vocab.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (  # noqa: E402
    PROCESSED_DIR,
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    VOCAB_PATH,
    model_config,
    train_config,
)
from tokenizer.tokenizer import PAD_TOKEN, UNK_TOKEN, Tokenizer  # noqa: E402


def build_vocab(text: str, min_freq: int = 1) -> dict[str, int]:
    """
    Scan all text and assign each unique word an integer id.

    Special tokens (<pad>, <unk>) always occupy the lowest ids.
    Words appearing fewer than min_freq times are excluded from vocab
    and will map to <unk> at encode time.

    Args:
        text: Full raw corpus as a single string.
        min_freq: Minimum word count to include in vocabulary.

    Returns:
        vocab dict mapping word → id.
    """
    # Count how often each word appears (same normalization as encode())
    word_counts: dict[str, int] = {}
    for line in text.splitlines():
        for word in Tokenizer._tokenize(line):
            word_counts[word] = word_counts.get(word, 0) + 1

    # Special tokens get the first ids (0, 1)
    vocab: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}

    # Remaining words sorted alphabetically for reproducibility
    idx = len(vocab)
    for word in sorted(word_counts):
        if word_counts[word] >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab


def text_to_token_ids(text: str, tokenizer: Tokenizer) -> np.ndarray:
    """
    Encode every non-empty line and concatenate into one flat array.

    Each line becomes a sequence of token ids separated implicitly
    by the line breaks in the source file.
    """
    all_ids: list[int] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            all_ids.extend(tokenizer.encode(line))
    return np.array(all_ids, dtype=np.int32)


def train_val_split(
    data: np.ndarray, val_fraction: float, seed: int, min_seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a flat token array into train and validation sets.

    Ensures the validation split is large enough to form at least one
    training sequence of length min_seq_len.
    """
    min_val_tokens = max(min_seq_len + 1, int(len(data) * val_fraction))
    min_val_tokens = min(min_val_tokens, len(data) // 5)  # keep at least 80% for train
    min_val_tokens = max(min_val_tokens, min_seq_len + 1)

    if len(data) <= min_val_tokens + min_seq_len:
        # Tiny corpus: train on all data, validate on a duplicated tail slice
        val_data = data[-(min_seq_len + 1) :].copy()
        train_data = data.copy()
        return train_data, val_data

    split_idx = len(data) - min_val_tokens
    return data[:split_idx], data[split_idx:]


def main() -> None:
    """Entry point: read raw text → build vocab → save processed arrays."""
    print(f"Reading corpus from {RAW_DATA_PATH}")
    text = RAW_DATA_PATH.read_text(encoding="utf-8")

    print("Building vocabulary...")
    vocab = build_vocab(text)
    tokenizer = Tokenizer(vocab)
    print(f"  Vocabulary size: {tokenizer.vocab_size} tokens")

    print("Tokenizing corpus...")
    token_ids = text_to_token_ids(text, tokenizer)
    print(f"  Total tokens: {len(token_ids)}")

    print(f"Splitting train/val (val_fraction={train_config.val_split})...")
    train_data, val_data = train_val_split(
        token_ids, train_config.val_split, train_config.seed, model_config.block_size
    )
    print(f"  Train tokens: {len(train_data)}")
    print(f"  Val tokens:   {len(val_data)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(VOCAB_PATH)
    np.save(TRAIN_DATA_PATH, train_data)
    np.save(VAL_DATA_PATH, val_data)

    print(f"\nSaved:")
    print(f"  {VOCAB_PATH}")
    print(f"  {TRAIN_DATA_PATH}")
    print(f"  {VAL_DATA_PATH}")
    print("\nDone! You can now run: python -m training.train")


if __name__ == "__main__":
    main()
