"""
Word-level tokenizer for TinyGPT-JAX.

Converts text ↔ integer token IDs using a fixed vocabulary.
Neural networks only understand numbers, so every word must be mapped
to an index before the model can process it.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, List


# Special tokens reserved at the start of the vocabulary.
PAD_TOKEN = "<pad>"   # Padding for batching sequences to equal length
UNK_TOKEN = "<unk>"   # Unknown words not seen during vocab building


class Tokenizer:
    """
    Whitespace tokenizer with normalization for robust inference.

    Normalization (applied at encode AND vocab-build time):
      - lowercase: "Rohit" and "rohit" map to the same token
      - punctuation split: "rohit?" → ["rohit", "?"]
    """

    def __init__(self, vocab: Dict[str, int]):
        """
        Args:
            vocab: Mapping from word string to integer token id.
                   Lower ids are reserved for special tokens.
        """
        self.vocab = vocab
        # Reverse mapping: id → word (used during decoding)
        self.id_to_token = {idx: word for word, idx in vocab.items()}

        self.pad_id = vocab[PAD_TOKEN]
        self.unk_id = vocab[UNK_TOKEN]

    @staticmethod
    def normalize(text: str) -> str:
        """
        Lowercase and isolate punctuation so prompts match training tokens.

        Example:
            "Who is Rohit?" → "who is rohit ?"
        """
        text = text.lower().strip()
        text = re.sub(r"([^\w\s])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        """Split normalized text into word tokens."""
        normalized = cls.normalize(text)
        if not normalized:
            return []
        return normalized.split()

    def encode(self, text: str) -> List[int]:
        """
        Convert a text string into a list of token ids.

        Unknown words are mapped to UNK_TOKEN.
        """
        return [self.vocab.get(word, self.unk_id) for word in self._tokenize(text)]

    def decode(self, token_ids: List[int]) -> str:
        """
        Convert a list of token ids back into a readable string.

        Skips padding tokens in the output.
        """
        words = []
        for token_id in token_ids:
            if token_id == self.pad_id:
                continue
            words.append(self.id_to_token.get(token_id, UNK_TOKEN))
        return " ".join(words)

    def save(self, path: Path) -> None:
        """Persist vocabulary to disk as a pickle file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.vocab, f)

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        """Load a saved vocabulary and return a Tokenizer instance."""
        with open(path, "rb") as f:
            vocab = pickle.load(f)
        return cls(vocab)

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary."""
        return len(self.vocab)
