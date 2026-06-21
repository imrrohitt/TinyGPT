# TinyGPT-JAX

A ~1M parameter autoregressive language model built from scratch with [JAX](https://github.com/google/jax). No LangChain, no RAG, no agents — just the fundamentals.

Learn every component of a GPT-style model by building it yourself.

## What You'll Learn

- Tokenization — text → numbers
- Embeddings — numbers → dense vectors
- Positional encoding — why word order matters
- Attention — how tokens look at each other
- Multi-head attention — multiple "experts" reading the same sentence
- Transformer blocks — stacked thinking steps
- Cross-entropy loss — measuring prediction quality
- Gradients & backpropagation — `jax.grad()` finds downhill
- AdamW optimizer — actually moving the weights
- Autoregressive generation — one token at a time

## Project Structure

```text
tinygpt-jax/
├── data/
│   ├── raw/stories.txt          # Raw text corpus
│   └── processed/               # vocab.pkl, train.npy, val.npy
├── tokenizer/                   # Fuel system
├── model/                       # Brain (embeddings, attention, GPT)
├── training/                    # Driving school
├── inference/                   # Real driving
├── checkpoints/                 # Game save points
└── configs/config.py            # All hyperparameters
```

## Quick Start

### 1. Install dependencies

```bash
cd tinygpt-jax
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Build vocabulary & process data

```bash
python -m tokenizer.build_vocab
```

### 3. Train the model

```bash
python -m training.train
```

Training runs for 10,000 steps by default (~1M parameters). Checkpoints save to `checkpoints/` every 1,000 steps.

### 4. Generate text

```bash
python -m inference.generate --prompt "The capital of France is"
python -m inference.generate --prompt "Rohit works as" --temperature 0.5
```

## Architecture

| Component | File | Role |
|-----------|------|------|
| Tokenizer | `tokenizer/tokenizer.py` | Text ↔ token IDs |
| Embeddings | `model/embeddings.py` | Token + positional vectors |
| Attention | `model/attention.py` | Causal multi-head self-attention |
| Transformer | `model/transformer.py` | Attention + FFN block |
| GPT | `model/gpt.py` | Full model with weight tying |
| Loss | `training/loss.py` | Cross-entropy for next-token prediction |
| Optimizer | `training/optimizer.py` | AdamW + gradient clipping |
| Train | `training/train.py` | Full training loop |
| Generate | `inference/generate.py` | Autoregressive sampling |

## Default Hyperparameters

| Setting | Value |
|---------|-------|
| Parameters | ~1M |
| d_model | 128 |
| n_layers | 4 |
| n_heads | 4 |
| d_ff | 512 |
| block_size | 128 |
| batch_size | 32 |
| learning_rate | 3e-4 |
| max_steps | 10,000 |

Edit `configs/config.py` to tune these.

## How Generation Works

```
Input:  "The capital of France is"
         ↓
Model predicts next token probabilities: Paris=80%, London=5%, ...
         ↓
Sample "Paris", append → "The capital of France is Paris"
         ↓
Repeat until max tokens reached
```

## Next Steps

Once you understand this codebase, these concepts become much clearer:

- **Fine-tuning** — same architecture, different data
- **LoRA** — low-rank updates to attention weights
- **RAG** — retrieve docs, then generate with this same loop
- **RLHF** — train a reward model, then optimize generation policy

Build the car before driving the Ferrari.
