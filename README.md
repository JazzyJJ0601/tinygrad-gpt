# TinyGrad GPT

A minimal, from-scratch implementation of a GPT-style language model built entirely with tinygrad. This project demonstrates the core concepts of transformer-based language modeling without relying on heavy frameworks like PyTorch or TensorFlow. It is designed to be fully transparent, readable, and educational.

## What It Is

TinyGrad GPT is a lightweight research project focused on understanding the mechanics of autoregressive language models. It implements a decoder-only transformer architecture optimized for educational purposes and rapid prototyping on consumer hardware.

## Architecture

The model consists of the following components:
- **Token Embeddings:** Maps integer input IDs to dense vector representations.
- **Transformer Blocks:** 2 stacked layers containing multi-head self-attention and feed-forward networks.
- **Attention Mechanism:** Causal self-attention ensuring each token can only attend to previous tokens.
- **Layer Normalization:** Applied pre-activation in each block for stability.
- **Output Head:** Linear projection mapping hidden states to vocabulary logits.

Default configuration: hidden size 64, 4 attention heads, vocabulary size 256 (character-level ASCII).

## Training and Loss

The model is trained on raw text datasets using cross-entropy loss. Training progress is visualized in `loss.png`, which displays the decay of training loss over steps. The plot shows a clear convergence pattern, with loss typically dropping significantly within 50 training steps on small datasets.

## Sample Output

Input: "The "
Output: "The quick brown fox jumps over the lazy dog"

## How to Run

1. Clone the repository.
2. Install dependencies: `pip install tinygrad numpy matplotlib`
3. Train: `python train.py`
4. Output files: `loss.png` (training plot), `samples.txt` (generated text)

## Tests

Run unit tests to verify numerical stability and attention masking:
```
pytest tests/ -v --cov=src
```

Coverage should exceed 85%.

## Limitations

- Context window limited to 32 tokens.
- Character-level vocabulary (no subword tokenization).
- Single-GPU or CPU training only; no distributed support.
- No float16 support for inference.
- Designed for learning, not production deployment.

This repo is intended for learning and experimentation, not production deployment.
