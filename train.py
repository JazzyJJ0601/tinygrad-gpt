#!/usr/bin/env python3
"""Train a character-level GPT on tiny.txt using tinygrad."""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the tinygrad package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinygrad.tensor import Tensor
from tinygrad.nn import GPT
from tinygrad.optim import AdamW


def load_text(path):
    """Load text file and return as a string."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def build_vocab(text):
    """Build vocabulary from text."""
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itoi = {i: ch for ch, i in stoi.items()}
    return stoi, itoi, len(chars)


def encode(text, stoi):
    """Encode text to token IDs."""
    return np.array([stoi[ch] for ch in text], dtype=np.int32)


def decode(tokens, itoi):
    """Decode token IDs to text."""
    return ''.join(itoi[int(t)] for t in tokens)


def get_batch(sequences, seq_len, batch_size, idx):
    """Get a batch of sequences for training."""
    # Create overlapping sequences
    batch_x = []
    batch_y = []
    for i in range(batch_size):
        start = (idx * batch_size + i) % (len(sequences) - seq_len)
        batch_x.append(sequences[start:start + seq_len])
        batch_y.append(sequences[start + 1:start + seq_len + 1])
    return np.array(batch_x), np.array(batch_y)


def compute_loss(model, x, y):
    """Compute cross-entropy loss."""
    logits = model(x)  # (batch, seq, vocab)
    batch_size, seq_len, vocab_size = logits.shape
    
    # Create one-hot for target tokens
    one_hot = np.zeros((batch_size, seq_len, vocab_size), dtype=np.float64)
    for i in range(batch_size):
        for j in range(seq_len):
            one_hot[i, j, int(y.data[i, j])] = 1.0
    one_hot = Tensor(one_hot)
    
    # Cross-entropy loss
    selected_logits = logits * one_hot
    total_loss = -(selected_logits.sum() / (batch_size * seq_len))
    return total_loss


def train():
    """Train the GPT model."""
    # Configuration
    vocab_size = 256  # Character level (ASCII)
    n_layers = 2
    dim = 32
    n_heads = 4
    head_dim = 8
    hidden_dim = 64
    max_seq_len = 32
    batch_size = 4
    learning_rate = 0.05
    max_steps = 50
    sample_steps = 10
    
    # Load and prepare data
    print("Loading text data...")
    text = load_text("data/tiny.txt")
    print(f"Text length: {len(text)} characters")
    
    stoi, itoi, actual_vocab = build_vocab(text)
    sequences = encode(text, stoi)
    print(f"Vocabulary size: {actual_vocab}")
    print(f"Total tokens: {len(sequences)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = GPT(
        vocab_size=actual_vocab,
        n_layers=n_layers,
        dim=dim,
        n_heads=n_heads,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len
    )
    
    # Initialize optimizer
    opt = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    
    # Training loop
    print("\nTraining...")
    losses = []
    
    for step in range(max_steps):
        # Get batch
        batch_x, batch_y = get_batch(sequences, max_seq_len, batch_size, step)
        batch_x = Tensor(batch_x)
        batch_y = Tensor(batch_y)
        
        # Forward pass
        opt.zero_grad()
        loss = compute_loss(model, batch_x, batch_y)
        loss.backward()
        
        # Store loss
        losses.append(float(loss.data))
        
        # Print progress
        if step % sample_steps == 0 or step == max_steps - 1:
            print(f"Step {step}: loss = {loss.data:.4f}")
        
        # Optimizer step
        opt.step()
    
    # Save loss plot
    print("\nSaving loss plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('GPT Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('loss.png', dpi=100, bbox_inches='tight')
    print("Saved loss.png")
    
    # Generate samples
    print("\nGenerating samples...")
    
    # Start with a seed sequence
    seed_text = "The "
    seed_tokens = np.array([[stoi[ch] for ch in seed_text]], dtype=np.int32)
    generated = seed_tokens.copy()
    
    for _ in range(200):
        # Get prediction for current sequence
        opt.zero_grad()
        np.seterr(all='ignore')
        logits = model(Tensor(generated))
        
        # Get next token (argmax)
        next_token = int(logits.data[0, -1, :].argmax())
        generated = np.concatenate([generated, [[next_token]]], axis=1)
    
    sample_text = decode(generated[0], itoi)
    
    # Save samples
    with open('samples.txt', 'w', encoding='utf-8') as f:
        f.write(f"Seed: {seed_text}\n\n")
        f.write(sample_text)
    
    print("Saved samples.txt")
    print("\nDone!")
    print(f"Sample text preview: {sample_text[:200]}...")


if __name__ == '__main__':
    train()
