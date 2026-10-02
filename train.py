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
    """Mean cross-entropy in nats: -log softmax(logits)[target], via a stable log-sum-exp."""
    logits = model(x)  # (batch, seq, vocab)
    batch_size, seq_len, vocab_size = logits.shape
    one_hot = np.zeros((batch_size, seq_len, vocab_size), dtype=np.float64)
    yd = y.data.astype(int)
    one_hot[np.arange(batch_size)[:, None], np.arange(seq_len)[None, :], yd] = 1.0
    m = Tensor(logits.data.max(axis=-1, keepdims=True))  # constant shift, no gradient needed
    lse = (logits - m).exp().sum(axis=-1, keepdims=True).log() + m
    logp = logits - lse
    return -((logp * Tensor(one_hot)).sum() / (batch_size * seq_len))


def eval_loss(model, sequences, seq_len, n=64):
    """Mean cross-entropy over n evenly spaced held-out windows."""
    starts = np.linspace(0, len(sequences) - seq_len - 1, n).astype(int)
    tot = 0.0
    for s in starts:
        x = Tensor(sequences[None, s:s + seq_len])
        y = Tensor(sequences[None, s + 1:s + seq_len + 1])
        tot += float(compute_loss(model, x, y).data)
    return tot / n


def baselines(train_ids, val_ids, vocab):
    """Held-out cross-entropy (nats/char) of add-one smoothed unigram and bigram counts from the training text."""
    uni = np.bincount(train_ids, minlength=vocab) + 1.0
    uni_nll = -np.mean(np.log(uni[val_ids] / uni.sum()))
    big = np.ones((vocab, vocab))
    np.add.at(big, (train_ids[:-1], train_ids[1:]), 1.0)
    big /= big.sum(axis=1, keepdims=True)
    big_nll = -np.mean(np.log(big[val_ids[:-1], val_ids[1:]]))
    return float(uni_nll), float(big_nll)


def train(max_steps=int(os.environ.get("STEPS", 1500)), seed=0):
    """Train on the first 80% of tiny.txt, keep the step with the best loss on the next 10% (validation),
    report loss on the last 10% (test, never used for any choice), write results/real.json."""
    import json, time
    np.random.seed(seed)
    n_layers, dim, n_heads, head_dim, hidden_dim = 2, 64, 1, 64, 256
    max_seq_len, batch_size, learning_rate = 32, 16, 3e-3

    text = load_text("data/tiny.txt")
    stoi, itoi, vocab = build_vocab(text)
    ids = encode(text, stoi)
    a, b = int(0.8 * len(ids)), int(0.9 * len(ids))
    train_ids, val_ids, test_ids = ids[:a], ids[a:b], ids[b:]
    uni_nll, big_nll = baselines(train_ids, test_ids, vocab)
    print(f"{len(text)} chars, vocab {vocab}; train {len(train_ids)}, val {len(val_ids)}, test {len(test_ids)}")
    print(f"baselines (test nats/char): uniform {np.log(vocab):.3f}, unigram {uni_nll:.3f}, bigram {big_nll:.3f}")

    model = GPT(vocab_size=vocab, n_layers=n_layers, dim=dim, n_heads=n_heads, head_dim=head_dim,
                hidden_dim=hidden_dim, max_seq_len=max_seq_len)
    opt = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    n_params = sum(p.data.size for p in model.parameters())
    losses, curve, best = [], [], (np.inf, -1, None)
    t0 = time.time()
    for step in range(max_steps):
        starts = np.random.randint(0, len(train_ids) - max_seq_len - 1, batch_size)
        bx = np.stack([train_ids[s:s + max_seq_len] for s in starts])
        by = np.stack([train_ids[s + 1:s + max_seq_len + 1] for s in starts])
        opt.zero_grad()
        loss = compute_loss(model, Tensor(bx), Tensor(by))
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
        if step % 100 == 0 or step == max_steps - 1:
            v = eval_loss(model, val_ids, max_seq_len)
            curve.append((step, v))
            if v < best[0]:
                best = (v, step, [p.data.copy() for p in model.parameters()])
            print(f"step {step}: train {np.mean(losses[-50:]):.3f}  val {v:.3f}  ({time.time() - t0:.0f}s)")
    for p, saved in zip(model.parameters(), best[2]):
        p.data[...] = saved
    test = eval_loss(model, test_ids, max_seq_len)
    print(f"best val {best[0]:.3f} at step {best[1]}; test {test:.3f}")

    plt.figure(figsize=(6, 4))
    plt.plot(losses, alpha=.4, label='train (per batch)')
    plt.plot([c[0] for c in curve], [c[1] for c in curve], 'o-', label='validation')
    plt.axhline(big_nll, ls='--', c='gray', label='bigram baseline')
    plt.xlabel('step'); plt.ylabel('cross-entropy (nats/char)'); plt.legend(); plt.tight_layout()
    plt.savefig('loss.png', dpi=100)

    seed_text = "The "
    gen = np.array([[stoi[c] for c in seed_text]])
    for _ in range(200):
        logits = model(Tensor(gen[:, -max_seq_len:])).data[0, -1]
        pr = np.exp(logits - logits.max()); pr /= pr.sum()
        gen = np.concatenate([gen, [[np.random.choice(vocab, p=pr)]]], axis=1)
    sample = decode(gen[0], itoi)
    with open('samples.txt', 'w', encoding='utf-8') as f:
        f.write(f"Seed: {seed_text!r}, sampled at temperature 1 from the best-validation checkpoint (step {best[1]})\n\n{sample}\n")

    os.makedirs('results', exist_ok=True)
    res = {"steps": max_steps, "params": int(n_params), "layers": n_layers, "dim": dim, "context": max_seq_len,
           "batch": batch_size, "lr": learning_rate, "train_chars": int(len(train_ids)), "val_chars": int(len(val_ids)), "test_chars": int(len(test_ids)),
           "best_step": best[1], "val_nats_per_char": round(best[0], 4), "test_nats_per_char": round(test, 4), "baseline_uniform": round(float(np.log(vocab)), 4),
           "baseline_unigram": round(uni_nll, 4), "baseline_bigram": round(big_nll, 4), "curve": curve,
           "seconds": round(time.time() - t0, 1)}
    with open('results/real.json', 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'curve'}))
    print("sample:", sample[:200])


if __name__ == '__main__':
    train()
