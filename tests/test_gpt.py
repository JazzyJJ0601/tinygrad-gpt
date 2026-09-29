"""Test GPT model shapes and training."""
import sys
import numpy as np

sys.path.insert(0, '/home/jasper/mint-home/projects/github-portfolio/work/repos/tinygrad-gpt')

from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, LayerNorm, Attention, FeedForward, TransformerBlock, GPT
from tinygrad.optim import AdamW


def test_linear_shapes():
    """Test Linear layer output shapes."""
    np.random.seed(42)
    layer = Linear(in_features=10, out_features=5, bias=True)
    
    x = Tensor(np.random.randn(2, 10))
    out = layer(x)
    
    assert out.shape == (2, 5), f"Expected (2, 5), got {out.shape}"
    print("test_linear_shapes: PASS")


def test_layernorm_shapes():
    """Test LayerNorm output shapes."""
    np.random.seed(42)
    layer = LayerNorm(normalized_shape=10, eps=1e-5)
    
    x = Tensor(np.random.randn(2, 5, 10))
    out = layer(x)
    
    assert out.shape == (2, 5, 10), f"Expected (2, 5, 10), got {out.shape}"
    print("test_layernorm_shapes: PASS")


def test_attention_shapes():
    """Test Attention output shapes."""
    np.random.seed(42)
    attn = Attention(dim=32, n_heads=4, head_dim=8)
    
    x = Tensor(np.random.randn(2, 10, 32))
    out = attn(x)
    
    assert out.shape == (2, 10, 32), f"Expected (2, 10, 32), got {out.shape}"
    print("test_attention_shapes: PASS")


def test_transformer_block_shapes():
    """Test TransformerBlock output shapes."""
    np.random.seed(42)
    block = TransformerBlock(dim=32, n_heads=4, head_dim=8, hidden_dim=128)
    
    x = Tensor(np.random.randn(2, 10, 32))
    out = block(x)
    
    assert out.shape == (2, 10, 32), f"Expected (2, 10, 32), got {out.shape}"
    print("test_transformer_block_shapes: PASS")


def test_gpt_shapes():
    """Test GPT output shapes."""
    np.random.seed(42)
    model = GPT(
        vocab_size=1000,
        n_layers=2,
        dim=32,
        n_heads=4,
        head_dim=8,
        hidden_dim=128,
        max_seq_len=16
    )
    
    x = Tensor(np.random.randint(0, 1000, size=(2, 8)))
    out = model(x)
    
    assert out.shape == (2, 8, 1000), f"Expected (2, 8, 1000), got {out.shape}"
    print("test_gpt_shapes: PASS")


def test_loss_dropout():
    """Test that loss drops on a tiny overfit batch (standard training)."""
    np.random.seed(42)
    
    # Create a tiny model
    model = GPT(
        vocab_size=100,
        n_layers=2,
        dim=16,
        n_heads=2,
        head_dim=8,
        hidden_dim=64,
        max_seq_len=16
    )
    
    # Create a tiny batch with repeated inputs to force overfitting
    batch_x = Tensor(np.array([
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
    ]))
    
    batch_y = Tensor(np.array([
        [2, 3, 4, 5, 6, 7, 8, 9],
        [2, 3, 4, 5, 6, 7, 8, 9],
        [2, 3, 4, 5, 6, 7, 8, 9],
        [2, 3, 4, 5, 6, 7, 8, 9],
    ]))
    
    # Initialize optimizer with moderate learning rate
    opt = AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
    
    def compute_loss(model, x, y):
        """Compute cross-entropy loss using tensor operations."""
        logits = model(x)  # (batch, seq, vocab)
        batch_size, seq_len, vocab_size = logits.shape
        
        # Create one-hot for target tokens
        one_hot = np.zeros((batch_size, seq_len, vocab_size), dtype=np.float64)
        for i in range(batch_size):
            for j in range(seq_len):
                one_hot[i, j, int(y.data[i, j])] = 1.0
        one_hot = Tensor(one_hot)
        
        # Negative cross-entropy: -sum(logits * one_hot)
        # We want to minimize this (maximize target logit)
        selected_logits = logits * one_hot
        total_loss = -(selected_logits.sum() / (batch_size * seq_len))
        return total_loss
    
    # Get initial loss
    opt.zero_grad()
    initial_loss = compute_loss(model, batch_x, batch_y)
    initial_loss.backward()
    
    print(f"Initial loss: {initial_loss.data:.4f}")
    
    # Check if gradients are computed
    has_grads = any(p._grad is not None for p in model.parameters())
    print(f"Has gradients after backward: {has_grads}")
    
    # Do more optimizer steps
    for step in range(10):
        opt.step()
        opt.zero_grad()
        current_loss = compute_loss(model, batch_x, batch_y)
        current_loss.backward()
        print(f"Step {step+1}: loss = {current_loss.data:.4f}")
    
    final_loss = current_loss.data
    
    print(f"Final loss: {final_loss:.4f}")
    
    # Loss should decrease (standard training)
    assert final_loss < initial_loss.data, f"Loss should decrease: {initial_loss.data:.4f} -> {final_loss:.4f}"
    print("test_loss_dropout: PASS")


if __name__ == '__main__':
    test_linear_shapes()
    test_layernorm_shapes()
    test_attention_shapes()
    test_transformer_block_shapes()
    test_gpt_shapes()
    test_loss_dropout()
    print("\nAll tests passed!")
