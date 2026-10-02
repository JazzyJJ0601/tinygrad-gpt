#!/usr/bin/env python3
"""Compare tinygrad with PyTorch gradient computation (skip gracefully if torch missing)."""

import os
import sys

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Skipping comparison.")
    print("Install with: pip install torch")


def compare_gradients():
    """Compare gradient computation between tinygrad and torch."""
    if not TORCH_AVAILABLE:
        return
    
    print("=" * 60)
    print("Tinygrad vs PyTorch Gradient Comparison")
    print("=" * 60)
    
    # Add the tinygrad package to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    import numpy as np
    from tinygrad.tensor import Tensor
    
    # Test case: simple matrix multiplication and element-wise operations
    print("\n1. Testing basic operations:")
    print("-" * 60)
    
    # Create test data
    np.random.seed(42)
    x_np = np.random.randn(3, 4).astype(np.float64)
    y_np = np.random.randn(4, 5).astype(np.float64)
    
    # Tinygrad computation
    x_tg = Tensor(x_np)
    y_tg = Tensor(y_np)
    z_tg = x_tg.matmul(y_tg).relu()
    loss_tg = z_tg.sum()
    loss_tg.backward()
    
    # PyTorch computation
    x_torch = torch.from_numpy(x_np).requires_grad_(True)
    y_torch = torch.from_numpy(y_np).requires_grad_(True)
    z_torch = torch.relu(torch.matmul(x_torch, y_torch))
    loss_torch = z_torch.sum()
    loss_torch.backward()
    
    print(f"Tinygrad loss: {loss_tg.data:.6f}")
    print(f"PyTorch loss:  {loss_torch.item():.6f}")
    print(f"Loss difference: {abs(loss_tg.data - loss_torch.item()):.8f}")
    
    # Compare gradients
    dx = np.abs(x_tg._grad - x_torch.grad.numpy()).max()
    dy = np.abs(y_tg._grad - y_torch.grad.numpy()).max()
    print(f"Max |grad difference|: x {dx:.2e}, y {dy:.2e}")
    print("\n" + "=" * 60)
    print("Comparison complete!")
    print("=" * 60)


if __name__ == '__main__':
    compare_gradients()
