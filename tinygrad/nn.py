"""Neural network layers built on top of Tensor."""
import numpy as np
from tinygrad.tensor import Tensor


def _uniform(shape, scale=0.1):
    """Xavier-style uniform initialization."""
    return Tensor(np.random.uniform(-scale, scale, size=shape))


def _zeros(shape):
    """Zero initialization."""
    return Tensor(np.zeros(shape))


class Linear:
    """Linear layer: y = x @ W + b"""
    
    def __init__(self, in_features, out_features, bias=True):
        scale = 1.0 / np.sqrt(in_features)
        self.weight = _uniform((in_features, out_features), scale=scale)
        self.bias = _zeros((out_features,)) if bias else None
        self._weights = [self.weight]
        if self.bias is not None:
            self._weights.append(self.bias)
    
    def forward(self, x):
        # x: (batch, in_features), out: (batch, out_features)
        out = x.matmul(self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out
    
    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        return self._weights


class LayerNorm:
    """Layer normalization."""
    
    def __init__(self, normalized_shape, eps=1e-5):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = Tensor(np.ones(normalized_shape))
        self.beta = Tensor(np.zeros(normalized_shape))
        self._weights = [self.gamma, self.beta]
    
    def forward(self, x):
        # x: (batch, ..., d_model)
        # Compute mean and var over the normalized dimensions
        reduce_axes = tuple(range(len(x.shape) - len(self.normalized_shape), len(x.shape)))
        mean = x.mean(axis=reduce_axes, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=reduce_axes, keepdims=True)
        
        # Normalize: (x - mean) / sqrt(var + eps)
        x_norm = (x - mean) / ((var + self.eps) ** 0.5)
        
        # Scale and shift
        out = x_norm * self.gamma + self.beta
        return out
    
    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        return self._weights


class Attention:
    """Causal self-attention."""
    
    def __init__(self, dim, n_heads, head_dim, dropout=0.0):
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = 1.0 / np.sqrt(head_dim)
        
        # Q, K, V projections
        self.q_proj = Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = Linear(dim, n_heads * head_dim, bias=False)
        self.v_proj = Linear(dim, n_heads * head_dim, bias=False)
        
        # Output projection
        self.out_proj = Linear(n_heads * head_dim, dim, bias=False)
        
        self._weights = self.q_proj.parameters() + self.k_proj.parameters() + \
                        self.v_proj.parameters() + self.out_proj.parameters()
    
    def forward(self, x, causal_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # For simplicity, don't do multi-head attention with reshaping
        # Just use the full dim directly
        # Compute attention: (batch, seq, dim) @ (batch, dim, seq) = (batch, seq, seq)
        scores = q.matmul(k.transpose()) * self.scale
        
        # Apply causal mask - compute on the fly for actual sequence length
        if causal_mask is None:
            causal_mask = np.tril(np.ones((seq_len, seq_len)))
        scores = scores + Tensor(causal_mask * 1e9)
        
        # Softmax and attention
        attn = scores.softmax(axis=-1)
        
        # Apply attention to values
        out = attn.matmul(v)
        
        # Output projection
        out = self.out_proj(out)
        return out
    
    def __call__(self, x, causal_mask=None):
        return self.forward(x, causal_mask)
    
    def parameters(self):
        return self._weights


class FeedForward:
    """MLP block in transformer."""
    
    def __init__(self, dim, hidden_dim):
        self.up = Linear(dim, hidden_dim, bias=False)
        self.down = Linear(hidden_dim, dim, bias=False)
        self._weights = self.up.parameters() + self.down.parameters()
    
    def forward(self, x):
        # Simplified MLP with relu
        out = self.up(x).relu()
        out = self.down(out)
        return out
    
    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        return self._weights


class TransformerBlock:
    """Single transformer block with pre-norm."""
    
    def __init__(self, dim, n_heads, head_dim, hidden_dim):
        self.attn_norm = LayerNorm(dim)
        self.attn = Attention(dim, n_heads, head_dim)
        self.mlp_norm = LayerNorm(dim)
        self.mlp = FeedForward(dim, hidden_dim)
        
        self._weights = self.attn_norm.parameters() + self.attn.parameters() + \
                        self.mlp_norm.parameters() + self.mlp.parameters()
    
    def forward(self, x, causal_mask=None):
        # Pre-norm attention with residual
        normed = self.attn_norm(x)
        attn_out = self.attn(normed, causal_mask)
        x = x + attn_out
        
        # Pre-norm MLP with residual
        normed = self.mlp_norm(x)
        mlp_out = self.mlp(normed)
        x = x + mlp_out
        
        return x
    
    def __call__(self, x, causal_mask=None):
        return self.forward(x, causal_mask)
    
    def parameters(self):
        return self._weights


class GPT:
    """Simple GPT model."""
    
    def __init__(self, vocab_size, n_layers, dim, n_heads, head_dim, hidden_dim, max_seq_len=512):
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Embedding: lookup table
        self.tok_emb = _uniform((vocab_size, dim))
        
        # Blocks
        self.blocks = [
            TransformerBlock(dim, n_heads, head_dim, hidden_dim)
            for _ in range(n_layers)
        ]
        
        # Output layer
        self.ln_f = LayerNorm(dim)
        self.out_proj = Linear(dim, vocab_size, bias=False)
        
        self._weights = [self.tok_emb]
        for b in self.blocks:
            self._weights.extend(b.parameters())
        self._weights.extend(self.ln_f.parameters())
        self._weights.extend(self.out_proj.parameters())
    
    def forward(self, x):
        # x: (batch, seq_len) tensor of token indices (as numpy array converted to Tensor)
        batch_size, seq_len = x.shape
        
        # Token embeddings via one-hot
        x_np = x.data if hasattr(x, 'data') else x
        x_onehot = np.zeros((batch_size, seq_len, self.vocab_size), dtype=np.float64)
        for i in range(batch_size):
            for j in range(seq_len):
                x_onehot[i, j, int(x_np[i, j])] = 1.0
        x = Tensor(x_onehot).matmul(self.tok_emb)
        
        # Build causal mask once - compute on the fly for actual sequence length
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        
        # Pass through blocks
        for block in self.blocks:
            x = block(x, causal_mask)
        
        # Final norm and output
        x = self.ln_f(x)
        x = self.out_proj(x)
        
        return x
    
    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        return self._weights
