import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tinygrad.tensor import Tensor
from tinygrad.nn import GPT
from train import compute_loss


def test_sum_mean_axis_grads_have_input_shape():
    x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
    (x.sum(axis=-1, keepdims=True) * Tensor(np.ones((2, 3, 1)))).sum().backward()
    assert x._grad.shape == (2, 3, 4) and np.allclose(x._grad, 1)
    x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
    x.mean(axis=-1, keepdims=True).sum().backward()
    assert np.allclose(x._grad, 0.25)


def test_loss_is_cross_entropy():
    np.random.seed(0)
    model = GPT(vocab_size=7, n_layers=1, dim=8, n_heads=2, head_dim=4, hidden_dim=16, max_seq_len=8)
    x = np.random.randint(0, 7, (2, 5)); y = np.random.randint(0, 7, (2, 5))
    loss = float(compute_loss(model, Tensor(x), Tensor(y)).data)
    logits = model(Tensor(x)).data
    lp = logits - np.log(np.exp(logits - logits.max(-1, keepdims=True)).sum(-1, keepdims=True)) - logits.max(-1, keepdims=True)
    ref = -np.mean(np.take_along_axis(lp, y[..., None], -1))
    assert loss > 0 and abs(loss - ref) < 1e-9


def test_whole_model_gradients_match_finite_differences():
    np.random.seed(1)
    model = GPT(vocab_size=7, n_layers=1, dim=8, n_heads=2, head_dim=4, hidden_dim=16, max_seq_len=8)
    x = Tensor(np.random.randint(0, 7, (2, 5))); y = Tensor(np.random.randint(0, 7, (2, 5)))
    params = model.parameters()
    for p in params:
        p.zero_grad()
    compute_loss(model, x, y).backward()
    checked = 0
    for p in params:
        flat = p.data.reshape(-1)
        for i in np.random.choice(flat.size, min(3, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + 1e-5; lp = float(compute_loss(model, x, y).data)
            flat[i] = old - 1e-5; lm = float(compute_loss(model, x, y).data)
            flat[i] = old
            num = (lp - lm) / 2e-5
            ana = p._grad.reshape(-1)[i]
            assert abs(num - ana) < 1e-4 + 1e-3 * abs(num), (p.data.shape, i, num, ana)
            checked += 1
    assert checked > 10


def test_transpose_2d_and_3d():
    a = np.arange(6.).reshape(2, 3)
    assert Tensor(a).transpose().data.shape == (3, 2)
    x = Tensor(np.random.randn(2, 3, 4))
    y = x.transpose()
    assert y.data.shape == (2, 4, 3)
    (y * Tensor(np.ones((2, 4, 3)))).sum().backward()
    assert x._grad.shape == (2, 3, 4)
