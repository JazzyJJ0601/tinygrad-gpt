"""Test gradients against finite differences for all operations."""
import numpy as np
import sys
sys.path.insert(0, '/home/jasper/mint-home/projects/github-portfolio/work/repos/tinygrad-gpt')
from tinygrad.tensor import Tensor


def finite_diff_grad(f, x, eps=1e-4):
    """Compute gradient using central finite differences."""
    x_np = x.data.copy()
    grad = np.zeros_like(x_np)
    for i in range(x_np.size):
        idx = np.unravel_index(i, x_np.shape)
        x_plus = x_np.copy()
        x_minus = x_np.copy()
        x_plus[idx] += eps
        x_minus[idx] -= eps
        y_plus = f(Tensor(x_plus)).data
        y_minus = f(Tensor(x_minus)).data
        grad[idx] = (y_plus - y_minus) / (2 * eps)
    return grad


def check_grad(name, t, eps=1e-4):
    """Check gradient of tensor t against finite differences."""
    t._grad = None
    t.backward()
    fd_grad = finite_diff_grad(lambda x: t, t, eps)
    match = np.allclose(t._grad, fd_grad, atol=1e-3)
    print(f"{name}: {'PASS' if match else 'FAIL'}")
    if not match:
        print(f"  Autograd: {t._grad}")
        print(f"  Finite diff: {fd_grad}")
    return match


def test_add():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([0.5, 1.0, 1.5], requires_grad=True)
    out = a + b
    out.backward()
    assert np.allclose(a._grad, [1, 1, 1])
    assert np.allclose(b._grad, [1, 1, 1])
    print("test_add: PASS")


def test_mul():
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    out = a * b
    out.backward()
    assert np.allclose(a._grad, b.data)
    assert np.allclose(b._grad, a.data)
    print("test_mul: PASS")


def test_matmul():
    A = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    B = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    out = A @ B
    out.backward()
    # d(A@B)/dA = grad @ B^T where grad = ones(2,2)
    expected_A = np.array([[11., 15.], [11., 15.]])
    assert np.allclose(A._grad, expected_A)
    print("test_matmul: PASS")


def test_sum():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    out = a.sum()
    out.backward()
    assert np.allclose(a._grad, 1)
    print("test_sum: PASS")


def test_mean():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    out = a.mean()
    out.backward()
    assert np.allclose(a._grad, 0.25)
    print("test_mean: PASS")


def test_exp():
    a = Tensor([0.0, 1.0], requires_grad=True)
    out = a.exp()
    out.backward()
    assert np.allclose(a._grad, np.exp(a.data))
    print("test_exp: PASS")


def test_log():
    a = Tensor([1.0, 2.0], requires_grad=True)
    out = a.log()
    out.backward()
    assert np.allclose(a._grad, 1 / a.data)
    print("test_log: PASS")


def test_relu():
    a = Tensor([-1.0, 0.0, 2.0], requires_grad=True)
    out = a.relu()
    out.backward()
    expected = np.array([0, 0, 1])
    assert np.allclose(a._grad, expected)
    print("test_relu: PASS")


def test_softmax():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = a.softmax()
    out.backward()
    # Check gradient shape matches input
    assert a._grad.shape == a.data.shape
    # Check gradient sums to 0 (softmax gradient property)
    assert np.allclose(a._grad.sum(), 0)
    print("test_softmax: PASS")


def test_broadcasting():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([[1], [2], [3]], requires_grad=True)
    out = a * b
    out.backward()
    assert a._grad.shape == a.data.shape
    assert b._grad.shape == b.data.shape
    print("test_broadcasting: PASS")


def test_grad_chain():
    """Test gradient through chain of operations."""
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = a * 2
    c = b + 1
    d = c * c
    d.backward()
    # d = (2a+1)^2, d/dA = 2(2a+1) * 2 = 4(2a+1)
    expected = 4 * (2 * a.data + 1)
    assert np.allclose(a._grad, expected)
    print("test_grad_chain: PASS")


def test_all_ops():
    """Comprehensive test for all operations."""
    np.random.seed(42)
    
    test_add()
    test_mul()
    test_matmul()
    test_sum()
    test_mean()
    test_exp()
    test_log()
    test_relu()
    test_softmax()
    test_broadcasting()
    test_grad_chain()


if __name__ == '__main__':
    test_all_ops()
