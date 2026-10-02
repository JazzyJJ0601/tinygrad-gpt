"""From-scratch reverse-mode autodiff Tensor over NumPy arrays."""
import numpy as np


def _softmax_np(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def _reduce(grad, shape):
    """Reduce gradient shape to match target shape by summing over extra/broadcast dims."""
    # First, sum over leading extra dimensions
    while len(grad.shape) > len(shape):
        grad = np.sum(grad, axis=0)
    
    # Now sum over dimensions where sizes differ (broadcasting)
    for i in range(len(shape)):
        if grad.shape[i] != shape[i]:
            grad = np.sum(grad, axis=i, keepdims=True)
    
    return grad


def _as_tensor(x):
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


class Tensor:
    """A tensor with reverse-mode automatic differentiation."""

    def __init__(self, data, requires_grad=False):
        if isinstance(data, Tensor):
            self.data = data.data
            self._grad = data._grad
            self._prev = data._prev
            self._ctx = data._ctx
            self._op = data._op
            self._parents = data._parents
        else:
            self.data = np.array(data, dtype=np.float64)
            self._grad = None
            self._prev = []
            self._ctx = None
            self._op = None
            self._parents = []
            if requires_grad:
                self._grad = np.zeros_like(self.data)

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor({self.data.shape})"

    def zero_grad(self):
        self._grad = None

    def backward(self):
        if self._grad is None:
            self._grad = np.ones_like(self.data)
        
        topo = []
        visited = []
        
        def build(t):
            if t in visited:
                return
            visited.append(t)
            for p in t._parents:
                build(p)
            topo.append(t)
        
        build(self)
        
        for t in reversed(topo):
            if t._ctx is not None and t._op is not None:
                grads = t._ctx.backward(t._grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)
                for parent, grad in zip(t._parents, grads):
                    if grad is not None and grad.size > 0:
                        if parent._grad is None:
                            parent._grad = np.zeros_like(parent.data)
                        parent._grad = parent._grad + _reduce(grad, parent.data.shape)

    def __neg__(self):
        out = Tensor(-self.data)
        out._ctx = NegCtx(self)
        out._op = 'neg'
        out._parents.append(self)
        return out

    def __add__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data + other.data)
        out._ctx = AddCtx(self, other)
        out._op = 'add'
        out._parents.append(self)
        out._parents.append(other)
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data - other.data)
        out._ctx = SubCtx(self, other)
        out._op = 'sub'
        out._parents.append(self)
        out._parents.append(other)
        return out

    def __rsub__(self, other):
        other = _as_tensor(other)
        out = Tensor(other.data - self.data)
        out._ctx = SubCtx(other, self)
        out._op = 'sub'
        out._parents.append(other)
        out._parents.append(self)
        return out

    def __mul__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data * other.data)
        out._ctx = MulCtx(self, other)
        out._op = 'mul'
        out._parents.append(self)
        out._parents.append(other)
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data / other.data)
        out._ctx = DivCtx(self, other)
        out._op = 'div'
        out._parents.append(self)
        out._parents.append(other)
        return out

    def __rtruediv__(self, other):
        other = _as_tensor(other)
        out = Tensor(other.data / self.data)
        out._ctx = DivCtx(other, self)
        out._op = 'div'
        out._parents.append(other)
        out._parents.append(self)
        return out

    def __pow__(self, other):
        other = _as_tensor(other)
        out = Tensor(np.power(self.data, other.data))
        out._ctx = PowCtx(self, other)
        out._op = 'pow'
        out._parents.append(self)
        out._parents.append(other)
        return out

    def matmul(self, other):
        other = _as_tensor(other)
        out = Tensor(np.matmul(self.data, other.data))
        out._ctx = MatmulCtx(self, other)
        out._op = 'matmul'
        out._parents.append(self)
        out._parents.append(other)
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def __rmatmul__(self, other):
        other = _as_tensor(other)
        return other.matmul(self)

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims))
        out._ctx = SumCtx(self, axis=axis, keepdims=keepdims)
        out._op = 'sum'
        out._parents.append(self)
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims))
        out._ctx = MeanCtx(self, axis=axis, keepdims=keepdims)
        out._op = 'mean'
        out._parents.append(self)
        return out

    def exp(self):
        out = Tensor(np.exp(self.data))
        out._ctx = ExpCtx(self)
        out._op = 'exp'
        out._parents.append(self)
        return out

    def log(self):
        out = Tensor(np.log(self.data))
        out._ctx = LogCtx(self)
        out._op = 'log'
        out._parents.append(self)
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0))
        out._ctx = ReluCtx(self)
        out._op = 'relu'
        out._parents.append(self)
        return out

    def softmax(self, axis=-1):
        out = Tensor(_softmax_np(self.data, axis=axis))
        out._ctx = SoftmaxCtx(self, axis=axis)
        out._op = 'softmax'
        out._parents.append(self)
        return out

    def reshape(self, *shape):
        # shape can be passed as (*dims) or (tuple_of_dims,)
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        out = Tensor(np.reshape(self.data, shape))
        out._ctx = ReshapeCtx(self, shape)
        out._op = 'reshape'
        out._parents.append(self)
        return out

    def transpose(self, axis1=-2, axis2=-1):
        out = Tensor(np.swapaxes(self.data, axis1, axis2))
        out._ctx = TransposeCtx(self, axis1, axis2)
        out._op = 'transpose'
        out._parents.append(self)
        return out


# Context classes for gradient computation

class NegCtx:
    def __init__(self, t):
        self.t = t

    def backward(self, grad):
        return -grad


class AddCtx:
    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def backward(self, grad):
        return _reduce(grad, self.t1.shape), _reduce(grad, self.t2.shape)


class SubCtx:
    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def backward(self, grad):
        return _reduce(grad, self.t1.shape), _reduce(-grad, self.t2.shape)


class MulCtx:
    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def backward(self, grad):
        g1 = _reduce(grad * self.t2.data, self.t1.shape)
        g2 = _reduce(grad * self.t1.data, self.t2.shape)
        return g1, g2


class DivCtx:
    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def backward(self, grad):
        g1 = _reduce(grad / self.t2.data, self.t1.shape)
        g2 = _reduce(-grad * self.t1.data / (self.t2.data ** 2), self.t2.shape)
        return g1, g2


class PowCtx:
    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def backward(self, grad):
        g1 = _reduce(grad * self.t2.data * (self.t1.data ** (self.t2.data - 1)), self.t1.shape)
        g2 = _reduce(grad * (self.t1.data ** self.t2.data) * np.log(self.t1.data), self.t2.shape)
        return g1, g2


class MatmulCtx:
    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def backward(self, grad):
        if self.t1.data.ndim == 1 and self.t2.data.ndim == 1:
            grad1 = grad * self.t2.data
            grad2 = grad * self.t1.data
        elif self.t1.data.ndim == 2 and self.t2.data.ndim == 2:
            grad1 = np.matmul(grad, self.t2.data.T)
            grad2 = np.matmul(self.t1.data.T, grad)
        else:
            # For higher dimensional matmul, broadcast appropriately
            # grad shape should be broadcast-compatible with output
            # grad1 = grad @ t2^T (over last two axes)
            # grad2 = t1^T @ grad (over last two axes)
            if self.t1.data.ndim >= 2 and self.t2.data.ndim >= 2:
                grad1 = np.matmul(grad, self.t2.data.swapaxes(-1, -2))
                grad2 = np.matmul(self.t1.data.swapaxes(-1, -2), grad)
            else:
                grad1 = grad
                grad2 = grad
        return _reduce(grad1, self.t1.shape), _reduce(grad2, self.t2.shape)


class SumCtx:
    def __init__(self, t, axis=None, keepdims=False):
        self.t = t
        self.axis = axis
        self.keepdims = keepdims

    def backward(self, grad):
        if self.axis is None:
            return np.full_like(self.t.data, grad)
        else:
            if not self.keepdims:
                grad = np.expand_dims(grad, axis=self.axis)
            return np.broadcast_to(grad, self.t.data.shape).copy()


class MeanCtx:
    def __init__(self, t, axis=None, keepdims=False):
        self.t = t
        self.axis = axis
        self.keepdims = keepdims

    def backward(self, grad):
        if self.axis is None:
            return np.full_like(self.t.data, grad / self.t.data.size)
        else:
            if not self.keepdims:
                grad = np.expand_dims(grad, axis=self.axis)
            n = np.prod([self.t.data.shape[a] for a in np.atleast_1d(self.axis)])
            return np.broadcast_to(grad / n, self.t.data.shape).copy()


class ExpCtx:
    def __init__(self, t):
        self.t = t

    def backward(self, grad):
        return grad * np.exp(self.t.data)


class LogCtx:
    def __init__(self, t):
        self.t = t

    def backward(self, grad):
        return grad / self.t.data


class ReluCtx:
    def __init__(self, t):
        self.t = t

    def backward(self, grad):
        return grad * (self.t.data > 0)


class SoftmaxCtx:
    def __init__(self, t, axis=-1):
        self.t = t
        self.axis = axis

    def backward(self, grad):
        s = _softmax_np(self.t.data, axis=self.axis)
        inner = np.sum(grad * s, axis=self.axis, keepdims=True)
        return s * (grad - inner)


class TransposeCtx:
    def __init__(self, t, axis1=-2, axis2=-1):
        self.t = t
        self.axis1 = axis1
        self.axis2 = axis2

    def backward(self, grad):
        return np.swapaxes(grad, self.axis1, self.axis2)


class ReshapeCtx:
    def __init__(self, t, shape):
        self.t = t
        self.shape = shape

    def backward(self, grad):
        return np.reshape(grad, self.t.data.shape)
