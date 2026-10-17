import numpy as np

from .tensor import Tensor


# adds two tensors and contains a backward function too
def add(a: Tensor, b: Tensor) -> Tensor:
    # a, b are Tensor objects
    out_data = a.data + b.data

    out = Tensor(out_data, _children=(a, b), _op='add')

    def _backward():
        a.grad += _unbroadcast(out.grad, a.data.shape)
        b.grad += _unbroadcast(out.grad, b.data.shape)

    out._backward = _backward
    return out


def mul(a: Tensor, b: Tensor) -> Tensor:
    # a, b are Tensor objects
    out_data = a.data * b.data

    # initializes the out Tensor
    out = Tensor(out_data, _children=(a, b), _op='mul')

    # does the backward operation when .backward is called upon at the final loss
    def _backward():
        # accumulate the gradient
        a.grad += _unbroadcast(b.data * out.grad, a.data.shape)
        b.grad += _unbroadcast(a.data * out.grad, b.data.shape)

    out._backward = _backward

    return out


# conducting matrix multiplication
def matmul(a: Tensor, b: Tensor) -> Tensor:
    out_data = a.data @ b.data

    out = Tensor(out_data, _children=(a, b), _op='matmul')

    def _backward():
        a.grad += out.grad @ b.data.T
        b.grad += a.data.T @ out.grad

    out._backward = _backward

    return out


# powers
def pow(a: Tensor, n: int) -> Tensor:
    out_data = a.data ** n

    out = Tensor(out_data, _children=(a,), _op='pow')

    def _backward():
        a.grad += (n * (a.data ** (n - 1))) * out.grad  # n*a^(n-1) * output gradient

    out._backward = _backward

    return out


# conducts relu
def relu(a: Tensor) -> Tensor:
    out_data = np.maximum(0, a.data)

    out = Tensor(out_data, _children=(a,), _op='relu')

    def _backward():
        a.grad += (out_data > 0) * out.grad

    out._backward = _backward

    return out


# conducts softmax
def softmax(a: Tensor) -> Tensor:
    x = a.data
    shifted = x - np.max(x, axis=1, keepdims=True)  # numerical stability
    exp_x = np.exp(shifted)
    out_data = exp_x / exp_x.sum(axis=1, keepdims=True)

    out = Tensor(out_data, _children=(a,), _op='softmax')

    def _backward():
        # out_data: (batch_size, num_classes)
        # for each row, build the n x n Jacobian and apply it to that row's upstream grad
        batch_size, num_classes = out_data.shape
        grad = np.zeros_like(out_data)

        for b in range(batch_size):
            s = out_data[b].reshape(-1, 1)  # (num_classes, 1)
            jacobian = np.diagflat(s) - s @ s.T  # (num_classes, num_classes)
            grad[b] = jacobian @ out.grad[b]

        a.grad += grad

    out._backward = _backward
    return out


# cross entropy loss method (used to measure accuracy of classification model)
def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    x = logits.data
    batch = x.shape[0]
    assert targets.shape[0] == batch

    # subtract the max so exp doesn't blow up
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)

    # log softmax directly so we never take log of 0
    log_softmax = shifted - np.log(np.sum(exp_x, axis=1, keepdims=True))
    probs = np.exp(log_softmax)
    picked = log_softmax[np.arange(batch), targets]

    loss = np.mean(-picked)

    out = Tensor(loss, _children=(logits,), _op='cross_entropy')

    def _backward():
        # probs - one_hot, but only the correct class entries change
        grad = probs.copy()
        grad[np.arange(batch), targets] -= 1

        logits.grad += grad / batch * out.grad

    out._backward = _backward
    return out


# sums the data in a Tensor
def sum(a: Tensor) -> Tensor:
    out_data = np.sum(a.data)

    out = Tensor(out_data, _children=(a,), _op='sum')

    def _backward():
        # every element of a contributed equally to the total,
        # so upstream gradient flows back to every element unchanged
        a.grad += np.ones_like(a.data) * out.grad

    out._backward = _backward
    return out


# calculates the mean of the model
def mean(a: Tensor) -> Tensor:
    # sum divided by a.data.size
    out_data = np.sum(a.data)  / a.data.size

    out = Tensor(out_data, _children=(a,), _op='mean')

    # Backward is the sum backward scaled by 1/size
    def _backward():
        a.grad += out.grad * (1/a.data.size)

    out._backward = _backward
    return out


def _unbroadcast(grad, shape):
    # numpy adds extra leading axes when the smaller operand has fewer dims, sum those away
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # numpy also stretches any axis that was size 1, sum those back down (keepdims so the shape matches)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad