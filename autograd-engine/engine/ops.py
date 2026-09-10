from tensor import Tensor


# adds two tensors and contains a backward function too
def add(a: Tensor, b: Tensor) -> Tensor:
    # a, b are Tensor objects
    out_data = a.data + b.data

    out = Tensor(out_data, _children=(a, b), _op='add')

    def _backward():
        a.grad += 1.0 * out.grad
        b.grad += 1.0 * out.grad

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
        a.grad += b.data * out.grad
        b.grad += a.data * out.grad

    out._backward = _backward

    return out


# conducting matrix multiplication
def matmul(a: Tensor, b: Tensor) -> Tensor:
    out_data = a.data @ b.data

    return ...


# powers
def pow(a: Tensor, n: int) -> Tensor:
    out_data = a.data ** n

    out = Tensor(out_data, _children=(a,), _op='pow')

    def _backward():
        a.grad += (n * (a.data ** (n - 1))) * out.grad  # n*a^(n-1) * output gradient

    out._backward = _backward

    return out


# conducts relu
def relu(a: Tensor):
    out_data = a.data if a.data > 0 else 0

    out = Tensor(out_data, _children=(a,), _op='relu')

    def _backward():
        a.grad += (out_data > 0) * out.grad

    out._backward = _backward

    return out



