# Translates the python syntax into engine language
class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0              # or zeros_like(data) once you support arrays
        self._prev = set(_children)  # parents in the graph, for topo sort later
        self._op = _op
        self._backward = lambda: None  # no-op by default (leaf nodes)

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(data={self.data}, op='{self._op}')"

    @staticmethod
    def _wrap(x):
        # lets `x * 2` and `x + 1` work with plain numbers/arrays
        return x if isinstance(x, Tensor) else Tensor(x)

    def __add__(self, other):
        from .ops import add
        return add(self, other)

    def __mul__(self, other):
        from .ops import mul
        return mul(self, other)

    def __matmul__(self, other):
        from .ops import matmul
        return matmul(self, other)

    def __pow__(self, other):
        from .ops import pow
        return pow(self, other)

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __truediv__(self, other):
        return self * (self._wrap(other) ** -1)

    # reflected versions: make `2 * x`, `1 + x`, `1 - x`, `1 / x` work
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return self._wrap(other) + (-self)

    def __rtruediv__(self, other):
        return self._wrap(other) * (self ** -1)

    def relu(self):
        from .ops import relu
        return relu(self)

    def softmax(self):
        from .ops import softmax
        return softmax(self)

    def sum(self):
        from engine.ops import sum as sum_op
        return sum_op(self)

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()
