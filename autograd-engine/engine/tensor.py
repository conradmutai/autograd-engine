# Translates the python syntax into engine language
class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0              # or zeros_like(data) once you support arrays
        self._prev = set(_children)  # parents in the graph, for topo sort later
        self._op = _op
        self._backward = lambda: None  # no-op by default (leaf nodes)

    def __add__(self, other):
        from ops import add
        return add(self, other)

    def __mul__(self, other):
        from ops import mul
        return mul(self, other)

    def __matmul__(self, other):
        ...

    def __pow__(self, other):
        from ops import pow
        return pow(self, other)


    def __relu__(self, other):
        from ops import relu
