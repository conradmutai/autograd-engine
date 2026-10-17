import numpy as np

from engine.tensor import Tensor
from engine.ops import matmul, add


class Linear:
    def __init__(self, in_features: int, out_features: int):
        self.std = np.sqrt(2 / in_features)
        self.W = Tensor(np.random.normal(loc=0.0, scale=self.std, size=(in_features, out_features)))
        self.b = Tensor(np.zeros((out_features,), dtype=float))

    def __call__(self, x: Tensor) -> Tensor:
        # x is (batch, in), W is (in, out), so this gives (batch, out)
        pre_out = matmul(x, self.W)

        # bias is (out,), gets broadcast across every row (backward sums it back down with _unbroadcast)
        out = add(pre_out, self.b)
        return out

    def parameters(self) -> list[Tensor]:
        return [self.W, self.b]


class MLP:
    def __init__(self, sizes: list[int]):
        self.sizes = sizes
        self.layers = [Linear(i, o) for i, o in zip(sizes, sizes[1:])]

    def __call__(self, x: Tensor) -> Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x)

            # relu here, but only if idx isn't the last layer
            if idx < len(self.layers) - 1:
                x = x.relu()

        return x

    def parameters(self) -> list[Tensor]:
        return [p for layer in self.layers for p in layer.parameters()]