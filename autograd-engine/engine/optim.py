import numpy as np

from tensor import Tensor


class SGD:
    def __init__(self, params: list[Tensor], lr: float = 0.001):
        self.params = params  # takes in a list of tensors of weights and biases
        self.lr = lr

    # looking for the optimal gradient
    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad

    # sets all parameters gradients back to zero
    def zero_grad(self):
        for param in self.params:
            param.grad = 0.0


class Adam:
    def __init__(self, params: list[Tensor], lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        ...

    def zero_grad(self):
        ...
