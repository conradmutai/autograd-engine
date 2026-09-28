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

        # running averages of the gradients and squared gradient
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0  # step count

    def step(self):
        self.t = self.t + 1
        for i, param in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * param.grad**2
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def zero_grad(self):
        for param in self.params:
            param.grad = 0.0
