import numpy as np
from engine.tensor import Tensor
from engine.ops import sum as tensor_sum


# calculates the numerical gradient
def numerical_grad(f, x: np.ndarray, eps=1e-5) -> np.ndarray:
    res = np.zeros_like(x, dtype=np.float64)

    for i in range(x.size):
        orig = x[i]

        x[i] = orig + eps
        f_plus = f(x)

        x[i] = orig - eps
        f_minus = f(x)

        x[i] = orig  # restore back to original
        res[i] = (f_plus - f_minus) / (2 * eps)

    return res


# performs the gradient check to see if it has died
def gradcheck(op_fn, *inputs, eps=1e-5, threshold=1e-5):
    tensors = [Tensor(x.copy()) for x in inputs]
    out = op_fn(*tensors)

    loss = tensor_sum(out)
    loss.backward()

    for i, (t, x) in enumerate(zip(tensors, inputs)):
        def f(perturbed_x, i=i):
            # rebuild all inputs fresh; swap in perturbed_x only at position i
            fresh_tensors = [
                Tensor(perturbed_x.copy()) if j == i else Tensor(inputs[j].copy())
                for j in range(len(inputs))
            ]
            f_out = op_fn(*fresh_tensors)
            f_loss = tensor_sum(f_out)
            return f_loss.data

        num_grad = numerical_grad(f, x.copy(), eps)
        analytical_grad = t.grad

        rel_error = np.abs(analytical_grad - num_grad) / np.maximum(np.abs(analytical_grad), np.abs(num_grad) + 1e-8)
        max_rel_error = np.max(rel_error)
        print(f"input {i}: max rel error = {max_rel_error}")
        assert max_rel_error < threshold


if __name__ == "__main__":
    gradcheck(lambda a, b: a + b, np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))