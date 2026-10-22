import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from engine.tensor import Tensor
from engine.ops import cross_entropy
from engine.nn import MLP
from engine.optim import SGD, Adam
from examples.train_mlp import make_dataset


# builds a torch model with the same layers as ours and copies our weights in
def build_torch_equivalent(my_model):
    layers = []
    linears = []  # kept separate so we can pair them with our layers when copying
    n_layers = len(my_model.layers)

    for idx, layer in enumerate(my_model.layers):
        in_features, out_features = layer.W.data.shape
        linear = nn.Linear(in_features, out_features)
        linears.append(linear)
        layers.append(linear)

        # relu between layers but not after the last one, same as our MLP
        if idx < n_layers - 1:
            layers.append(nn.ReLU())

    # our engine is float64 and torch defaults to float32
    torch_model = nn.Sequential(*layers).double()

    with torch.no_grad():
        for mine, theirs in zip(my_model.layers, linears):
            # our W is (in, out) but torch stores (out, in), so transpose
            theirs.weight.copy_(torch.from_numpy(mine.W.data.T.copy()))
            theirs.bias.copy_(torch.from_numpy(mine.b.data.copy()))

    return torch_model


# builds our optimizer and the matching torch one with the same settings
def make_optimizers(name, my_model, torch_model, lr):
    if name == "SGD":
        return SGD(my_model.parameters(), lr=lr), torch.optim.SGD(torch_model.parameters(), lr=lr)
    return (Adam(my_model.parameters(), lr=lr),
            torch.optim.Adam(torch_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8))


# returns a function that does one full training step with our engine
def make_my_step(model, optimizer, X, y):
    def step():
        loss = cross_entropy(model(Tensor(X)), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.data)
    return step


# returns a function that does one full training step with torch
def make_torch_step(model, optimizer, X, y):
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).long()
    loss_fn = nn.CrossEntropyLoss()

    def step():
        loss = loss_fn(model(X_t), y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.item())
    return step


# calculates the average seconds per step
def time_training(step_fn, n_steps, warmup=5) -> float:
    # first few calls are slower so they don't get timed
    for _ in range(warmup):
        step_fn()

    start = time.perf_counter()
    for _ in range(n_steps):
        step_fn()
    elapsed = time.perf_counter() - start

    return elapsed / n_steps


# trains both models from the same weights on the same data and returns both loss curves
def compare_loss_curves(sizes, X, y, optimizer_name, lr, n_steps):
    np.random.seed(42)
    my_model = MLP(sizes)
    torch_model = build_torch_equivalent(my_model)
    my_opt, torch_opt = make_optimizers(optimizer_name, my_model, torch_model, lr)

    my_step = make_my_step(my_model, my_opt, X, y)
    torch_step = make_torch_step(torch_model, torch_opt, X, y)

    my_losses = [my_step() for _ in range(n_steps)]
    torch_losses = [torch_step() for _ in range(n_steps)]
    return my_losses, torch_losses


if __name__ == "__main__":
    np.random.seed(0)
    X, y = make_dataset(100, 3)
    sizes = [2, 16, 16, 3]

    # 1. same weights in should give the same logits out
    np.random.seed(42)
    my_model = MLP(sizes)
    torch_model = build_torch_equivalent(my_model)
    mine = my_model(Tensor(X)).data
    theirs = torch_model(torch.from_numpy(X)).detach().numpy()
    print(f"logits match: {np.allclose(mine, theirs)}   max diff: {np.max(np.abs(mine - theirs)):.2e}")

    # 2. the loss curves should track each other for both optimizers
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, lr) in zip(axes, [("SGD", 0.1), ("Adam", 0.01)]):
        my_losses, torch_losses = compare_loss_curves(sizes, X, y, name, lr, n_steps=200)
        diff = np.max(np.abs(np.array(my_losses) - np.array(torch_losses)))
        print(f"{name}: final loss mine {my_losses[-1]:.4f}  torch {torch_losses[-1]:.4f}  max curve diff {diff:.2e}")
        ax.plot(my_losses, label="mine")
        ax.plot(torch_losses, "--", label="torch")
        ax.set_title(name)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend()
    plt.tight_layout()
    plt.savefig("benchmark_curves.png", dpi=120)

    # 3. time per step on a tiny network and a wider one
    # under 1.0x means ours is faster, over 1.0x means torch is faster
    print(f"\n{'sizes':>18} {'batch':>6} {'mine (ms)':>11} {'torch (ms)':>11} {'mine/torch':>11}")
    for bench_sizes, batch in [([2, 16, 16, 3], 32), ([2, 16, 16, 3], 300), ([2, 512, 512, 3], 300)]:
        np.random.seed(42)
        my_model = MLP(bench_sizes)
        torch_model = build_torch_equivalent(my_model)
        my_opt, torch_opt = make_optimizers("Adam", my_model, torch_model, 0.01)

        my_t = time_training(make_my_step(my_model, my_opt, X[:batch], y[:batch]), n_steps=50)
        torch_t = time_training(make_torch_step(torch_model, torch_opt, X[:batch], y[:batch]), n_steps=50)
        print(f"{str(bench_sizes):>18} {batch:>6} {my_t * 1000:>11.3f} {torch_t * 1000:>11.3f} {my_t / torch_t:>10.2f}x")

    plt.show()