import numpy as np
import matplotlib.pyplot as plt

from engine.tensor import Tensor
from engine.ops import cross_entropy
from engine.nn import MLP
from engine.optim import SGD, Adam


# creates a dataset
def make_dataset(n_per_class: int, n_classes: int, noise: float = 0.2):
    N = n_per_class * n_classes
    X = np.zeros((N, 2))                 # one row per point, columns are (x1, x2)
    y = np.zeros(N, dtype=int)           # one label per point

    for k in range(n_classes):
        idx = range(n_per_class * k, n_per_class * (k + 1))   # rows that belong to class k

        # radius grows from the center outward
        r = np.linspace(0.0, 1.0, n_per_class)

        # angle sweeps along the arm, each class starts where the last one ended
        theta = np.linspace(k * 4, (k + 1) * 4, n_per_class) + np.random.randn(n_per_class) * noise

        # polar to cartesian
        X[idx] = np.c_[r * np.sin(theta), r * np.cos(theta)]
        y[idx] = k

    return X, y


# calculates the model accuracy
def accuracy(model, X, y) -> float:
    logits = model(Tensor(X))
    y_preds = np.argmax(logits.data, axis=1)

    true, count = 0, y_preds.size

    # compares predicted to each actual result
    for idx in range(y_preds.size):
        if y_preds[idx] == y[idx]:
            true += 1

    acc = true / count
    return float(acc)


# trains the model
def train(model, optimizer, X, y, epochs, batch_size):
    losses = []   # average loss per epoch, for the README curves

    for e in range(epochs):
        # one shuffle for both X and y so every point keeps its label
        indices = np.random.permutation(X.shape[0])
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, X.shape[0], batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            logits = model(Tensor(X_batch))
            loss = cross_entropy(logits, y_batch)

            # zero first, otherwise gradients from the last batch pile up
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.data)
            n_batches += 1

        losses.append(epoch_loss / n_batches)

        if e % 10 == 0:
            acc = accuracy(model, X, y)
            print(f"Epoch: {e + 1}   loss: {losses[-1]:.4f}   acc: {acc:.3f}")

    return losses


if __name__ == "__main__":
    # train and test sets: different seeds so the noise differs
    np.random.seed(0)
    X, y = make_dataset(100, 3)
    np.random.seed(1)
    X_test, y_test = make_dataset(100, 3)

    sizes = [2, 16, 16, 3]
    results = {}

    for name in ["SGD", "Adam"]:
        # same seed before building the model, so both start from identical weights
        np.random.seed(42)
        model = MLP(sizes)

        if name == "SGD":
            optimizer = SGD(model.parameters(), lr=0.1)
        else:
            optimizer = Adam(model.parameters(), lr=0.01)

        # reseed again so both see the same shuffle order
        np.random.seed(7)
        losses = train(model, optimizer, X, y, epochs=200, batch_size=32)

        results[name] = losses
        print(f"{name}: train acc {accuracy(model, X, y):.3f}   test acc {accuracy(model, X_test, y_test):.3f}")

    # loss curves for the README
    for name, losses in results.items():
        plt.plot(losses, label=name)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("SGD vs Adam on the spiral dataset")
    plt.legend()
    plt.savefig("loss_curves.png", dpi=120)
    plt.show()