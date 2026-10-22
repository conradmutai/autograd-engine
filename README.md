# Autograd Engine
Creating a reverse-mode automatic differentiation engine without the usage of autograd libraries such as **PyTorch** or **TensorFlow**, while mainly utilizing **NumPy**. Implements the computation graph, backpropagation, gradient checking, and gradient descent following essential principles.

## What This Is
A small automatic differentiation engine, in the spirit of Andrej Karpathy's micrograd but working on NumPy arrays instead of single numbers, made without the use of any external autograd library. A `Tensor` records how it was created, so calling `.backward()` on a loss walks the graph in reverse and fills in the gradient of every weight. On top of the engine sit a small neural network library (`Linear` and `MLP`), two optimizers (SGD and Adam), a numerical gradient checker, and a training example on a 3-class spiral dataset. This project was created as a display and test of my knowledge of how backpropagation actually works underneath libraries like PyTorch, rather than just trusting `loss.backward()`. In addition, it is benchmarked against an equivalent PyTorch model, first for correctness and then for speed.

## Architecture
Computation graph: Tensor -> Operation (forward result + a `_backward` closure) -> new Tensor -> ... -> loss -> `backward()` (topological sort, then each `_backward` in reverse) -> `optimizer.step()`

Differentiable operations: add, mul, pow, relu, matmul, softmax, sum, mean, cross_entropy

Model used for training: Linear(2, 16) -> ReLU -> Linear(16, 16) -> ReLU -> Linear(16, 3) -> CrossEntropyLoss

**Parameters for Model**
- sizes = [2, 16, 16, 3] (371 trainable parameters)
- dataset = 3-class spiral, 100 points per class, noise = 0.2
- batch_size = 32
- epochs = 200
- SGD: lr = 0.1
- Adam: lr = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8

## Results
### Metric
Test Accuracy and Train Accuracy in %, Loss in decimal (average loss over the final epoch).

### This Model
| Optimizer | Train Accuracy | Test Accuracy | Final Epoch Loss |
| --- | --- | --- | --- |
| SGD (lr = 0.1) | 92.0% | 91.0% | 0.056 |
| Adam (lr = 0.01) | 99.3% | 99.3% | 0.035 |

The loss starts near ln(3) ≈ 1.1, which is what a 3-class model that is guessing should give. SGD's accuracy bounces between roughly 92% and 99% from epoch to epoch late in training, so its final number is a snapshot rather than a stable value. The test set is generated with the same function as the training set but a different seed, meaning it has the same underlying spirals with different noise, which makes it a weaker test than data from a different distribution.

![Loss per epoch](results/loss_curves.png)

### Gradient Checking
Every operation's backward pass is compared against a numerical gradient (centered difference, eps = 1e-5). Max relative error:

| Operation | Max Relative Error |
| --- | --- |
| add | 5.1e-11 |
| mul | 1.6e-10 |
| pow | 5.9e-10 |
| relu | 1.6e-11 |
| matmul | 2.0e-10 |
| softmax | 2.5e-9 |

Anything above roughly 1e-7 would indicate a bug. `cross_entropy` and the broadcasting in the bias add are not in this table, but they are covered end to end by the PyTorch comparison below, since the loss curves could not match to this precision if their gradients were wrong.

### Comparison Against PyTorch
The PyTorch model is built with the same layers and my starting weights are copied into it, then both are trained on the same data with the same optimizer settings (full batch, float64).

| Check | Result |
| --- | --- |
| Logits on the same input | match, max difference 2.2e-16 |
| SGD loss curve over 200 steps | max difference 4.4e-16 |
| Adam loss curve over 200 steps | max difference 2.2e-16 |

![Mine vs PyTorch](results/benchmark_curves.png)

Time per training step with Adam (median of three runs, measured on a 2-thread CPU, so the exact numbers will differ on other hardware):

| Network | Batch | Mine (ms) | PyTorch (ms) | Mine / PyTorch |
| --- | --- | --- | --- | --- |
| 2-16-16-3 | 32 | 0.21 | 0.64 | 0.35 |
| 2-16-16-3 | 300 | 0.41 | 0.56 | 0.64 |
| 2-512-512-3 | 300 | 52.2 | 7.7 | 6.7 |

On the small network this engine is faster, and on the wide one PyTorch is about 7 times faster. A ratio below 1 means mine is faster.

## Design Decisions
### Operations as Free Functions
Operations such as `add`, `mul`, and `matmul` are free functions in `ops.py`, and the `Tensor` class only has thin wrappers (`__add__`, `__mul__`, `__matmul__`) that call them. These operations hold no state, so grouping them under a class would add a layer of indirection without giving anything back, as there is no instance data, inheritance, or polymorphism to justify it. This mirrors how PyTorch splits it, where `torch.add` is a function and `Tensor.__add__` delegates to it. The wrappers import from `ops.py` inside each method, as `ops.py` itself imports `Tensor`, and importing in the other direction at the top of the file would be circular.

### Backward Closures and Gradient Accumulation
Each operation attaches a `_backward` closure to its output tensor, which captures the inputs and the output, and only knows its own local derivative multiplied by the upstream gradient (`out.grad`). `backward()` then stitches them together by walking the graph in reverse topological order, so a node's gradient is fully collected from everywhere it is used before it is passed on to its own inputs. Every closure accumulates with `+=` rather than overwriting, since a tensor can feed several places in the graph, and each place contributes its own gradient.

### Fused Softmax and Cross-Entropy
The gradient of a standalone softmax is a full Jacobian for every row, because each output depends on every input in its row. Fused with cross-entropy, it collapses into `(probabilities - one_hot) / batch`. The fused operation also computes log-softmax directly (`x - max - log(sum(exp(x - max)))`) instead of taking the log of a softmax, so there is never a `log(0)`. The standalone `softmax` operation was still kept, with its per-row Jacobian, as it is part of the engine and a good test of non-elementwise gradients, but training uses the fused version. For this reason `MLP` returns raw logits, as a softmax at the end of the model would apply softmax twice.

### Broadcasting in Backward
When a bias of shape `(out,)` is added to a result of shape `(batch, out)`, NumPy broadcasts it in the forward pass, which means its gradient has to be summed back down to its own shape in the backward pass. `_unbroadcast` handles this by summing away any axes that NumPy prepended and any axes that were stretched from size 1, and it is used in both `add` and `mul`.

### Initialization
Weights in `Linear` use He (Kaiming) initialization with a standard deviation of `sqrt(2 / in_features)`, and biases start at zero. All-zero weights would make every neuron in a layer learn the same thing, and a fixed standard deviation of 1 would make the outputs grow with the width of the layer. The factor of 2 compensates for ReLU zeroing out about half of the values.

### Optimizers as Classes
Unlike the operations, the optimizers are classes, as they hold state: the list of parameters for both, and for Adam the running averages of the gradient (`m`) and squared gradient (`v`) for every parameter plus a shared step counter `t`. `Adam.step()` increments `t` before the loop so the bias correction `1 - beta ** t` is never zero, and `m_hat` and `v_hat` are kept as separate local variables so the raw running averages carry over to the next step. `zero_grad()` is needed because every backward pass accumulates, and it resets each gradient with `np.zeros_like` so the shape stays the same as the parameter.

### float64 and Mean Loss
Everything is float64, which keeps the gradient checks accurate, and the PyTorch model in the benchmark is converted to float64 to match. The loss is a mean rather than a sum, so the size of the gradient does not depend on the batch size.

## How to Run
To install libraries:
```
pip install -r requirements.txt
```

The project lives in the `autograd-engine` folder, and files are run as modules from inside it, as running them directly breaks the imports:
```
cd autograd-engine
```

How to run the gradient checks:
```
python -m tests.test_gradcheck
```

How to run the training example (SGD vs Adam on the spiral dataset):
```
python -m examples.train_mlp
```

How to run the PyTorch benchmark:
```
python -m benchmarks.vs_pytorch
```

A minimal example of the engine itself:
```python
from engine.tensor import Tensor

a = Tensor(2.0)
b = Tensor(3.0)
c = a * b + a   # 8.0
c.backward()

print(a.grad)   # 4.0, which is b + 1
print(b.grad)   # 2.0, which is a
```

## What I Learned
Writing the backward pass for every operation myself showed how many small details have to be right, and how few of them throw an error when they are wrong. Most of the bugs below produced no crash at all, just a model that quietly learned worse, which is the reason the gradient checker exists.

### Accumulating vs. Overwriting
In `mul`, I first wrote the backward pass with `*=` instead of accumulating, and used `1.0` as the local derivative, which is only correct for addition. For multiplication, the derivative with respect to each input is the value of the other input, multiplied by the upstream gradient. Gradients always have to be added with `+=`, as a tensor used in more than one place would otherwise keep only the last contribution.

### Storing the Function vs. Calling It
In both `matmul` and `mean`, I wrote `out._backward = _backward()`, which calls the function immediately and stores `None`, rather than storing the function to be called later by `backward()`. It was the same mistake twice.

### Scalar Code Breaking on Arrays
My first `relu` used `a.data if a.data > 0 else 0`, which works on one number but fails on an array with "the truth value of an array is ambiguous", since `a.data > 0` gives an array of booleans instead of one. It had to become `np.maximum(0, a.data)` for the forward pass and an array mask for the backward pass. The same thing happened with `grad = 0.0` in both `Tensor` and `zero_grad`, where a scalar silently replaced a gradient array and changed its shape between steps, which `np.zeros_like` fixed.

### Seeding the Backward Pass
My first `backward()` walked the graph correctly but never set the starting gradient, so every gradient came out zero with no error. The chain rule needs to start from d(loss)/d(loss) = 1, and without it, every `_backward` multiplies by zero.

### Gradient Checking Pitfalls
Summing a softmax output always gives 1, so the gradient through a plain sum is zero and a test would pass while testing nothing. The softmax check multiplies the output by fixed random weights before summing instead. A separate issue was that my first `numerical_grad` looped over `range(x.size)` and indexed with `x[i]`, which on a 2D array indexes a row instead of an element and raised an `IndexError`, fixed by using `np.nditer` with `multi_index`.

### Softmax Applied Twice, Again
My first training loop applied a softmax before `cross_entropy`, and my first PyTorch model ended with `nn.Softmax()`. Both would apply softmax twice, since `cross_entropy` and `nn.CrossEntropyLoss` apply it internally. It is the same mistake I made in Mini GPT-2, and the fix was the same, return raw logits from the model.

### Matching PyTorch
Building the PyTorch equivalent had three places where the numbers could be silently wrong. My `W` is `(in, out)` because I compute `x @ W`, while PyTorch stores `weight` as `(out, in)`, so it has to be transposed when copied, and the square middle layer would not even raise a shape error if I forgot. My engine is float64 and PyTorch defaults to float32, so the PyTorch model has to be converted with `.double()` or rounding alone makes the curves drift apart. And the first layer of my first attempt was `Linear(1, 16)` instead of `Linear(2, 16)`, since the data has two features per point.

### Imports and Project Structure
Once the code was split into folders, `from tensor import Tensor` stopped working and had to become `from engine.tensor import Tensor`, with files run using `python -m` from the project folder. Running `python tests.test_gradcheck` without the `-m` makes Python treat it as a file name, from the folder that contains the packages.

At the end, I had an engine whose gradients agree with numerical gradients on every operation and whose training curves match PyTorch to floating point precision.

## References
1. Karpathy, A. *micrograd*. https://github.com/karpathy/micrograd
2. Stanford CS231n. *Gradient Checking*. https://cs231n.github.io/neural-networks-3/#gradcheck
3. Kingma, D. and Ba, J. *Adam: A Method for Stochastic Optimization*. https://arxiv.org/abs/1412.6980
4. PyTorch. *Autograd Mechanics*. https://pytorch.org/docs/stable/notes/autograd.html
5. APXML. *Kaiming (He) Initialization – How to Build a Large Language Model, Ch. 12*. https://apxml.com/courses/how-to-build-a-large-language-model/chapter-12-initialization-techniques-deep-networks/kaiming-he-initialization