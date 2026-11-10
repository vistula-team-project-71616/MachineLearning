
import time
import statistics
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_result(filename, content):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)


class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power):
        out = Value(self.data ** power, (self,), f'**{power}')

        def _backward():
            self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else -float(other))

    def __rsub__(self, other):
        return self.__neg__() + other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        out = Value(1.0 / (1.0 + math.exp(-self.data)), (self,), 'sigmoid')

        def _backward():
            self.grad += out.data * (1.0 - out.data) * out.grad

        out._backward = _backward
        return out

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
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()


def check_gradients():
    a = Value(2.0)
    b = Value(3.0)
    c = Value(1.0)

    result = ((a * b + c) ** 2).relu()
    result.backward()

    our_grads = [a.grad, b.grad, c.grad]

    ta = torch.tensor(2.0, requires_grad=True)
    tb = torch.tensor(3.0, requires_grad=True)
    tc = torch.tensor(1.0, requires_grad=True)

    torch_result = ((ta * tb + tc) ** 2).relu().sum()
    torch_result.backward()

    torch_grads = [ta.grad.item(), tb.grad.item(), tc.grad.item()]

    def f(a, b, c):
        return max(0, (a * b + c) ** 2)

    h = 1e-5
    numerical_grads = [
        (f(2 + h, 3, 1) - f(2 - h, 3, 1)) / (2 * h),
        (f(2, 3 + h, 1) - f(2, 3 - h, 1)) / (2 * h),
        (f(2, 3, 1 + h) - f(2, 3, 1 - h)) / (2 * h)
    ]

    max_diff = max(abs(x - y) for x, y in zip(our_grads, torch_grads))
    numerical_diff = max(abs(x - y) for x, y in zip(our_grads, numerical_grads))

    content = (
        f"Our gradients: {our_grads}\n"
        f"PyTorch gradients: {torch_grads}\n"
        f"Numerical gradients: {numerical_grads}\n"
        f"Max difference with PyTorch: {max_diff}\n"
        f"Max difference with numerical: {numerical_diff}\n"
    )

    print("\nGRADIENT CHECK")
    print(content)

    save_result("gradient_check.txt", content)

    assert max_diff < 1e-6
    assert numerical_diff < 1e-6

    return content


def prepare_data():
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


class CustomNeuron:
    def __init__(self, inputs):
        self.weights = [
            Value(np.random.uniform(-0.5, 0.5))
            for _ in range(inputs)
        ]
        self.bias = Value(0.0)

    def __call__(self, x):
        result = self.bias
        for weight, value in zip(self.weights, x):
            result = result + weight * value
        return result.relu()

    def parameters(self):
        return self.weights + [self.bias]


class CustomModel:
    def __init__(self):
        self.layer1 = [CustomNeuron(2) for _ in range(8)]
        self.layer2 = CustomNeuron(8)

    def __call__(self, x):
        hidden = [neuron(x) for neuron in self.layer1]

        output = self.layer2.bias
        for weight, value in zip(self.layer2.weights, hidden):
            output = output + weight * value

        return output.sigmoid()

    def parameters(self):
        params = []
        for neuron in self.layer1:
            params.extend(neuron.parameters())
        params.extend(self.layer2.parameters())
        return params


def custom_training(X_train, y_train, X_test, y_test):
    np.random.seed(42)

    model = CustomModel()
    params = model.parameters()

    learning_rate = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    m = [0.0 for _ in params]
    v = [0.0 for _ in params]
    step = 0

    start = time.perf_counter()

    for epoch in range(100):
        for x, target in zip(X_train, y_train):
            prediction = model(x)

            loss = -(
                float(target) * prediction.log()
                + (1.0 - float(target)) * (1.0 - prediction).log()
            )

            for param in params:
                param.grad = 0.0

            loss.backward()
            step += 1

            for i, param in enumerate(params):
                m[i] = beta1 * m[i] + (1 - beta1) * param.grad
                v[i] = beta2 * v[i] + (1 - beta2) * param.grad ** 2

                m_hat = m[i] / (1 - beta1 ** step)
                v_hat = v[i] / (1 - beta2 ** step)

                param.data -= learning_rate * m_hat / (
                    math.sqrt(v_hat) + epsilon
                )

    elapsed = time.perf_counter() - start

    correct = 0
    for x, target in zip(X_test, y_test):
        prediction = model(x).data
        if (prediction >= 0.5) == bool(target):
            correct += 1

    return elapsed, correct / len(y_test)


class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def pytorch_training(X_train, y_train, X_test, y_test):
    torch.manual_seed(42)

    model = TorchModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    start = time.perf_counter()

    for epoch in range(100):
        optimizer.zero_grad()
        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)
        loss.backward()
        optimizer.step()

    elapsed = time.perf_counter() - start

    with torch.no_grad():
        predictions = model(X_test_t).view(-1)
        accuracy = ((predictions >= 0.5).numpy() == y_test).mean()

    return elapsed, accuracy


def training_comparison():
    X_train, X_test, y_train, y_test = prepare_data()

    custom_times = []
    torch_times = []
    custom_accuracy = []
    torch_accuracy = []

    for run in range(5):
        custom_time, custom_acc = custom_training(
            X_train, y_train, X_test, y_test
        )
        torch_time, torch_acc = pytorch_training(
            X_train, y_train, X_test, y_test
        )

        custom_times.append(custom_time)
        torch_times.append(torch_time)
        custom_accuracy.append(custom_acc)
        torch_accuracy.append(torch_acc)

    custom_median = statistics.median(custom_times)
    torch_median = statistics.median(torch_times)
    custom_acc_mean = statistics.mean(custom_accuracy)
    torch_acc_mean = statistics.mean(torch_accuracy)

    content = (
        f"Runs: 5\n"
        f"Epochs: 100\n"
        f"Architecture: 2 -> 8 -> 1\n"
        f"Custom autograd median time: {custom_median:.6f} seconds\n"
        f"PyTorch median time: {torch_median:.6f} seconds\n"
        f"Custom average accuracy: {custom_acc_mean:.4f}\n"
        f"PyTorch average accuracy: {torch_acc_mean:.4f}\n"
    )

    print("\nTRAINING COMPARISON")
    print(content)

    save_result("training_comparison.txt", content)

    return content


def execution_comparison():
    model = TorchModel()
    x = torch.randn(1000, 2)
    model.eval()

    for _ in range(10):
        model(x)

    start = time.perf_counter()
    for _ in range(100):
        model(x)
    eager_time = (time.perf_counter() - start) / 100

    traced_time = None

    try:
        traced_model = torch.jit.trace(model, x)

        for _ in range(10):
            traced_model(x)

        start = time.perf_counter()
        for _ in range(100):
            traced_model(x)
        traced_time = (time.perf_counter() - start) / 100

    except Exception as error:
        print("Tracing failed:", error)

    content = (
        f"Eager time: {eager_time:.9f} seconds\n"
        f"Traced time: "
        f"{traced_time:.9f} seconds\n" if traced_time is not None
        else f"Eager time: {eager_time:.9f} seconds\nTracing failed\n"
    )

    print("\nEXECUTION COMPARISON")
    print(content)

    save_result("execution_comparison.txt", content)

    return content


def tensor_comparison():
    x = torch.randn(1000, 1000)
    non_contiguous = x.t()

    for _ in range(10):
        result = x * 2
        result = non_contiguous * 2

    start = time.perf_counter()
    for _ in range(100):
        result = x * 2
    contiguous_time = (time.perf_counter() - start) / 100

    start = time.perf_counter()
    for _ in range(100):
        result = non_contiguous * 2
    non_contiguous_time = (time.perf_counter() - start) / 100

    content = (
        f"Shape: {x.shape}\n"
        f"Data type: {x.dtype}\n"
        f"Contiguous tensor: {x.is_contiguous()}\n"
        f"Transposed shape: {non_contiguous.shape}\n"
        f"Transposed tensor contiguous: {non_contiguous.is_contiguous()}\n"
        f"Contiguous operation time: {contiguous_time:.9f} seconds\n"
        f"Non-contiguous operation time: {non_contiguous_time:.9f} seconds\n"
    )

    print("\nTENSOR INFORMATION")
    print(content)

    save_result("tensor_comparison.txt", content)

    return content


def create_report(gradient, training, execution, tensor):
    report = (
        "# Lab 5: Frameworks and Automatic Differentiation\n\n"
        "## 1. Gradient verification\n\n"
        f"```text\n{gradient}```\n"
        "## 2. Training comparison\n\n"
        f"```text\n{training}```\n"
        "## 3. Eager and traced execution\n\n"
        f"```text\n{execution}```\n"
        "## 4. Tensor comparison\n\n"
        f"```text\n{tensor}```\n"
        "## 5. Conclusion\n\n"
        "The custom automatic differentiation engine was implemented "
        "and its gradients were compared with PyTorch and numerical "
        "estimates. The training experiment compared the custom model "
        "with a PyTorch implementation on the make_moons dataset. "
        "Eager and traced execution times were measured, and contiguous "
        "and non-contiguous tensor operations were compared.\n"
    )

    save_result("report.md", report)


if __name__ == "__main__":
    gradient_results = check_gradients()
    training_results = training_comparison()
    execution_results = execution_comparison()
    tensor_results = tensor_comparison()

    create_report(
        gradient_results,
        training_results,
        execution_results,
        tensor_results
    )

    print("\nAll results saved in the results folder.")
