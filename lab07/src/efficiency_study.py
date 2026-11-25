import copy
import time
import torch
import torch.nn.utils.prune as prune
import pandas as pd
import numpy as np

from pathlib import Path

from main import BaselineCNN
from pruning import (
    load_config,
    get_prunable_layers,
    apply_global_pruning,
    apply_layerwise_pruning
)


ROOT = Path(__file__).resolve().parent.parent


def measure_latency(model, device):
    model.eval()
    example = torch.randn(1, 1, 28, 28).to(device)

    with torch.no_grad():
        for _ in range(20):
            model(example)

        measurements = []

        for _ in range(5):
            start = time.perf_counter()

            for _ in range(100):
                model(example)

            measurements.append(
                (time.perf_counter() - start) / 100 * 1000
            )

    return float(np.median(measurements))


def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def get_sparsity(model):
    total = 0
    zeros = 0

    for module, _ in get_prunable_layers(model):
        weights = (
            module.weight_mask
            if hasattr(module, "weight_mask")
            else module.weight
        )

        total += weights.numel()
        zeros += (weights == 0).sum().item()

    return 100 * zeros / total


def save_dense_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    return path.stat().st_size / (1024 * 1024)


def save_sparse_checkpoint(model, path):
    sparse_state = {}

    for name, parameter in model.named_parameters():
        tensor = parameter.detach().cpu()

        if tensor.ndim >= 2:
            sparse_state[name] = tensor.to_sparse()
        else:
            sparse_state[name] = tensor

    torch.save(sparse_state, path)
    return path.stat().st_size / (1024 * 1024)


def prepare_model(baseline, method, amount):
    model = copy.deepcopy(baseline)

    if method == "Global":
        apply_global_pruning(model, amount)

    elif method == "Layer-wise":
        apply_layerwise_pruning(model, amount)

    for module, name in get_prunable_layers(model):
        if hasattr(module, "weight_orig"):
            prune.remove(module, name)

    return model


def main():
    config = load_config()

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cpu")
    print("Device:", device)

    checkpoint_path = (
        ROOT / config["checkpoint"]["directory"] /
        config["checkpoint"]["filename"]
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    baseline = BaselineCNN(
        input_channels=config["model"]["input_channels"],
        channels=config["model"]["channels"],
        hidden_size=config["model"]["hidden_size"],
        num_classes=config["model"]["num_classes"]
    ).to(device)

    baseline.load_state_dict(checkpoint)
    baseline.eval()

    results_dir = ROOT / "results"
    checkpoints_dir = ROOT / "checkpoints"

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("Baseline", 0),
        ("Global", 0.50),
        ("Global", 0.90),
        ("Global", 0.95),
        ("Layer-wise", 0.50),
        ("Layer-wise", 0.90),
        ("Layer-wise", 0.95)
    ]

    results = []

    for method, amount in experiments:
        print(f"\nTesting {method}, sparsity {amount * 100:.0f}%")

        if method == "Baseline":
            model = copy.deepcopy(baseline)
        else:
            model = prepare_model(baseline, method, amount)

        model.eval()

        actual_sparsity = get_sparsity(model)
        latency = measure_latency(model, device)

        name = f"{method.lower()}_{int(amount * 100)}"

        dense_path = checkpoints_dir / f"{name}_dense.pth"
        sparse_path = checkpoints_dir / f"{name}_sparse.pth"

        dense_size = save_dense_checkpoint(model, dense_path)
        sparse_size = save_sparse_checkpoint(model, sparse_path)

        parameters = get_parameter_count(model)

        result = {
            "model": method,
            "target_sparsity": amount * 100,
            "actual_sparsity": actual_sparsity,
            "parameters": parameters,
            "dense_size_mb": dense_size,
            "sparse_size_mb": sparse_size,
            "latency_ms": latency
        }

        results.append(result)

        print(f"Actual sparsity: {actual_sparsity:.2f}%")
        print(f"Parameters: {parameters:,}")
        print(f"Dense size: {dense_size:.2f} MB")
        print(f"Sparse size: {sparse_size:.2f} MB")
        print(f"CPU latency: {latency:.3f} ms")

    df = pd.DataFrame(results)

    df.to_csv(
        results_dir / "efficiency_results.csv",
        index=False
    )

    print("\nEFFICIENCY RESULTS")
    print(df.to_string(index=False))

    print("\nResults saved in:", results_dir)


if __name__ == "__main__":
    main()
