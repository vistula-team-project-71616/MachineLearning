import copy
import yaml
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from main import BaselineCNN, evaluate


ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(ROOT / "config.yaml", "r") as file:
        return yaml.safe_load(file)


def load_data(config):
    transform = transforms.ToTensor()

    dataset = datasets.FashionMNIST(
        root=ROOT / config["data"]["path"],
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root=ROOT / config["data"]["path"],
        train=False,
        download=True,
        transform=transform
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [55000, 5000],
        generator=torch.Generator().manual_seed(config["seed"])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"]
    )

    return train_loader, val_loader, test_loader


def get_prunable_layers(model):
    return [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]


def apply_global_pruning(model, amount):
    prune.global_unstructured(
        get_prunable_layers(model),
        pruning_method=prune.L1Unstructured,
        amount=amount
    )


def apply_layerwise_pruning(model, amount):
    for module, name in get_prunable_layers(model):
        prune.l1_unstructured(
            module,
            name=name,
            amount=amount
        )


def apply_structured_pruning(model, amount):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(
                module,
                name="weight",
                amount=amount,
                n=2,
                dim=0
            )


def calculate_sparsity(model):
    total_weights = 0
    zero_weights = 0

    for module, _ in get_prunable_layers(model):
        weights = (
            module.weight_mask
            if hasattr(module, "weight_mask")
            else module.weight
        )

        total_weights += weights.numel()
        zero_weights += (weights == 0).sum().item()

    return 100 * zero_weights / total_weights


def calculate_filter_pruning(model):
    total_filters = 0
    removed_filters = 0

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            weights = (
                module.weight_mask
                if hasattr(module, "weight_mask")
                else module.weight
            )

            filter_norms = weights.reshape(
                weights.shape[0], -1
            ).norm(p=2, dim=1)

            total_filters += filter_norms.numel()
            removed_filters += (filter_norms == 0).sum().item()

    if total_filters == 0:
        return 0.0

    return 100 * removed_filters / total_filters


def fine_tune(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0005
    )

    history = []

    for epoch in range(3):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            for module, _ in get_prunable_layers(model):
                if hasattr(module, "weight_mask"):
                    module.weight_orig.data.mul_(module.weight_mask)

            total_loss += loss.item()

        accuracy = evaluate(model, val_loader, device)

        history.append({
            "epoch": epoch + 1,
            "loss": total_loss / len(train_loader),
            "val_accuracy": accuracy
        })

        print(
            f"    Fine-tune epoch {epoch + 1}: "
            f"Loss={history[-1]['loss']:.4f}, "
            f"Accuracy={accuracy:.2f}%"
        )

    return history


def run_experiment(
    baseline,
    method,
    amount,
    train_loader,
    val_loader,
    device
):
    model = copy.deepcopy(baseline)

    if method == "Global":
        apply_global_pruning(model, amount)

    elif method == "Layer-wise":
        apply_layerwise_pruning(model, amount)

    elif method == "Structured":
        apply_structured_pruning(model, amount)

    print(f"\n{method} pruning: {amount * 100:.0f}%")

    initial_sparsity = calculate_sparsity(model)
    initial_filter_pruning = calculate_filter_pruning(model)

    print(f"    Weight sparsity: {initial_sparsity:.2f}%")

    if method == "Structured":
        print(f"    Removed filters: {initial_filter_pruning:.2f}%")

    fine_tune(
        model,
        train_loader,
        val_loader,
        device
    )

    accuracy = evaluate(model, val_loader, device)
    final_sparsity = calculate_sparsity(model)
    final_filter_pruning = calculate_filter_pruning(model)

    print(f"    Final accuracy: {accuracy:.2f}%")
    print(f"    Final weight sparsity: {final_sparsity:.2f}%")

    if method == "Structured":
        print(f"    Final removed filters: {final_filter_pruning:.2f}%")

    return {
        "method": method,
        "target_sparsity": amount * 100,
        "weight_sparsity": final_sparsity,
        "removed_filters": final_filter_pruning,
        "accuracy": accuracy
    }


def main():
    config = load_config()

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cpu")
    print("Device:", device)

    train_loader, val_loader, test_loader = load_data(config)

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

    baseline_accuracy = evaluate(
        baseline,
        val_loader,
        device
    )

    print(f"Baseline validation accuracy: {baseline_accuracy:.2f}%")

    results = []

    sparsity_levels = [0, 0.25, 0.50, 0.75, 0.90, 0.95]

    for method in ["Global", "Layer-wise"]:
        for amount in sparsity_levels:
            if amount == 0:
                results.append({
                    "method": method,
                    "target_sparsity": 0,
                    "weight_sparsity": 0,
                    "removed_filters": 0,
                    "accuracy": baseline_accuracy
                })
                continue

            result = run_experiment(
                baseline,
                method,
                amount,
                train_loader,
                val_loader,
                device
            )

            results.append(result)

    structured_levels = [0.10, 0.25, 0.50]

    for amount in structured_levels:
        result = run_experiment(
            baseline,
            "Structured",
            amount,
            train_loader,
            val_loader,
            device
        )

        results.append(result)

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        results_dir / "pruning_results.csv",
        index=False
    )

    plt.figure(figsize=(9, 6))

    for method in ["Global", "Layer-wise"]:
        subset = results_df[
            results_df["method"] == method
        ]

        plt.plot(
            subset["weight_sparsity"],
            subset["accuracy"],
            marker="o",
            label=method
        )

    plt.xlabel("Weight sparsity (%)")
    plt.ylabel("Validation accuracy (%)")
    plt.title("Accuracy vs Weight Sparsity")
    plt.legend()
    plt.grid(True)

    plt.savefig(results_dir / "pruning_sparsity.png")
    plt.close()

    subset = results_df[
        results_df["method"] == "Structured"
    ]

    plt.figure(figsize=(9, 6))

    plt.plot(
        subset["removed_filters"],
        subset["accuracy"],
        marker="o",
        label="Structured"
    )

    plt.xlabel("Removed filters (%)")
    plt.ylabel("Validation accuracy (%)")
    plt.title("Accuracy vs Structured Filter Pruning")
    plt.legend()
    plt.grid(True)

    plt.savefig(results_dir / "structured_pruning.png")
    plt.close()

    print("\nPRUNING RESULTS")
    print(results_df.to_string(index=False))

    print("\nResults saved in:", results_dir)


if __name__ == "__main__":
    main()