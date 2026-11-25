import os
import time
import random
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]


class BaselineCNN(nn.Module):
    def __init__(self, input_channels, channels, hidden_size, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[2] * 7 * 7, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


def measure_latency(model, device):
    model.eval()
    sample = torch.randn(1, 1, 28, 28).to(device)

    with torch.no_grad():
        for _ in range(20):
            model(sample)

        measurements = []

        for _ in range(5):
            start = time.perf_counter()

            for _ in range(100):
                model(sample)

            end = time.perf_counter()
            measurements.append((end - start) * 1000 / 100)

    return float(np.median(measurements))


def main():
    with open(ROOT / "config.yaml", "r") as file:
        config = yaml.safe_load(file)

    set_seed(config["seed"])

    device = torch.device("cpu")
    print("Device:", device)

    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(
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

    train_size = 55000
    val_size = 5000

    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config["seed"])
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"]
    )

    val_loader = DataLoader(
        val_subset,
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

    model = BaselineCNN(
        input_channels=config["model"]["input_channels"],
        channels=config["model"]["channels"],
        hidden_size=config["model"]["hidden_size"],
        num_classes=config["model"]["num_classes"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    checkpoint_dir = ROOT / config["checkpoint"]["directory"]
    results_dir = ROOT / "results"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / config["checkpoint"]["filename"]

    best_accuracy = 0
    history = []

    for epoch in range(config["training"]["epochs"]):
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

            total_loss += loss.item()

        val_accuracy = evaluate(model, val_loader, device)
        average_loss = total_loss / len(train_loader)

        history.append({
            "epoch": epoch + 1,
            "loss": average_loss,
            "val_accuracy": val_accuracy
        })

        print(
            f"Epoch {epoch + 1}: "
            f"Loss={average_loss:.4f}, "
            f"Val Accuracy={val_accuracy:.2f}%"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_accuracy = evaluate(model, test_loader, device)
    latency = measure_latency(model, device)

    parameters = sum(p.numel() for p in model.parameters())
    checkpoint_size = os.path.getsize(checkpoint_path) / (1024 * 1024)

    print("\nBASELINE RESULTS")
    print("Parameters:", parameters)
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print(f"Checkpoint size: {checkpoint_size:.2f} MB")
    print(f"Latency: {latency:.3f} ms/image")

    pd.DataFrame(history).to_csv(
        results_dir / "baseline_history.csv",
        index=False
    )

    pd.DataFrame([{
        "parameters": parameters,
        "best_val_accuracy": best_accuracy,
        "test_accuracy": test_accuracy,
        "checkpoint_size_mb": checkpoint_size,
        "latency_ms": latency,
        "device": str(device)
    }]).to_csv(
        results_dir / "baseline_metrics.csv",
        index=False
    )


if __name__ == "__main__":
    main()