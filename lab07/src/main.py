
import os
import time
import random
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent.parent


class BaselineCNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        channels = config["model"]["channels"]
        hidden = config["model"]["hidden_size"]

        self.features = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(channels[0], channels[1], 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(channels[1], channels[2], 3, padding=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[2] * 7 * 7, hidden),
            nn.ReLU(),
            nn.Linear(hidden, config["model"]["num_classes"])
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
    example = torch.randn(1, 1, 28, 28).to(device)

    with torch.no_grad():
        for _ in range(20):
            model(example)

        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []

        for _ in range(5):
            start = time.perf_counter()

            for _ in range(100):
                model(example)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            times.append(elapsed / 100 * 1000)

    return float(np.median(times))


def main():
    with open(ROOT / "config.yaml", "r") as file:
        config = yaml.safe_load(file)

    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
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

    model = BaselineCNN(config).to(device)

    parameters = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {parameters:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    checkpoint_dir = ROOT / config["checkpoint"]["directory"]
    checkpoint_dir.mkdir(exist_ok=True)

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

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

            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch + 1,
                "val_accuracy": val_accuracy
            }, checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_accuracy = evaluate(model, test_loader, device)
    latency = measure_latency(model, device)

    model_size = os.path.getsize(checkpoint_path) / (1024 * 1024)

    print("\nBASELINE RESULTS")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print(f"Parameters: {parameters:,}")
    print(f"Checkpoint size: {model_size:.2f} MB")
    print(f"CPU/GPU latency: {latency:.3f} ms per image")

    pd.DataFrame(history).to_csv(
        results_dir / "baseline_history.csv",
        index=False
    )

    pd.DataFrame([{
        "parameters": parameters,
        "test_accuracy": test_accuracy,
        "model_size_mb": model_size,
        "latency_ms": latency
    }]).to_csv(
        results_dir / "baseline_metrics.csv",
        index=False
    )

    print("\nBaseline model saved:", checkpoint_path)


if __name__ == "__main__":
    main()
