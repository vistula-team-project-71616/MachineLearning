
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_model():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 10)
    )


def run_experiment(optimizer_name, scheduler_name,
                   train_loader, val_loader, test_loader, device):

    set_seed(42)

    model = create_model().to(device)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9
        )

    if scheduler_name == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.5
        )
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )

    criterion = nn.CrossEntropyLoss()
    history = []
    start_time = time.perf_counter()

    for epoch in range(10):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "optimizer": optimizer_name,
            "scheduler": scheduler_name,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": current_lr
        })

        if scheduler_name == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        print(
            f"{optimizer_name} + {scheduler_name} | "
            f"Epoch {epoch + 1}/10 | "
            f"Val Accuracy: {val_accuracy:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

    training_time = time.perf_counter() - start_time

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    test_accuracy = 100 * correct / total

    result = {
        "optimizer": optimizer_name,
        "scheduler": scheduler_name,
        "test_accuracy": round(test_accuracy, 2),
        "training_time_sec": round(training_time, 2)
    }

    return result, history


def main():
    set_seed(42)

    device = torch.device("cpu")
    print("Device:", device)

    transform = transforms.ToTensor()

    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    train_dataset, val_dataset = random_split(
        train_data,
        [55000, 5000],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=128,
        shuffle=True, num_workers=0
    )

    val_loader = DataLoader(
        val_dataset, batch_size=128,
        shuffle=False, num_workers=0
    )

    test_loader = DataLoader(
        test_data, batch_size=128,
        shuffle=False, num_workers=0
    )

    results = []
    all_history = []

    optimizers = ["Adam", "SGD"]
    schedulers = [
        "StepLR",
        "CosineAnnealingLR",
        "ReduceLROnPlateau"
    ]

    for optimizer_name in optimizers:
        for scheduler_name in schedulers:
            print(f"\nStarting: {optimizer_name} + {scheduler_name}")

            result, history = run_experiment(
                optimizer_name,
                scheduler_name,
                train_loader,
                val_loader,
                test_loader,
                device
            )

            results.append(result)
            all_history.extend(history)

            print("Result:", result)

    os.makedirs("results", exist_ok=True)

    results_df = pd.DataFrame(results)
    history_df = pd.DataFrame(all_history)

    results_df.to_csv(
        "results/scheduler_comparison.csv", index=False
    )
    history_df.to_csv(
        "results/scheduler_history.csv", index=False
    )

    print("\nFinal comparison:")
    print(results_df.to_string(index=False))

    plt.figure(figsize=(10, 6))

    for optimizer_name in optimizers:
        for scheduler_name in schedulers:
            subset = history_df[
                (history_df["optimizer"] == optimizer_name) &
                (history_df["scheduler"] == scheduler_name)
            ]

            label = f"{optimizer_name} + {scheduler_name}"

            plt.plot(
                subset["epoch"],
                subset["val_accuracy"],
                marker="o",
                label=label
            )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Optimizer and Learning Rate Scheduler Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/scheduler_comparison.png")
    plt.close()

    print("\nSaved results in the results folder.")


if __name__ == "__main__":
    main()


