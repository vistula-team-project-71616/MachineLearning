
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


def train_model(train_loader, accumulation_steps, device):
    model = create_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    history = []
    start_time = time.perf_counter()

    for epoch in range(5):
        model.train()
        total_loss = 0.0

        optimizer.zero_grad()

        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

        # Apply remaining gradients if the final batch is incomplete
        if (step + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        average_loss = total_loss / len(train_loader)
        history.append(average_loss)

        print(
            f"Epoch {epoch + 1}/5, "
            f"Loss: {average_loss:.4f}"
        )

    training_time = time.perf_counter() - start_time

    return model, history, training_time


def evaluate(model, test_loader, device):
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

    return 100 * correct / total


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

    train_dataset, _ = random_split(
        train_data,
        [55000, 5000],
        generator=torch.Generator().manual_seed(42)
    )

    test_loader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    # Experiment 1: real batch size 256
    print("\nTraining with batch size 256")
    loader_256 = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0
    )

    set_seed(42)
    model_256, loss_256, time_256 = train_model(
        loader_256, 1, device
    )
    accuracy_256 = evaluate(model_256, test_loader, device)

    # Experiment 2: batch size 64, accumulation 4
    print("\nTraining with batch size 64 and accumulation 4")
    loader_64 = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )

    set_seed(42)
    model_acc, loss_acc, time_acc = train_model(
        loader_64, 4, device
    )
    accuracy_acc = evaluate(model_acc, test_loader, device)

    results = [
        {
            "method": "Batch size 256",
            "effective_batch_size": 256,
            "test_accuracy": round(accuracy_256, 2),
            "training_time_sec": round(time_256, 2)
        },
        {
            "method": "Batch 64 + accumulation 4",
            "effective_batch_size": 256,
            "test_accuracy": round(accuracy_acc, 2),
            "training_time_sec": round(time_acc, 2)
        }
    ]

    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(
        "results/gradient_accumulation.csv",
        index=False
    )

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, 6), loss_256, marker="o",
             label="Batch size 256")
    plt.plot(range(1, 6), loss_acc, marker="o",
             label="Batch 64 + accumulation 4")

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Gradient Accumulation Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/gradient_accumulation.png")
    plt.close()

    print("\nFinal comparison:")
    print(df.to_string(index=False))
    print("\nSaved results in results folder.")


if __name__ == "__main__":
    main()


