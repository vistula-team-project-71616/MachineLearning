
import os
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


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


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

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    model = create_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    checkpoint_path = "checkpoints/resume_checkpoint.pth"
    history = []

    # First part: train for 3 epochs
    print("\nStarting training: epochs 1-3")

    for epoch in range(3):
        loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        accuracy = evaluate(model, test_loader, device)

        history.append({
            "epoch": epoch + 1,
            "loss": loss,
            "accuracy": accuracy
        })

        print(
            f"Epoch {epoch + 1}/5 | "
            f"Loss: {loss:.4f} | "
            f"Test Accuracy: {accuracy:.2f}%"
        )

    # Save model and optimizer state
    torch.save({
        "epoch": 3,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history
    }, checkpoint_path)

    print("\nTraining interrupted after epoch 3.")
    print("Checkpoint saved:", checkpoint_path)

    # Simulate restarting: create new model and optimizer
    print("\nResuming from checkpoint...")

    resumed_model = create_model().to(device)
    resumed_optimizer = optim.Adam(
        resumed_model.parameters(), lr=0.001
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    resumed_model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    resumed_optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"]
    history = checkpoint["history"]

    print("Resumed from epoch:", start_epoch)

    # Continue training: epochs 4-5
    for epoch in range(start_epoch, 5):
        loss = train_epoch(
            resumed_model,
            train_loader,
            resumed_optimizer,
            criterion,
            device
        )

        accuracy = evaluate(
            resumed_model, test_loader, device
        )

        history.append({
            "epoch": epoch + 1,
            "loss": loss,
            "accuracy": accuracy
        })

        print(
            f"Epoch {epoch + 1}/5 | "
            f"Loss: {loss:.4f} | "
            f"Test Accuracy: {accuracy:.2f}%"
        )

    # Save final checkpoint
    torch.save({
        "epoch": 5,
        "model_state_dict": resumed_model.state_dict(),
        "optimizer_state_dict": resumed_optimizer.state_dict(),
        "history": history
    }, "checkpoints/resumed_final.pth")

    df = pd.DataFrame(history)
    df.to_csv("results/resume_history.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss After Checkpoint Resume")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/resume_training.png")
    plt.close()

    print("\nFinal accuracy:", round(history[-1]["accuracy"], 2), "%")
    print("Training completed successfully.")
    print("Results saved in results folder.")


if __name__ == "__main__":
    main()


