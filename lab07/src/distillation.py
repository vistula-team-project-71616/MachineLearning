import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random

from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from main import BaselineCNN, evaluate


ROOT = Path(__file__).resolve().parent.parent


class StudentCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def load_config():
    with open(ROOT / "config.yaml", "r") as file:
        return yaml.safe_load(file)


def load_data(config):
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

    train_dataset, val_dataset = random_split(
        train_dataset,
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


def train_plain_student(model, train_loader, val_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
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

        accuracy = evaluate(model, val_loader, device)

        print(
            f"Plain Student Epoch {epoch + 1}: "
            f"Loss={total_loss / len(train_loader):.4f}, "
            f"Val Accuracy={accuracy:.2f}%"
        )


def train_distilled_student(
    student,
    teacher,
    train_loader,
    val_loader,
    device,
    temperature=4,
    alpha=0.5
):
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    teacher.eval()

    for epoch in range(10):
        student.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_outputs = teacher(images)

            student_outputs = student(images)

            hard_loss = criterion(student_outputs, labels)

            soft_loss = F.kl_div(
                F.log_softmax(student_outputs / temperature, dim=1),
                F.softmax(teacher_outputs / temperature, dim=1),
                reduction="batchmean"
            ) * (temperature ** 2)

            loss = alpha * hard_loss + (1 - alpha) * soft_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        accuracy = evaluate(student, val_loader, device)

        print(
            f"Distilled Student Epoch {epoch + 1}: "
            f"Loss={total_loss / len(train_loader):.4f}, "
            f"Val Accuracy={accuracy:.2f}%"
        )


def measure_latency(model, device):
    model.eval()
    example = torch.randn(1, 1, 28, 28).to(device)

    with torch.no_grad():
        for _ in range(20):
            model(example)

        times = []

        for _ in range(5):
            start = time.perf_counter()

            for _ in range(100):
                model(example)

            times.append(
                (time.perf_counter() - start) / 100 * 1000
            )

    return float(np.median(times))


def save_model(model, path):
    torch.save(model.state_dict(), path)
    return path.stat().st_size / (1024 * 1024)


def main():
    config = load_config()

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cpu")
    print("Device:", device)

    train_loader, val_loader, test_loader = load_data(config)

    teacher_checkpoint = torch.load(
        ROOT / config["checkpoint"]["directory"] /
        config["checkpoint"]["filename"],
        map_location=device
    )

    teacher = BaselineCNN(
        input_channels=config["model"]["input_channels"],
        channels=config["model"]["channels"],
        hidden_size=config["model"]["hidden_size"],
        num_classes=config["model"]["num_classes"]
    ).to(device)

    teacher.load_state_dict(teacher_checkpoint)
    teacher.eval()

    teacher_accuracy = evaluate(teacher, test_loader, device)

    plain_student = StudentCNN().to(device)

    plain_parameters = sum(
        p.numel() for p in plain_student.parameters()
    )

    print("\nPLAIN STUDENT")
    print("Parameters:", f"{plain_parameters:,}")

    train_plain_student(
        plain_student,
        train_loader,
        val_loader,
        device
    )

    plain_accuracy = evaluate(
        plain_student,
        test_loader,
        device
    )

    distilled_student = StudentCNN().to(device)

    distilled_parameters = sum(
        p.numel() for p in distilled_student.parameters()
    )

    print("\nDISTILLED STUDENT")
    print("Parameters:", f"{distilled_parameters:,}")

    train_distilled_student(
        distilled_student,
        teacher,
        train_loader,
        val_loader,
        device,
        temperature=4,
        alpha=0.5
    )

    distilled_accuracy = evaluate(
        distilled_student,
        test_loader,
        device
    )

    checkpoints_dir = ROOT / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    teacher_size = save_model(
        teacher,
        checkpoints_dir / "teacher.pth"
    )

    plain_size = save_model(
        plain_student,
        checkpoints_dir / "plain_student.pth"
    )

    distilled_size = save_model(
        distilled_student,
        checkpoints_dir / "distilled_student.pth"
    )

    teacher_latency = measure_latency(teacher, device)
    plain_latency = measure_latency(plain_student, device)
    distilled_latency = measure_latency(distilled_student, device)

    results = [
        {
            "model": "Teacher",
            "parameters": sum(p.numel() for p in teacher.parameters()),
            "test_accuracy": teacher_accuracy,
            "size_mb": teacher_size,
            "latency_ms": teacher_latency
        },
        {
            "model": "Plain Student",
            "parameters": plain_parameters,
            "test_accuracy": plain_accuracy,
            "size_mb": plain_size,
            "latency_ms": plain_latency
        },
        {
            "model": "Distilled Student",
            "parameters": distilled_parameters,
            "test_accuracy": distilled_accuracy,
            "size_mb": distilled_size,
            "latency_ms": distilled_latency
        }
    ]

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)

    df.to_csv(
        results_dir / "distillation_results.csv",
        index=False
    )

    print("\nDISTILLATION RESULTS")
    print(df.to_string(index=False))

    print("\nResults saved in:", results_dir)


if __name__ == "__main__":
    main()
