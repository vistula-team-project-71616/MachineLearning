
import os
import time
import threading
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


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


def measure_memory(stop_event, memory_values):
    process = psutil.Process(os.getpid())

    while not stop_event.is_set():
        memory_values.append(
            process.memory_info().rss / (1024 ** 2)
        )
        time.sleep(0.01)


def run_experiment(batch_size, train_dataset, test_loader, device):
    model = create_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    stop_event = threading.Event()
    memory_values = []

    monitor = threading.Thread(
        target=measure_memory,
        args=(stop_event, memory_values)
    )
    monitor.start()

    epoch_times = []
    total_samples = 0

    for epoch in range(5):
        start_time = time.perf_counter()
        model.train()
        epoch_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_samples += images.size(0)

        elapsed = time.perf_counter() - start_time
        epoch_times.append(elapsed)
        total_samples += epoch_samples

        print(
            f"Batch {batch_size}, Epoch {epoch + 1}/5: "
            f"{elapsed:.2f}s"
        )

    stop_event.set()
    monitor.join()

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

    accuracy = 100 * correct / total
    throughput = total_samples / sum(epoch_times)
    peak_memory = max(memory_values) if memory_values else 0

    return {
        "batch_size": batch_size,
        "throughput": round(throughput, 2),
        "peak_memory_mb": round(peak_memory, 2),
        "test_accuracy": round(accuracy, 2)
    }


def main():
    set_seed()

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

    results = []

    for batch_size in [32, 64, 128, 256, 512]:
        print(f"\nTesting batch size: {batch_size}")

        result = run_experiment(
            batch_size,
            train_dataset,
            test_loader,
            device
        )

        results.append(result)
        print("Result:", result)

    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv("results/batch_study.csv", index=False)

    print("\nFinal results:")
    print(df.to_string(index=False))
    print("\nSaved: results/batch_study.csv")


if __name__ == "__main__":
    main()


