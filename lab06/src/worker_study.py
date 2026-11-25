
import os
import time
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from statistics import median


def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    dataset = datasets.FashionMNIST(
        config["data"]["path"],
        train=True,
        download=True,
        transform=transform
    )

    train_dataset, _ = random_split(
        dataset,
        [config["data"]["train_size"], config["data"]["val_size"]],
        generator=torch.Generator().manual_seed(seed)
    )

    batch_size = config["data"]["batch_size"]
    worker_values = [0, 1, 2, 4, 8]
    results = []

    os.makedirs("results", exist_ok=True)

    for workers in worker_values:
        print(f"\nTesting num_workers={workers}", flush=True)

        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers
        )

        def measure():
            start = time.perf_counter()
            samples = 0

            for images, labels in loader:
                samples += len(labels)

            elapsed = time.perf_counter() - start
            throughput = samples / elapsed

            return elapsed, throughput

        warmup_time, _ = measure()

        times = []
        throughputs = []

        for run in range(5):
            elapsed, throughput = measure()
            times.append(elapsed)
            throughputs.append(throughput)

            print(
                f"Run {run + 1}/5: "
                f"{elapsed:.2f}s, "
                f"{throughput:.2f} samples/s",
                flush=True
            )

        result = {
            "num_workers": workers,
            "median_time": median(times),
            "throughput": median(throughputs),
            "warmup_time": warmup_time
        }

        results.append(result)

        print(
            f"Median time: {result['median_time']:.2f}s",
            f"Throughput: {result['throughput']:.2f} samples/s",
            flush=True
        )

    with open("results/worker_study.csv", "w") as file:
        file.write(
            "num_workers,median_time,throughput,warmup_time\n"
        )

        for result in results:
            file.write(
                f"{result['num_workers']},"
                f"{result['median_time']:.4f},"
                f"{result['throughput']:.2f},"
                f"{result['warmup_time']:.4f}\n"
            )

    plt.figure()
    plt.plot(
        [r["num_workers"] for r in results],
        [r["throughput"] for r in results],
        marker="o"
    )

    plt.xlabel("Number of Workers")
    plt.ylabel("Throughput (samples/s)")
    plt.title("Data Loading Performance")
    plt.grid()
    plt.savefig("results/worker_study.png")
    plt.close()

    print("\nResults saved.")
    print("Best num_workers:", max(
        results, key=lambda r: r["throughput"]
    )["num_workers"])


if __name__ == "__main__":
    main()


