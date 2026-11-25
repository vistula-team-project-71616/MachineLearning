
import os
import random
import time
import yaml
import torch
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.profiler import profile, ProfilerActivity, record_function


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

seed = config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("results", exist_ok=True)
os.makedirs(config["checkpoint"]["directory"], exist_ok=True)


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

test_dataset = datasets.FashionMNIST(
    config["data"]["path"],
    train=False,
    download=True,
    transform=transform
)

train_dataset, val_dataset = random_split(
    dataset,
    [config["data"]["train_size"], config["data"]["val_size"]],
    generator=torch.Generator().manual_seed(seed)
)

generator = torch.Generator().manual_seed(seed)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["data"]["batch_size"],
    shuffle=True,
    num_workers=config["data"]["num_workers"],
    generator=generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config["data"]["batch_size"]
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config["data"]["batch_size"]
)


model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 7 * 7, 10)
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config["training"]["learning_rate"]
)


def accuracy(model, loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(1)
            correct += (predictions == labels).sum().item()

    return correct / len(loader.dataset) * 100


history = []
best_accuracy = 0

checkpoint_path = os.path.join(
    config["checkpoint"]["directory"],
    config["checkpoint"]["filename"]
)

mlflow.set_experiment(config["mlflow"]["experiment_name"])


with mlflow.start_run(run_name=config["mlflow"]["run_name"]):

    mlflow.log_params({
        "seed": seed,
        "batch_size": config["data"]["batch_size"],
        "learning_rate": config["training"]["learning_rate"],
        "epochs": config["training"]["epochs"],
        "optimizer": config["training"]["optimizer"],
        "model": config["model"]["name"],
        "num_workers": config["data"]["num_workers"]
    })

    print("Device:", device)

    for epoch in range(config["training"]["epochs"]):
        model.train()
        total_loss = 0

        stage_times = {
            "Data Loading": 0,
            "Forward Pass": 0,
            "Backward Pass": 0,
            "Optimizer Step": 0
        }

        data_iterator = iter(train_loader)
        step = 0

        profiler = None

        if epoch == 0:
            profiler = profile(
                activities=[ProfilerActivity.CPU],
                record_shapes=False,
                profile_memory=False
            )
            profiler.__enter__()

        while step < len(train_loader):

            start = time.perf_counter()
            try:
                images, labels = next(data_iterator)
            except StopIteration:
                break

            stage_times["Data Loading"] += time.perf_counter() - start

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if profiler is not None and step < 10:
                with record_function("Forward Pass"):
                    start = time.perf_counter()
                    output = model(images)
                    loss = criterion(output, labels)
                    stage_times["Forward Pass"] += time.perf_counter() - start

                with record_function("Backward Pass"):
                    start = time.perf_counter()
                    loss.backward()
                    stage_times["Backward Pass"] += time.perf_counter() - start

                with record_function("Optimizer Step"):
                    start = time.perf_counter()
                    optimizer.step()
                    stage_times["Optimizer Step"] += time.perf_counter() - start

            else:
                start = time.perf_counter()
                output = model(images)
                loss = criterion(output, labels)
                stage_times["Forward Pass"] += time.perf_counter() - start

                start = time.perf_counter()
                loss.backward()
                stage_times["Backward Pass"] += time.perf_counter() - start

                start = time.perf_counter()
                optimizer.step()
                stage_times["Optimizer Step"] += time.perf_counter() - start

            total_loss += loss.item() * len(labels)
            step += 1

            if profiler is not None and step == 10:
                profiler.__exit__(None, None, None)

                with open("results/profiling.txt", "w") as file:
                    file.write(
                        profiler.key_averages().table(
                            sort_by="self_cpu_time_total",
                            row_limit=30
                        )
                    )

                print("Torch profiling completed.")

                profiler = None

        if profiler is not None:
            profiler.__exit__(None, None, None)

            with open("results/profiling.txt", "w") as file:
                file.write(
                    profiler.key_averages().table(
                        sort_by="self_cpu_time_total",
                        row_limit=30
                    )
                )

        total_time = sum(stage_times.values())

        with open("results/profiling_breakdown.csv", "w") as file:
            file.write("Stage,Time (seconds),Percentage\n")

            for stage, elapsed in stage_times.items():
                percentage = elapsed / total_time * 100
                file.write(
                    f"{stage},{elapsed:.4f},{percentage:.2f}\n"
                )

        train_loss = total_loss / len(train_dataset)
        val_accuracy = accuracy(model, val_loader)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_accuracy": val_accuracy
        })

        mlflow.log_metric(
            "train_loss", train_loss, step=epoch + 1
        )
        mlflow.log_metric(
            "val_accuracy", val_accuracy, step=epoch + 1
        )

        if epoch == 0:
            for stage, elapsed in stage_times.items():
                mlflow.log_metric(
                    stage.lower().replace(" ", "_") + "_time",
                    elapsed
                )
                mlflow.log_metric(
                    stage.lower().replace(" ", "_") + "_percent",
                    elapsed / total_time * 100
                )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_accuracy
            }, checkpoint_path)

        print(
            f"Epoch {epoch + 1}: "
            f"Loss={train_loss:.4f}, "
            f"Val Accuracy={val_accuracy:.2f}%"
        )

        if epoch == 0:
            print("\nProfiling breakdown:")
            for stage, elapsed in stage_times.items():
                percentage = elapsed / total_time * 100
                print(
                    f"{stage}: {elapsed:.2f}s "
                    f"({percentage:.2f}%)"
                )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_accuracy = accuracy(model, test_loader)

    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("best_val_accuracy", best_accuracy)

    pd.DataFrame(history).to_csv(
        "results/training_history.csv",
        index=False
    )

    plt.plot(
        [item["epoch"] for item in history],
        [item["train_loss"] for item in history],
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss")
    plt.grid()
    plt.savefig("results/training_loss.png")
    plt.close()

    for artifact in [
        checkpoint_path,
        "config.yaml",
        "results/training_history.csv",
        "results/training_loss.png",
        "results/profiling.txt",
        "results/profiling_breakdown.csv"
    ]:
        mlflow.log_artifact(artifact)

    print("\nBest validation accuracy:", round(best_accuracy, 2), "%")
    print("Test accuracy:", round(test_accuracy, 2), "%")
    print("Checkpoint saved:", checkpoint_path)



