
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_DIR / "results"

print("Project folder:", PROJECT_DIR)
print("Results folder:", RESULTS)

if RESULTS.exists():
    print("\nFiles in results:")
    for file in RESULTS.iterdir():
        print("-", file.name)
else:
    print("\nERROR: Results folder not found!")
    exit()

pruning_file = RESULTS / "pruning_results.csv"
efficiency_file = RESULTS / "efficiency_results.csv"
distillation_file = RESULTS / "distillation_results.csv"

missing_files = [
    file.name
    for file in [pruning_file, efficiency_file, distillation_file]
    if not file.exists()
]

if missing_files:
    print("\nMissing CSV files:")
    for name in missing_files:
        print("-", name)
    print("\nCheck the filenames in the results folder.")
    exit()

pruning = pd.read_csv(pruning_file)
efficiency = pd.read_csv(efficiency_file)
distillation = pd.read_csv(distillation_file)

print("\n===== PRUNING SUMMARY =====")
print(pruning.to_string(index=False))

print("\n===== EFFICIENCY SUMMARY =====")
print(efficiency.to_string(index=False))

print("\n===== DISTILLATION SUMMARY =====")
print(distillation.to_string(index=False))

print("\n===== FINAL MODEL COMPARISON =====")

final_table = distillation[
    ["model", "parameters", "test_accuracy", "size_mb", "latency_ms"]
].copy()

final_table.to_csv(RESULTS / "final_model_comparison.csv", index=False)
print(final_table.to_string(index=False))

print("\n===== CHAPTER 9: EFFICIENCY PILLARS =====")

pillars = pd.DataFrame([
    {
        "Pillar": "Data",
        "Experiment": "Fashion-MNIST",
        "Result": "Same dataset and data split used in all experiments"
    },
    {
        "Pillar": "Model",
        "Experiment": "Pruning and Knowledge Distillation",
        "Result": "Reduced weights and smaller student models"
    },
    {
        "Pillar": "Compute",
        "Experiment": "CPU inference",
        "Result": "Student models had lower latency than the teacher"
    },
    {
        "Pillar": "Energy",
        "Experiment": "Not measured",
        "Result": "No direct energy measurements were available"
    }
])

pillars.to_csv(RESULTS / "efficiency_pillars.csv", index=False)
print(pillars.to_string(index=False))

print("\nFinal analysis completed.")
print("Results saved in:", RESULTS)
