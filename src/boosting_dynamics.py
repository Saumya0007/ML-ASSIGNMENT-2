
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

input_path = RESULTS_DIR / "boosting_grid_search.json"

with input_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

rows = data["grid_results"]

# Save the exact data used in the plot.
csv_path = RESULTS_DIR / "boosting_dynamics.csv"
fieldnames = [
    "max_depth",
    "n_estimators",
    "train_accuracy",
    "validation_accuracy",
    "clean_test_accuracy",
    "corrupted_test_accuracy",
]

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row[key] for key in fieldnames})

# Plot training and validation accuracy for each depth.
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
axes = axes.flatten()

for ax, depth in zip(axes, [1, 2, 3, 4]):
    depth_rows = sorted(
        [r for r in rows if r["max_depth"] == depth],
        key=lambda r: r["n_estimators"],
    )

    rounds = [r["n_estimators"] for r in depth_rows]
    train_acc = [r["train_accuracy"] for r in depth_rows]
    val_acc = [r["validation_accuracy"] for r in depth_rows]

    ax.plot(rounds, train_acc, marker="o", label="Training accuracy")
    ax.plot(rounds, val_acc, marker="s", label="Validation accuracy")
    ax.set_title(f"Tree depth = {depth}")
    ax.set_xlabel("Boosting rounds")
    ax.set_ylabel("Accuracy")
    ax.set_xticks([10, 25, 50, 100])
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.suptitle("SAMME Boosting: Training and Validation Dynamics")
fig.tight_layout()

plot_path = FIGURES_DIR / "boosting_dynamics.png"
fig.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close(fig)

# Print the train-validation gap at every checkpoint.
print("\nTraining-validation gaps")
for row in sorted(
    rows, key=lambda r: (r["max_depth"], r["n_estimators"])
):
    gap = row["train_accuracy"] - row["validation_accuracy"]
    print(
        f"depth={row['max_depth']}, "
        f"rounds={row['n_estimators']:3d}, "
        f"train={row['train_accuracy']:.4f}, "
        f"validation={row['validation_accuracy']:.4f}, "
        f"gap={gap:.4f}"
    )

print(f"\nSaved plot: {plot_path}")
print(f"Saved CSV:  {csv_path}")
