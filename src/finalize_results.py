
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def load_json(filename):
    with (RESULTS / filename).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


tree_data = load_json("tree_baseline.json")
bagging_data = load_json("bagging.json")
bias_data = load_json("bias_variance_exact.json")
depth_data = load_json("bagging_depth_bias_variance.json")
samme_data = load_json("samme_corrected_summary.json")

tree = tree_data["best_config"]
bagging = bagging_data["best_config"]
bagging_base = bagging_data["base_tree_config"]

# Use the verified 100-round SAMME result, not the inconsistent
# historical comparison-script output.
samme_checkpoints = {
    int(row["learners"]): row
    for row in samme_data["checkpoint_results"]
}
samme = samme_checkpoints[100]

rows = [
    {
        "model": "Decision tree",
        "configuration": (
            f"depth={tree.get('max_depth')}, "
            f"min_leaf={tree.get('min_samples_leaf')}"
        ),
        "train_accuracy": tree["train_accuracy"],
        "validation_accuracy": tree["validation_accuracy"],
        "clean_test_accuracy": tree["clean_test_accuracy"],
        "masked_test_accuracy": tree["corrupted_test_accuracy"],
    },
    {
        "model": "Bagging",
        "configuration": (
            f"trees={bagging['n_estimators']}, "
            f"depth={bagging_base['max_depth']}, "
            f"min_leaf={bagging_base['min_samples_leaf']}"
        ),
        "train_accuracy": bagging["train_accuracy"],
        "validation_accuracy": bagging["validation_accuracy"],
        "clean_test_accuracy": bagging["clean_test_accuracy"],
        "masked_test_accuracy": bagging["corrupted_test_accuracy"],
    },
    {
        "model": "SAMME boosting",
        "configuration": (
            f"depth=4, rounds=100, min_leaf=1"
        ),
        "train_accuracy": samme["train_accuracy"],
        "validation_accuracy": samme["validation_accuracy"],
        "clean_test_accuracy": samme["clean_test_accuracy"],
        "masked_test_accuracy": samme["masked_test_accuracy"],
    },
]

save_csv(RESULTS / "q3d_final_comparison.csv", rows)

# Selected-model accuracy plot.
metrics = [
    ("train_accuracy", "Training"),
    ("validation_accuracy", "Validation"),
    ("clean_test_accuracy", "Clean test"),
    ("masked_test_accuracy", "Masked test"),
]

fig, ax = plt.subplots(figsize=(10, 6))
x = list(range(len(rows)))
width = 0.19

for i, (key, label) in enumerate(metrics):
    values = [row[key] for row in rows]
    offsets = [position + (i - 1.5) * width for position in x]
    ax.bar(offsets, values, width=width, label=label)

ax.set_xticks(x)
ax.set_xticklabels([row["model"] for row in rows])
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05)
ax.set_title("Selected Model Comparison")
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(
    FIGURES / "final_accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# Robustness loss from central occlusion.
fig, ax = plt.subplots(figsize=(8, 5))
models = [row["model"] for row in rows]
drops = [
    row["clean_test_accuracy"] - row["masked_test_accuracy"]
    for row in rows
]
ax.bar(models, drops)
ax.set_ylabel("Clean minus masked accuracy")
ax.set_title("Effect of Central 16x16 Occlusion")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(
    FIGURES / "final_robustness_drop.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# Q3(a): bias-variance results across resamples.
bias_rows = []
for model_name, result in bias_data["summary"].items():
    bias_rows.append({
        "model": model_name,
        "bias_squared": result["bias_squared"],
        "variance": result["variance"],
        "mean_test_accuracy": result["mean_test_accuracy"],
        "std_test_accuracy": result["std_test_accuracy"],
        "n_resamples": bias_data["n_resamples"],
    })

save_csv(RESULTS / "q3a_bias_variance_summary.csv", bias_rows)

# Q3(b)-(c): corrected depth-sensitivity results (100 trees).
depth_rows = []
for label, result in depth_data["summary"].items():
    depth_rows.append({
        "configuration": label,
        "max_depth": result["max_depth"],
        "min_samples_leaf": result["min_samples_leaf"],
        "n_trees": result["n_trees"],
        "bias_squared": result["bias_squared"],
        "variance": result["variance"],
        "mean_test_accuracy": result["mean_test_accuracy"],
        "std_test_accuracy": result["std_test_accuracy"],
        "mean_leaf_count": result.get("mean_leaf_count"),
        "mean_singleton_leaf_fraction": result.get(
            "mean_singleton_leaf_fraction"
        ),
        "mean_samples_per_leaf": result.get("mean_samples_per_leaf"),
        "mean_tree_disagreement": result.get("mean_tree_disagreement"),
    })

save_csv(RESULTS / "q3b_depth_comparison_summary.csv", depth_rows)

print("\nQ3(d): selected model comparison")
for row in rows:
    print(
        f"{row['model']:18s} "
        f"train={row['train_accuracy']:.4f} "
        f"val={row['validation_accuracy']:.4f} "
        f"clean={row['clean_test_accuracy']:.4f} "
        f"masked={row['masked_test_accuracy']:.4f}"
    )

print("\nQ3(a): bias-variance summary")
for row in bias_rows:
    print(
        f"{row['model']:18s} "
        f"bias_squared={row['bias_squared']:.6f} "
        f"variance={row['variance']:.6f} "
        f"mean_accuracy={row['mean_test_accuracy']:.4f}"
    )

print("\nQ3(b)-(c): depth comparison")
for row in depth_rows:
    print(
        f"{row['configuration']:26s} "
        f"depth={row['max_depth']} "
        f"trees={row['n_trees']} "
        f"bias_squared={row['bias_squared']:.6f} "
        f"variance={row['variance']:.6f} "
        f"mean_accuracy={row['mean_test_accuracy']:.4f} "
        f"disagreement={row['mean_tree_disagreement']}"
    )

print("\nSaved final tables and figures.")
