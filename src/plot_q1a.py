
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
input_file = ROOT / "results" / "tree_baseline.json"
output_file = ROOT / "results" / "figures" / "q1a_tree_grid.png"

with open(input_file, "r", encoding="utf-8") as f:
    results = json.load(f)

rows = results["all_results"]
depths = [2, 4, 8, 16, None]
depth_labels = ["2", "4", "8", "16", "None"]
leaf_sizes = [1, 2, 4]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

metrics = [
    ("train_accuracy", "Training accuracy"),
    ("validation_accuracy", "Validation accuracy"),
]

for ax, (metric, title) in zip(axes, metrics):
    for leaf_size in leaf_sizes:
        selected = [
            next(
                r for r in rows
                if r["max_depth"] == depth
                and r["min_samples_leaf"] == leaf_size
            )
            for depth in depths
        ]

        values = [r[metric] * 100 for r in selected]
        ax.plot(
            depth_labels,
            values,
            marker="o",
            linewidth=1.8,
            label=f"min_samples_leaf={leaf_size}",
        )

    ax.set_title(title)
    ax.set_xlabel("Maximum tree depth")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)

# Highlight the selected model on the validation plot.
axes[1].scatter(
    ["None"], [61.25],
    marker="*",
    s=180,
    color="red",
    zorder=5,
    label="Selected model",
)
axes[1].legend(fontsize=8)

fig.suptitle("Single Decision Tree: Hyperparameter Search", fontsize=12)
fig.tight_layout()

output_file.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved figure to: {output_file}")
