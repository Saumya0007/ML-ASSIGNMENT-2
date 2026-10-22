
"""
make_figures.py

Generate reproducible figures and a comparison table from:
    results/tree_baseline.json
    results/bagging.json
    results/boosting.json

Outputs:
    results/figures/accuracy_comparison.png
    results/figures/robustness_comparison.png
    results/figures/comparison_table.csv
"""

from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results")
FIGURE_DIR = RESULTS_DIR / "figures"

TREE_FILE = RESULTS_DIR / "tree_baseline.json"
BAGGING_FILE = RESULTS_DIR / "bagging.json"
BOOSTING_FILE = RESULTS_DIR / "boosting.json"


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the corresponding experiment first."
        )
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_best(data):
    return data["best_config"]


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    tree_data = read_json(TREE_FILE)
    bagging_data = read_json(BAGGING_FILE)
    boosting_data = read_json(BOOSTING_FILE)

    tree = get_best(tree_data)
    bagging = get_best(bagging_data)
    boosting = get_best(boosting_data)

    # Normalize the tree result key names.
    selected = [
        {
            "model": "Decision tree",
            "configuration": (
                f"depth={tree['max_depth']}, "
                f"min_leaf={tree['min_samples_leaf']}"
            ),
            "validation_accuracy": tree["validation_accuracy"],
            "clean_test_accuracy": tree.get(
                "clean_test_accuracy", tree.get("test_accuracy")
            ),
            "corrupted_test_accuracy": tree[
                "corrupted_test_accuracy"
            ],
        },
        {
            "model": "Bagging",
            "configuration": f"trees={bagging['n_estimators']}",
            "validation_accuracy": bagging["validation_accuracy"],
            "clean_test_accuracy": bagging["clean_test_accuracy"],
            "corrupted_test_accuracy":
                bagging["corrupted_test_accuracy"],
        },
        {
            "model": "Boosting (SAMME)",
            "configuration":
                f"learners={boosting['requested_estimators']}",
            "validation_accuracy": boosting["validation_accuracy"],
            "clean_test_accuracy": boosting["clean_test_accuracy"],
            "corrupted_test_accuracy":
                boosting["corrupted_test_accuracy"],
        },
    ]

    names = [row["model"] for row in selected]
    validation = [
        100 * row["validation_accuracy"] for row in selected
    ]
    clean = [
        100 * row["clean_test_accuracy"] for row in selected
    ]
    corrupted = [
        100 * row["corrupted_test_accuracy"] for row in selected
    ]

    # Figure 1: Validation and clean test accuracy.
    x = np.arange(len(names))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(
        x - width / 2, validation, width,
        label="Validation accuracy"
    )
    bars2 = ax.bar(
        x + width / 2, clean, width,
        label="Clean test accuracy"
    )

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Decision Tree, Bagging and Boosting")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.bar_label(bars1, fmt="%.2f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.2f", padding=3, fontsize=8)
    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "accuracy_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 2: Clean versus corrupted test accuracy.
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(
        x - width / 2, clean, width,
        label="Clean test"
    )
    bars2 = ax.bar(
        x + width / 2, corrupted, width,
        label="Corrupted test"
    )

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Robustness to Center-Block Occlusion")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.bar_label(bars1, fmt="%.2f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.2f", padding=3, fontsize=8)
    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "robustness_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Save a machine-readable comparison table.
    table_path = FIGURE_DIR / "comparison_table.csv"

    with table_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "configuration",
                "validation_accuracy",
                "clean_test_accuracy",
                "corrupted_test_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(selected)

    print("Figures and comparison table created:")
    print(FIGURE_DIR / "accuracy_comparison.png")
    print(FIGURE_DIR / "robustness_comparison.png")
    print(table_path)

    print("\nSelected-model comparison:")
    for row in selected:
        print(
            f"{row['model']}: "
            f"validation={row['validation_accuracy']:.4f}, "
            f"clean_test={row['clean_test_accuracy']:.4f}, "
            f"corrupted_test={row['corrupted_test_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
