
"""
tree_baseline.py

Q1(a): Decision-tree baseline with robustness evaluation.

Dataset: Olivetti Faces
Features: 4096 raw flattened pixels
Split: results/split.npz
Seed: 26007

Hyperparameter grid:
    max_depth = [2, 4, 8, 16, None]
    min_samples_leaf = [1, 2, 4]

Robustness:
    Set rows 24:40 and columns 24:40 to zero
    in test images only. Train on clean images.
"""

from pathlib import Path
import json

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


SEED = 26007
SPLIT_PATH = Path("results/split.npz")
OUTPUT_PATH = Path("results/tree_baseline.json")

DEPTHS = [2, 4, 8, 16, None]
MIN_SAMPLES_LEAF = [1, 2, 4]


def load_data():
    """Load raw pixels and the fixed saved split."""
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"{SPLIT_PATH} not found. Run src/data_setup.py first."
        )

    dataset = fetch_olivetti_faces()
    images = dataset.images
    y = dataset.target
    X = images.reshape(len(images), -1)

    with np.load(SPLIT_PATH) as split:
        train_idx = split["train_indices"].copy()
        val_idx = split["val_indices"].copy()
        test_idx = split["test_indices"].copy()

    assert len(train_idx) == 240
    assert len(val_idx) == 80
    assert len(test_idx) == 80

    assert len(np.unique(np.concatenate(
        [train_idx, val_idx, test_idx]
    ))) == 400

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx],
        images[test_idx].copy(),
    )


def corrupt_test_images(test_images):
    """Zero the centered 16x16 block of each test image."""
    corrupted = test_images.copy()
    corrupted[:, 24:40, 24:40] = 0.0
    return corrupted.reshape(len(corrupted), -1)


def evaluate_accuracy(model, X, y):
    return float(accuracy_score(y, model.predict(X)))


def evaluate_tree(model, X_train):
    """Compute leaf statistics on the fitted training tree."""
    tree = model.tree_
    leaf_mask = tree.children_left == -1
    leaf_samples = tree.n_node_samples[leaf_mask]

    return {
        "leaf_count": int(leaf_mask.sum()),
        "mean_samples_per_leaf": float(leaf_samples.mean()),
        "singleton_leaf_fraction": float(
            np.mean(leaf_samples == 1)
        ),
    }


def main():
    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        test_images,
    ) = load_data()

    X_test_corrupted = corrupt_test_images(test_images)

    print("Loaded fixed split.")
    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Clean test shape:", X_test.shape)
    print("Corrupted test shape:", X_test_corrupted.shape)
    print()

    results = []

    for depth in DEPTHS:
        for min_leaf in MIN_SAMPLES_LEAF:
            model = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=min_leaf,
                random_state=SEED,
            )

            # Fit only on clean training images.
            model.fit(X_train, y_train)

            stats = evaluate_tree(model, X_train)

            result = {
                "max_depth": depth,
                "min_samples_leaf": min_leaf,
                "train_accuracy": evaluate_accuracy(
                    model, X_train, y_train
                ),
                "validation_accuracy": evaluate_accuracy(
                    model, X_val, y_val
                ),
                "clean_test_accuracy": evaluate_accuracy(
                    model, X_test, y_test
                ),
                "corrupted_test_accuracy": evaluate_accuracy(
                    model, X_test_corrupted, y_test
                ),
                **stats,
            }

            results.append(result)

            print(
                f"depth={str(depth):>4}, "
                f"min_leaf={min_leaf} | "
                f"Train={result['train_accuracy']:.4f} | "
                f"Val={result['validation_accuracy']:.4f} | "
                f"Clean test={result['clean_test_accuracy']:.4f} | "
                f"Corrupted test="
                f"{result['corrupted_test_accuracy']:.4f}"
            )

    # Select using validation accuracy only.
    best = max(
        results,
        key=lambda r: (
            r["validation_accuracy"],
            -(
                r["max_depth"]
                if r["max_depth"] is not None
                else 10**9
            ),
            r["min_samples_leaf"],
        ),
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "seed": SEED,
        "dataset": "Olivetti Faces",
        "feature_type": "raw flattened pixels",
        "split_file": str(SPLIT_PATH),
        "selection_metric": "validation_accuracy",
        "best_config": best,
        "all_results": results,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("SELECTED DECISION TREE")
    print("=" * 60)
    print(json.dumps(best, indent=2))
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
