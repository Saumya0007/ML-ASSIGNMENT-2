
"""
Q1(c): Multiclass SAMME boosting implemented from scratch.

Dataset: Olivetti Faces
Features: 4096 raw flattened pixels
Split: results/split.npz
Seed: 26007 (roll number m26cps007)

Base tree:
    max_depth=None
    min_samples_leaf=4

Ensemble checkpoints:
    10, 25, 50, 100, 200

Robustness:
    Zero rows 24:40 and columns 24:40 in test images.
    Train exclusively on clean training images.
"""

from pathlib import Path
import json

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


SEED = 26007
SPLIT_PATH = Path("results/split.npz")
OUTPUT_PATH = Path("results/boosting.json")

CHECKPOINTS = [10, 25, 50, 100, 200]

BASE_TREE_CONFIG = {
    "max_depth": None,
    "min_samples_leaf": 4,
}


def load_data():
    """Load raw pixels and the fixed split saved by data_setup.py."""
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
    """
    Set rows 24:40 and columns 24:40 to zero.
    The input has shape (n_images, 64, 64).
    """
    corrupted = test_images.copy()
    corrupted[:, 24:40, 24:40] = 0.0
    return corrupted.reshape(len(corrupted), -1)


class ScratchSAMME:
    """Sequential multiclass SAMME boosting implementation."""

    def __init__(self, seed, base_config):
        self.seed = seed
        self.base_config = base_config
        self.models = []
        self.alphas = []
        self.errors = []
        self.classes_ = None
        self.sample_weights_ = None
        self.rng = None
        self.stopped_reason = None

    def initialize(self, X, y):
        """Initialize the weights and RNG once."""
        self.classes_ = np.unique(y)
        self.rng = np.random.default_rng(self.seed)
        self.sample_weights_ = np.full(
            len(X), 1.0 / len(X), dtype=np.float64
        )

    def add_one(self, X, y):
        """Train and append one boosting learner."""
        n_classes = len(self.classes_)
        weights = self.sample_weights_

        tree = DecisionTreeClassifier(
            **self.base_config,
            random_state=int(
                self.rng.integers(0, 2**31 - 1)
            ),
        )

        # Fit on clean training data using current sample weights.
        tree.fit(X, y, sample_weight=weights)

        predictions = tree.predict(X)
        incorrect = (predictions != y)

        error = float(np.sum(weights * incorrect) / weights.sum())

        # SAMME requires weighted error below 1 - 1/K.
        max_error = 1.0 - 1.0 / n_classes

        if error >= max_error:
            self.stopped_reason = (
                f"Rejected learner: weighted error {error:.6f} "
                f">= {max_error:.6f}"
            )
            return False

        if error <= 1e-12:
            # Perfect learner: adding more learners is unnecessary.
            self.models.append(tree)
            self.alphas.append(1.0)
            self.errors.append(error)
            self.stopped_reason = "Perfect learner reached"
            return False

        alpha = (
            np.log((1.0 - error) / error)
            + np.log(n_classes - 1)
        )

        self.models.append(tree)
        self.alphas.append(float(alpha))
        self.errors.append(error)

        # Update sample weights. Misclassified images gain weight.
        log_weights = np.log(
            np.maximum(self.sample_weights_, 1e-300)
        )
        log_weights += alpha * incorrect
        log_weights -= np.max(log_weights)

        self.sample_weights_ = np.exp(log_weights)
        self.sample_weights_ /= self.sample_weights_.sum()

        return True

    def fit_until(self, X, y, n_estimators):
        """
        Continue training until the requested total ensemble size.
        State and RNG are preserved across checkpoints.
        """
        if self.classes_ is None:
            self.initialize(X, y)

        while len(self.models) < n_estimators:
            if self.stopped_reason is not None:
                break

            added = self.add_one(X, y)

            if not added:
                break

        if not self.models:
            raise RuntimeError(
                "No boosting learner was accepted. "
                "Check the base-tree configuration."
            )

        return self

    def decision_function(self, X):
        """Calculate class scores using weighted votes."""
        scores = np.zeros(
            (len(X), len(self.classes_)),
            dtype=np.float64,
        )

        class_to_index = {
            label: i for i, label in enumerate(self.classes_)
        }

        for tree, alpha in zip(self.models, self.alphas):
            predictions = tree.predict(X)

            for row, label in enumerate(predictions):
                scores[row, class_to_index[label]] += alpha

        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]


def accuracy(model, X, y):
    return float(accuracy_score(y, model.predict(X)))


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
    print("Base tree:", BASE_TREE_CONFIG)
    print()

    # One continuously trained boosting sequence.
    model = ScratchSAMME(
        seed=SEED,
        base_config=BASE_TREE_CONFIG,
    )

    results = []

    for requested_count in CHECKPOINTS:
        model.fit_until(X_train, y_train, requested_count)
        actual_count = len(model.models)

        result = {
            "requested_estimators": requested_count,
            "actual_estimators": actual_count,
            "train_accuracy": accuracy(
                model, X_train, y_train
            ),
            "validation_accuracy": accuracy(
                model, X_val, y_val
            ),
            "clean_test_accuracy": accuracy(
                model, X_test, y_test
            ),
            "corrupted_test_accuracy": accuracy(
                model, X_test_corrupted, y_test
            ),
        }

        results.append(result)

        print(
            f"Requested={requested_count:3d}, "
            f"actual={actual_count:3d} | "
            f"Train={result['train_accuracy']:.4f} | "
            f"Val={result['validation_accuracy']:.4f} | "
            f"Clean test={result['clean_test_accuracy']:.4f} | "
            f"Corrupted test="
            f"{result['corrupted_test_accuracy']:.4f}"
        )

        if actual_count < requested_count:
            print("Boosting stopped early:", model.stopped_reason)
            break

    # Select a checkpoint using validation accuracy only.
    best = max(
        results,
        key=lambda r: (
            r["validation_accuracy"],
            -r["requested_estimators"],
        ),
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "seed": SEED,
        "dataset": "Olivetti Faces",
        "feature_type": "raw flattened pixels",
        "split_file": str(SPLIT_PATH),
        "algorithm": "SAMME multiclass boosting from scratch",
        "base_tree_config": BASE_TREE_CONFIG,
        "selection_metric": "validation_accuracy",
        "best_config": best,
        "all_results": results,
        "learner_errors": model.errors,
        "learner_weights": model.alphas,
        "stopped_reason": model.stopped_reason,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("BEST BOOSTING CONFIGURATION")
    print("=" * 60)
    print(json.dumps(best, indent=2))
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
