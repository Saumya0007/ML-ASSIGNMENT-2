
from pathlib import Path
import json

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

SEED = 26007
DEPTHS = [1, 2, 3, 4]
ROUND_CHECKPOINTS = [10, 25, 50, 100]
MAX_ROUNDS = max(ROUND_CHECKPOINTS)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load dataset and reuse the saved split
# --------------------------------------------------
dataset = fetch_olivetti_faces(shuffle=False)
images = dataset.images
X = images.reshape(len(images), -1)
y = dataset.target

split = np.load(RESULTS_DIR / "split.npz")
train_idx = split["train_indices"]
val_idx = split["val_indices"]
test_idx = split["test_indices"]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Corrupt only the test images. Training remains clean.
corrupted_images = images[test_idx].copy()
corrupted_images[:, 24:40, 24:40] = 0
X_test_corrupted = corrupted_images.reshape(len(test_idx), -1)

classes = np.unique(y_train)
n_classes = len(classes)

# --------------------------------------------------
# Scratch implementation of multiclass SAMME
# --------------------------------------------------
class ScratchSAMME:
    def __init__(self, max_depth, n_estimators, seed=SEED):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.seed = seed
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []

    def fit_until(self, X, y, checkpoints):
        """Train one continuous sequence and evaluate checkpoints."""
        checkpoints = sorted(set(checkpoints))
        sample_weights = np.full(len(y), 1.0 / len(y))
        results = {}
        n_classes = len(np.unique(y))

        for round_number in range(1, max(checkpoints) + 1):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=1,
                random_state=self.seed + round_number - 1,
            )

            tree.fit(X, y, sample_weight=sample_weights)
            predictions = tree.predict(X)
            incorrect = predictions != y

            error = float(
                np.sum(sample_weights * incorrect)
                / np.sum(sample_weights)
            )

            # Reject a learner no better than random guessing.
            if error >= 1.0 - 1.0 / n_classes:
                print(
                    f"Depth {self.max_depth}: stopping at round "
                    f"{round_number}; weighted error {error:.6f}"
                )
                break

            # SAMME multiclass learner weight.
            error = float(np.clip(error, 1e-12, 1 - 1e-12))
            alpha = np.log((1.0 - error) / error)
            alpha += np.log(n_classes - 1)

            # Update and normalize sample weights.
            sample_weights *= np.exp(alpha * incorrect)
            sample_weights /= sample_weights.sum()

            self.estimators_.append(tree)
            self.estimator_weights_.append(float(alpha))
            self.estimator_errors_.append(error)

            if round_number in checkpoints:
                results[round_number] = self

        return results

    def predict(self, X):
        if not self.estimators_:
            raise RuntimeError("No accepted weak learners.")

        # Weighted multiclass voting.
        scores = np.zeros((len(X), n_classes))

        for tree, alpha in zip(
            self.estimators_, self.estimator_weights_
        ):
            predictions = tree.predict(X)
            for class_position, class_label in enumerate(classes):
                scores[:, class_position] += (
                    alpha * (predictions == class_label)
                )

        return classes[np.argmax(scores, axis=1)]


def evaluate(model, X, y):
    return float(accuracy_score(y, model.predict(X)))


# --------------------------------------------------
# Run grid: 4 depths x 4 round checkpoints
# --------------------------------------------------
all_results = []

for depth in DEPTHS:
    print(f"\nTraining SAMME with max_depth={depth}")

    model = ScratchSAMME(
        max_depth=depth,
        n_estimators=MAX_ROUNDS,
        seed=SEED,
    )

    checkpoint_models = model.fit_until(
        X_train, y_train, ROUND_CHECKPOINTS
    )

    for requested_rounds in ROUND_CHECKPOINTS:
        if requested_rounds not in checkpoint_models:
            print(
                f"  depth={depth}, rounds={requested_rounds}: "
                "not available (training stopped early)"
            )
            continue

        # The stored model is the same continuous sequence,
        # so temporarily evaluate only the requested prefix.
        full_estimators = model.estimators_
        full_weights = model.estimator_weights_
        full_errors = model.estimator_errors_

        model.estimators_ = full_estimators[:requested_rounds]
        model.estimator_weights_ = full_weights[:requested_rounds]
        model.estimator_errors_ = full_errors[:requested_rounds]

        row = {
            "max_depth": depth,
            "n_estimators": requested_rounds,
            "actual_estimators": len(model.estimators_),
            "train_accuracy": evaluate(model, X_train, y_train),
            "validation_accuracy": evaluate(model, X_val, y_val),
            "clean_test_accuracy": evaluate(model, X_test, y_test),
            "corrupted_test_accuracy": evaluate(
                model, X_test_corrupted, y_test
            ),
            "last_learner_weight": model.estimator_weights_[-1],
            "last_weighted_error": model.estimator_errors_[-1],
        }
        all_results.append(row)

        print(
            f"  rounds={requested_rounds:3d} | "
            f"train={row['train_accuracy']:.4f} | "
            f"val={row['validation_accuracy']:.4f} | "
            f"test={row['clean_test_accuracy']:.4f} | "
            f"corrupt={row['corrupted_test_accuracy']:.4f}"
        )

        # Restore the full sequence for the next checkpoint.
        model.estimators_ = full_estimators
        model.estimator_weights_ = full_weights
        model.estimator_errors_ = full_errors

# --------------------------------------------------
# Select using validation accuracy only.
# Ties: fewer rounds, then shallower depth.
# --------------------------------------------------
if not all_results:
    raise RuntimeError(
        "No valid grid configurations. Check weak learner errors."
    )

best = sorted(
    all_results,
    key=lambda row: (
        -row["validation_accuracy"],
        row["n_estimators"],
        row["max_depth"],
    ),
)[0]

output = {
    "seed": SEED,
    "split_file": "results/split.npz",
    "depths": DEPTHS,
    "round_checkpoints": ROUND_CHECKPOINTS,
    "selection_metric": "validation_accuracy",
    "grid_results": all_results,
    "best_config": best,
}

output_path = RESULTS_DIR / "boosting_grid_search.json"
with output_path.open("w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("\nBest configuration selected using validation accuracy:")
print(json.dumps(best, indent=2))
print(f"\nSaved results to: {output_path}")
