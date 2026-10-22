
"""
Q1(b-c): Bagging from scratch with OOB and diversity metrics.

Records at B = 1, 5, 10, ..., 200:
- Train, validation, clean-test and corrupted-test accuracy
- Average number of distinct examples in bootstrap samples
- Out-of-bag (OOB) accuracy
- Mean pairwise disagreement between trees
- Mean vote margin

Features: raw flattened pixels (4096)
Split: results/split.npz
Seed: 26007
Base tree: max_depth=None, min_samples_leaf=4
"""

from pathlib import Path
import json

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


SEED = 26007
SPLIT_PATH = Path("results/split.npz")
OUTPUT_PATH = Path("results/bagging.json")

MAX_TREES = 200

# Include B=1 and preserve the original 5-tree checkpoints.
CHECKPOINTS ={1, 5, 10, 25, 50, 100, 200}

BASE_TREE_CONFIG = {
    "max_depth": None,
    "min_samples_leaf": 4,
}


def load_data():
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


def corrupt_test_images(images):
    """Zero rows 24:40 and columns 24:40."""
    corrupted = images.copy()
    corrupted[:, 24:40, 24:40] = 0.0
    return corrupted.reshape(len(corrupted), -1)


def accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))


def majority_vote(predictions, classes):
    """
    predictions shape: (number_of_trees, number_of_samples).
    Return class predictions and per-class vote fractions.
    """
    n_trees, n_samples = predictions.shape
    votes = np.zeros(
        (n_samples, len(classes)),
        dtype=np.float64,
    )

    class_indices = np.searchsorted(classes, predictions)

    for tree_index in range(n_trees):
        votes[
            np.arange(n_samples), class_indices[tree_index]
        ] += 1.0

    votes /= n_trees
    predicted_labels = classes[np.argmax(votes, axis=1)]

    return predicted_labels, votes


def mean_vote_margin(votes):
    """Mean winning vote fraction minus runner-up vote fraction."""
    ordered = np.sort(votes, axis=1)

    if ordered.shape[1] < 2:
        return 1.0

    margins = ordered[:, -1] - ordered[:, -2]
    return float(np.mean(margins))


def pairwise_disagreement(predictions):
    """
    Mean fraction of test examples on which a pair of trees
    predicts different classes. Each unordered pair is counted once.
    """
    n_trees = predictions.shape[0]

    if n_trees < 2:
        return 0.0

    total = 0
    pair_count = 0

    for i in range(n_trees - 1):
        disagreements = np.sum(
            predictions[i + 1:] != predictions[i]
        )
        total += int(disagreements)
        pair_count += n_trees - i - 1

    return float(total / (pair_count * predictions.shape[1]))


def oob_accuracy(train_labels, train_predictions, inbag_masks):
    """
    Aggregate predictions only from trees for which each
    training example was absent from the bootstrap sample.
    """
    n_trees, n_samples = train_predictions.shape
    oob_predictions = [[] for _ in range(n_samples)]

    for tree_idx in range(n_trees):
        oob_indices = np.flatnonzero(~inbag_masks[tree_idx])

        for sample_idx in oob_indices:
            oob_predictions[sample_idx].append(
                train_predictions[tree_idx, sample_idx]
            )

    covered = [
        i for i, predictions in enumerate(oob_predictions)
        if predictions
    ]

    if not covered:
        return None, 0

    y_true = []
    y_pred = []

    for i in covered:
        values, counts = np.unique(
            oob_predictions[i], return_counts=True
        )
        # np.unique sorts labels; ties choose the lowest label.
        prediction = values[np.argmax(counts)]

        y_true.append(train_labels[i])
        y_pred.append(prediction)

    return accuracy(np.asarray(y_true), np.asarray(y_pred)), len(covered)


def theoretical_distinct_count(n_samples):
    """Expected distinct observations in one bootstrap sample."""
    return float(
        n_samples * (1.0 - (1.0 - 1.0 / n_samples) ** n_samples)
    )


def main():
    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        test_images,
    ) = load_data()

    X_test_corrupted = corrupt_test_images(test_images)
    classes = np.unique(y_train)

    n_train = len(X_train)
    expected_distinct = theoretical_distinct_count(n_train)

    print("Loaded fixed split.")
    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Clean test:", X_test.shape)
    print("Corrupted test:", X_test_corrupted.shape)
    print("Base tree:", BASE_TREE_CONFIG)
    print("Training one sequence of 200 trees.")
    print(f"Theoretical distinct examples per bootstrap: "
          f"{expected_distinct:.3f}")
    print()

    # This RNG sequence makes the bootstrap samples reproducible.
    rng = np.random.default_rng(SEED)

    inbag_masks = []
    train_predictions = []
    val_predictions = []
    test_predictions = []
    corrupted_predictions = []
    distinct_counts = []

    results = []

    for tree_idx in range(MAX_TREES):
        # Draw a bootstrap sample of the original training-set size.
        bootstrap_indices = rng.choice(
            n_train,
            size=n_train,
            replace=True,
        )

        # Count unique observations in this bootstrap sample.
        n_distinct = len(np.unique(bootstrap_indices))
        distinct_counts.append(n_distinct)

        # Record which original training observations were included.
        inbag = np.zeros(n_train, dtype=bool)
        inbag[bootstrap_indices] = True
        inbag_masks.append(inbag)

        # Keep the same tree-seed sequence as the original script.
        tree = DecisionTreeClassifier(
            **BASE_TREE_CONFIG,
            random_state=SEED + tree_idx,
        )
        tree.fit(
            X_train[bootstrap_indices],
            y_train[bootstrap_indices],
        )

        # Cache predictions for consistent checkpoint evaluation.
        train_predictions.append(tree.predict(X_train))
        val_predictions.append(tree.predict(X_val))
        test_predictions.append(tree.predict(X_test))
        corrupted_predictions.append(
            tree.predict(X_test_corrupted)
        )

        n_trees = tree_idx + 1

        if n_trees not in CHECKPOINTS:
            continue

        train_matrix = np.asarray(train_predictions)
        val_matrix = np.asarray(val_predictions)
        test_matrix = np.asarray(test_predictions)
        corrupted_matrix = np.asarray(corrupted_predictions)
        inbag_matrix = np.asarray(inbag_masks)

        train_pred, _ = majority_vote(train_matrix, classes)
        val_pred, _ = majority_vote(val_matrix, classes)
        test_pred, test_votes = majority_vote(test_matrix, classes)
        corrupted_pred, _ = majority_vote(
            corrupted_matrix, classes
        )

        oob_acc, oob_covered = oob_accuracy(
            y_train, train_matrix, inbag_matrix
        )

        disagreement = pairwise_disagreement(test_matrix)
        margin = mean_vote_margin(test_votes)

        mean_distinct = float(np.mean(distinct_counts))
        result = {
            "n_estimators": n_trees,
            "train_accuracy": accuracy(y_train, train_pred),
            "validation_accuracy": accuracy(y_val, val_pred),
            "clean_test_accuracy": accuracy(y_test, test_pred),
            "corrupted_test_accuracy": accuracy(
                y_test, corrupted_pred
            ),
            "mean_distinct_examples": mean_distinct,
            "theoretical_distinct_examples": expected_distinct,
            "oob_accuracy": oob_acc,
            "oob_covered_samples": oob_covered,
            "pairwise_disagreement": disagreement,
            "mean_vote_margin": margin,
        }
        results.append(result)

        print(
            f"B={n_trees:3d} | "
            f"Train={result['train_accuracy']:.4f} | "
            f"Val={result['validation_accuracy']:.4f} | "
            f"Test={result['clean_test_accuracy']:.4f} | "
            f"Distinct={mean_distinct:.2f} | "
            f"Theory={expected_distinct:.2f} | "
            f"Corrupt={result['corrupted_test_accuracy']:.4f} | "
            f"OOB={oob_acc if oob_acc is not None else float('nan'):.4f} | "
            f"Disagreement={disagreement:.4f} | "
            f"Margin={margin:.4f}"
        )

    # Select the best checkpoint using validation accuracy only.
    best = max(
        results,
        key=lambda r: (
            r["validation_accuracy"],
            -r["n_estimators"],
        ),
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "seed": SEED,
        "dataset": "Olivetti Faces",
        "feature_type": "raw flattened pixels",
        "split_file": str(SPLIT_PATH),
        "algorithm": "scratch bootstrap aggregation",
        "base_tree_config": BASE_TREE_CONFIG,
        "selection_metric": "validation_accuracy",
        "bootstrap_sample_size": n_train,
        "theoretical_distinct_examples": expected_distinct,
        "best_config": best,
        "all_results": results,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print("\n" + "=" * 60)
    print("BEST BAGGING CHECKPOINT")
    print("=" * 60)
    print(json.dumps(best, indent=2))
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
