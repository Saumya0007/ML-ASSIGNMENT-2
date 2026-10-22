
from pathlib import Path
import csv
import json

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

SEED = 26007
N_RESAMPLES = 30
N_BAGGED_TREES = 175
N_BOOSTING_ROUNDS = 100
BOOSTING_DEPTH = 4

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# --------------------------------------------------
# Dataset and fixed split
# --------------------------------------------------
data = fetch_olivetti_faces()
images = data.images
X = images.reshape(len(images), -1)
y = data.target

split = np.load(RESULTS / "split.npz")
train_idx = split["train_indices"]
test_idx = split["test_indices"]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

classes = np.arange(40)
n_classes = len(classes)
n_test = len(y_test)

# Index training examples by identity.
identity_train_positions = {
    c: np.flatnonzero(y_train == c) for c in classes
}

# --------------------------------------------------
# Prediction vectors
# --------------------------------------------------
def one_hot(labels):
    result = np.zeros((len(labels), n_classes))
    result[np.arange(len(labels)), labels.astype(int)] = 1.0
    return result


def tree_probability_vector(tree, X_input):
    """One-hot hard prediction for a decision tree."""
    return one_hot(tree.predict(X_input))


def train_bagging(X_fit, y_fit, rng, n_trees=N_BAGGED_TREES,
                  max_depth=None, min_samples_leaf=4):
    """Student-implemented bootstrap bagging."""
    votes = np.zeros((len(X_test), n_classes))

    for t in range(n_trees):
        bootstrap_idx = rng.integers(0, len(X_fit), size=len(X_fit))
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=SEED + t,
        )
        tree.fit(X_fit[bootstrap_idx], y_fit[bootstrap_idx])
        pred = tree.predict(X_test)
        votes[np.arange(len(X_test)), pred.astype(int)] += 1

    return votes / n_trees


def train_samme(X_fit, y_fit, max_depth=BOOSTING_DEPTH,
                n_rounds=N_BOOSTING_ROUNDS):
    """Scratch multiclass SAMME, returning normalized vote vectors."""
    n = len(X_fit)
    weights = np.full(n, 1.0 / n)
    learners = []
    alphas = []
    rng = np.random.default_rng(SEED)

    for t in range(n_rounds):
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=1,
            random_state=SEED + t,
        )
        tree.fit(X_fit, y_fit, sample_weight=weights)
        pred = tree.predict(X_fit)
        incorrect = pred != y_fit
        error = float(np.sum(weights * incorrect))

        if error >= 1.0 - 1.0 / n_classes:
            break

        error = np.clip(error, 1e-12, 1 - 1e-12)
        alpha = np.log((1 - error) / error) + np.log(n_classes - 1)

        weights *= np.exp(alpha * incorrect)
        weights /= weights.sum()

        learners.append(tree)
        alphas.append(float(alpha))

    if not learners:
        raise RuntimeError("SAMME accepted no weak learners.")

    scores = np.zeros((len(X_test), n_classes))
    for tree, alpha in zip(learners, alphas):
        pred = tree.predict(X_test)
        scores[np.arange(len(X_test)), pred.astype(int)] += alpha

    # The assignment requests a normalized weighted vote vector.
    row_sums = scores.sum(axis=1, keepdims=True)
    return scores / np.maximum(row_sums, 1e-12)


# --------------------------------------------------
# Resampling experiment
# --------------------------------------------------
rng = np.random.default_rng(SEED + 1)

prediction_vectors = {
    "single_tree": [],
    "bagging": [],
    "boosting": [],
}

resample_accuracies = []

for r in range(N_RESAMPLES):
    resampled_indices = []

    # Six draws with replacement from each identity's six
    # original training images.
    for identity in classes:
        positions = identity_train_positions[identity]
        selected = rng.choice(positions, size=6, replace=True)
        resampled_indices.extend(selected)

    resampled_indices = np.asarray(resampled_indices, dtype=int)
    X_r, y_r = X_train[resampled_indices], y_train[resampled_indices]

    # Selected single decision tree.
    single = DecisionTreeClassifier(
        max_depth=None,
        min_samples_leaf=4,
        random_state=SEED,
    )
    single.fit(X_r, y_r)
    single_vec = tree_probability_vector(single, X_test)

    # Selected bagging model.
    bag_rng = np.random.default_rng(SEED + 1000 + r)
    bag_vec = train_bagging(X_r, y_r, bag_rng)

    # Selected SAMME model.
    boost_vec = train_samme(X_r, y_r)

    prediction_vectors["single_tree"].append(single_vec)
    prediction_vectors["bagging"].append(bag_vec)
    prediction_vectors["boosting"].append(boost_vec)

    resample_accuracies.append({
        "resample": r + 1,
        "single_tree_accuracy": float(
            np.mean(single.predict(X_test) == y_test)
        ),
        "bagging_accuracy": float(
            np.mean(np.argmax(bag_vec, axis=1) == y_test)
        ),
        "boosting_accuracy": float(
            np.mean(np.argmax(boost_vec, axis=1) == y_test)
        ),
    })

    print(f"Completed resample {r + 1}/{N_RESAMPLES}")


# --------------------------------------------------
# Bias squared and variance
# --------------------------------------------------
target_vectors = one_hot(y_test)
summary = {}

for model_name, vectors in prediction_vectors.items():
    # Shape: (resamples, test examples, classes)
    predictions = np.asarray(vectors)
    mean_prediction = predictions.mean(axis=0)

    # Mean squared distance from average prediction to one-hot target.
    bias_squared = float(
        np.mean(np.sum((mean_prediction - target_vectors) ** 2, axis=1))
    )

    # Average squared distance of each resample prediction
    # from the per-test-example mean prediction.
    variance = float(
        np.mean(
            np.sum(
                (predictions - mean_prediction[None, :, :]) ** 2,
                axis=2,
            )
        )
    )

    summary[model_name] = {
        "bias_squared": bias_squared,
        "variance": variance,
        "mean_test_accuracy": float(np.mean([
            row[f"{model_name}_accuracy"]
            for row in resample_accuracies
        ])),
        "std_test_accuracy": float(np.std([
            row[f"{model_name}_accuracy"]
            for row in resample_accuracies
        ])),
    }

output = {
    "seed": SEED,
    "n_resamples": N_RESAMPLES,
    "test_set_fixed": True,
    "resampling": "six draws with replacement per identity",
    "single_tree": {
        "max_depth": None,
        "min_samples_leaf": 4,
    },
    "bagging": {
        "n_estimators": N_BAGGED_TREES,
        "max_depth": None,
        "min_samples_leaf": 4,
    },
    "boosting": {
        "max_depth": BOOSTING_DEPTH,
        "n_estimators_requested": N_BOOSTING_ROUNDS,
    },
    "summary": summary,
    "per_resample_accuracy": resample_accuracies,
}

with (RESULTS / "bias_variance_exact.json").open(
    "w", encoding="utf-8"
) as f:
    json.dump(output, f, indent=2)

with (RESULTS / "bias_variance_exact.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    fields = [
        "resample",
        "single_tree_accuracy",
        "bagging_accuracy",
        "boosting_accuracy",
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(resample_accuracies)

print("\n=== Q3(a) bias–variance summary ===")
for name, metrics in summary.items():
    print(
        f"{name:12s} "
        f"bias_squared={metrics['bias_squared']:.6f}, "
        f"variance={metrics['variance']:.6f}, "
        f"mean_accuracy={metrics['mean_test_accuracy']:.4f}, "
        f"std_accuracy={metrics['std_test_accuracy']:.4f}"
    )

print("\nSaved results/bias_variance_exact.json")
print("Saved results/bias_variance_exact.csv")
