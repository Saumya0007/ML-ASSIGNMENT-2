
from pathlib import Path
import csv
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SPLIT_FILE = RESULTS / "split.npz"
CHECKPOINT = RESULTS / "q3b_checkpoint.npz"

SEED = 26007
N_RESAMPLES = 30
N_TREES = 100
MIN_LEAF = 4

# Load the completed bagging experiment
with np.load(CHECKPOINT) as d:
    assert int(d["completed"]) == N_RESAMPLES
    assert int(d["n_trees"]) == N_TREES
    depths = d["depth_values"].astype(int)
    bagging_predictions = d["predictions"].copy()

# Load original split and dataset
split = np.load(SPLIT_FILE)
faces = fetch_olivetti_faces(shuffle=False)
X = faces.data.astype(np.float32)
y = faces.target.astype(int)

train_idx = split["train_indices"].astype(int)
test_idx = split["test_indices"].astype(int)

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

classes = np.sort(np.unique(y_train))
n_classes = len(classes)
n_test = len(y_test)

# Recreate the exact resamples from the original script
identity_positions = {
    c: np.flatnonzero(y_train == c) for c in classes
}
rng = np.random.default_rng(SEED + 1)
resamples = []

for _ in range(N_RESAMPLES):
    positions = []
    for c in classes:
        candidates = identity_positions[c]
        positions.extend(rng.choice(candidates, size=6, replace=True))
    resamples.append(np.asarray(positions, dtype=int))

# Fit only 3 x 30 = 90 single trees
single_predictions = np.zeros(
    (len(depths), N_RESAMPLES, n_test, n_classes),
    dtype=np.float32,
)

for r, idx in enumerate(resamples):
    for ci, depth in enumerate(depths):
        tree = DecisionTreeClassifier(
            max_depth=int(depth),
            min_samples_leaf=MIN_LEAF,
            random_state=SEED + 10000 * r,
        )
        tree.fit(X_train[idx], y_train[idx])
        pred = tree.predict(X_test).astype(int)
        single_predictions[ci, r, np.arange(n_test), pred] = 1.0
    print(f"Single-tree refits: {r + 1}/{N_RESAMPLES}")

target = np.eye(n_classes, dtype=np.float32)[y_test]


def metrics(predictions):
    mean_prediction = predictions.mean(axis=0)
    bias2 = np.mean(np.sum((mean_prediction - target) ** 2, axis=1))
    variance = np.mean(np.sum(
        (predictions - mean_prediction[None, :, :]) ** 2,
        axis=2,
    ))
    accuracies = np.array([
        np.mean(np.argmax(predictions[r], axis=1) == y_test)
        for r in range(N_RESAMPLES)
    ])
    return {
        "bias_squared": float(bias2),
        "variance": float(variance),
        "mean_test_accuracy": float(accuracies.mean()),
        "std_test_accuracy": float(accuracies.std(ddof=1)),
    }


rows = []
comparisons = []

for ci, depth in enumerate(depths):
    tree_result = metrics(single_predictions[ci])
    bag_result = metrics(bagging_predictions[ci])

    rows.append({
        "model": "single_tree",
        "max_depth": int(depth),
        **tree_result,
    })
    rows.append({
        "model": "bagging",
        "max_depth": int(depth),
        **bag_result,
    })

    reduction = tree_result["variance"] - bag_result["variance"]
    reduction_pct = (
        100 * reduction / tree_result["variance"]
        if tree_result["variance"] != 0 else float("nan")
    )

    comparisons.append({
        "max_depth": int(depth),
        "single_tree_accuracy": tree_result["mean_test_accuracy"],
        "bagging_accuracy": bag_result["mean_test_accuracy"],
        "accuracy_improvement": (
            bag_result["mean_test_accuracy"]
            - tree_result["mean_test_accuracy"]
        ),
        "single_tree_bias_squared": tree_result["bias_squared"],
        "bagging_bias_squared": bag_result["bias_squared"],
        "single_tree_variance": tree_result["variance"],
        "bagging_variance": bag_result["variance"],
        "variance_reduction": reduction,
        "variance_reduction_percent": reduction_pct,
    })

for filename, data in [
    ("q3_depth_single_and_bagging.csv", rows),
    ("q3_depth_bagging_improvements.csv", comparisons),
]:
    path = RESULTS / filename
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"Saved: {path}")

print("\nDEPTH-MATCHED COMPARISON")
for row in comparisons:
    print(
        f"Depth {row['max_depth']}: "
        f"accuracy gain = {100*row['accuracy_improvement']:.2f} "
        f"percentage points; "
        f"variance reduction = {row['variance_reduction']:.6f} "
        f"({row['variance_reduction_percent']:.2f}%)"
    )
