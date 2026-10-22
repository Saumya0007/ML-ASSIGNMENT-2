
from pathlib import Path
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# Q3(b)-(c): Bagging bias-variance and depth sensitivity
# ============================================================
SEED = 26007
N_RESAMPLES = 30
N_TREES = 100
MIN_SAMPLES_LEAF = 4

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
SPLIT_FILE = RESULTS / "split.npz"
CHECKPOINT_FILE = RESULTS / "q3b_checkpoint.npz"

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

if not SPLIT_FILE.exists():
    raise FileNotFoundError(
        f"Missing {SPLIT_FILE}. Run src/data_setup.py first."
    )

# ------------------------------------------------------------
# Load the saved split indices and recreate the original dataset
# ------------------------------------------------------------
split = np.load(SPLIT_FILE)

required = {"train_indices", "val_indices", "test_indices"}
if not required.issubset(split.files):
    raise KeyError(
        f"Expected split indices {sorted(required)}; "
        f"found {split.files}."
    )

faces = fetch_olivetti_faces(shuffle=False)
X = faces.data.astype(np.float32)
y = faces.target.astype(int)

train_idx = split["train_indices"].astype(int)
val_idx = split["val_indices"].astype(int)
test_idx = split["test_indices"].astype(int)

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Sanity checks: the assignment split should contain 6/2/2
# images per identity for train/validation/test.
for name, labels, expected in [
    ("training", y_train, 6),
    ("validation", y_val, 2),
    ("test", y_test, 2),
]:
    counts = np.unique(labels, return_counts=True)[1]
    if len(counts) != 40 or not np.all(counts == expected):
        raise ValueError(
            f"Unexpected {name} split: expected 40 identities "
            f"with {expected} images each."
        )

classes = np.sort(np.unique(y_train))
n_classes = len(classes)
n_test = len(y_test)

if not np.array_equal(classes, np.arange(n_classes)):
    raise ValueError("Expected class labels 0, 1, ..., K-1.")

print(
    f"Data loaded: train={len(y_train)}, "
    f"validation={len(y_val)}, test={len(y_test)}"
)


# ------------------------------------------------------------
# Determine the effective depth of the Q1-selected tree
# ------------------------------------------------------------
selected_tree = DecisionTreeClassifier(
    max_depth=None,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=SEED,
)
selected_tree.fit(X_train, y_train)

selected_effective_depth = int(selected_tree.tree_.max_depth)

# The selected Q1 hyperparameter was max_depth=None. Because this
# has no finite numeric value, use its observed fitted depth as the
# middle comparison cap and document this interpretation.
shallower_depth = max(1, selected_effective_depth // 2)
deeper_depth = max(
    selected_effective_depth + 5,
    selected_effective_depth * 2,
)

depth_values = [
    shallower_depth,
    selected_effective_depth,
    deeper_depth,
]

if len(set(depth_values)) != 3:
    raise ValueError(
        f"Could not construct three distinct depth caps: {depth_values}"
    )

CONFIGS = [
    {"label": "shallower", "max_depth": depth_values[0]},
    {
        "label": "selected_effective_depth",
        "max_depth": depth_values[1],
    },
    {"label": "much_deeper", "max_depth": depth_values[2]},
]
labels = [config["label"] for config in CONFIGS]

print(f"Q1 selected hyperparameter: max_depth=None")
print(f"Q1 selected tree's observed depth: {selected_effective_depth}")
print("Q3(b) depth caps:")
for config in CONFIGS:
    print(f"  {config['label']}: {config['max_depth']}")
print(f"Trees per ensemble: {N_TREES}")
print(f"Resamples: {N_RESAMPLES}")


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def one_hot(labels):
    out = np.zeros((len(labels), n_classes), dtype=np.float32)
    out[np.arange(len(labels)), labels.astype(int)] = 1.0
    return out


def atomic_checkpoint(done):
    """Write a checkpoint safely after each completed resample."""
    temp_file = CHECKPOINT_FILE.with_suffix(".tmp.npz")

    with temp_file.open("wb") as f:
        np.savez_compressed(
            f,
            predictions=predictions,
            mean_leaf_count=mean_leaf_count,
            singleton_leaf_fraction=singleton_leaf_fraction,
            mean_leaf_occupancy=mean_leaf_occupancy,
            tree_disagreement=tree_disagreement,
            completed=np.array(done),
            n_trees=np.array(N_TREES),
            n_resamples=np.array(N_RESAMPLES),
            selected_effective_depth=np.array(
                selected_effective_depth
            ),
            depth_values=np.asarray(depth_values),
        )

    os.replace(temp_file, CHECKPOINT_FILE)


# ------------------------------------------------------------
# Generate the 30 fixed identity-stratified resamples
# ------------------------------------------------------------
identity_positions = {
    identity: np.flatnonzero(y_train == identity)
    for identity in classes
}

resample_rng = np.random.default_rng(SEED + 1)
resample_indices = []

for _ in range(N_RESAMPLES):
    positions = []

    for identity in classes:
        candidates = identity_positions[identity]

        if len(candidates) != 6:
            raise ValueError(
                f"Identity {identity} has {len(candidates)} "
                "training images; expected 6."
            )

        positions.extend(
            resample_rng.choice(candidates, size=6, replace=True)
        )

    resample_indices.append(np.asarray(positions, dtype=int))


# ------------------------------------------------------------
# Initialize arrays or resume a compatible checkpoint
# ------------------------------------------------------------
prediction_shape = (
    len(CONFIGS),
    N_RESAMPLES,
    n_test,
    n_classes,
)
diagnostic_shape = (len(CONFIGS), N_RESAMPLES)

predictions = np.zeros(prediction_shape, dtype=np.float32)
mean_leaf_count = np.full(diagnostic_shape, np.nan)
singleton_leaf_fraction = np.full(diagnostic_shape, np.nan)
mean_leaf_occupancy = np.full(diagnostic_shape, np.nan)
tree_disagreement = np.full(diagnostic_shape, np.nan)
completed = 0

if CHECKPOINT_FILE.exists():
    try:
        with np.load(CHECKPOINT_FILE) as saved:
            compatible = (
                int(saved["n_trees"]) == N_TREES
                and int(saved["n_resamples"]) == N_RESAMPLES
                and int(saved["selected_effective_depth"])
                    == selected_effective_depth
                and np.array_equal(
                    saved["depth_values"], depth_values
                )
                and saved["predictions"].shape == prediction_shape
            )

            if compatible:
                predictions = saved["predictions"].copy()
                mean_leaf_count = saved["mean_leaf_count"].copy()
                singleton_leaf_fraction = saved[
                    "singleton_leaf_fraction"
                ].copy()
                mean_leaf_occupancy = saved[
                    "mean_leaf_occupancy"
                ].copy()
                tree_disagreement = saved[
                    "tree_disagreement"
                ].copy()
                completed = int(saved["completed"])
            else:
                print(
                    "Existing checkpoint settings do not match; "
                    "starting a fresh Q3(b) run."
                )

    except (OSError, KeyError, ValueError) as exc:
        print(f"Could not load checkpoint ({exc}); starting fresh.")

print(f"Starting at resample {completed + 1}/{N_RESAMPLES}.")


# ------------------------------------------------------------
# Train bagging ensembles and save progress
# ------------------------------------------------------------
for r in range(completed, N_RESAMPLES):
    idx = resample_indices[r]
    X_r = X_train[idx]
    y_r = y_train[idx]

    for ci, config in enumerate(CONFIGS):
        rng = np.random.default_rng(
            SEED + 10000 * r + 100
        )

        vote_counts = np.zeros(
            (n_test, n_classes), dtype=np.float32
        )
        individual_predictions = np.zeros(
            (N_TREES, n_test), dtype=np.int32
        )

        leaf_counts = []
        singleton_fractions = []
        leaf_occupancies = []

        for t in range(N_TREES):
            bootstrap_idx = rng.integers(
                0, len(X_r), size=len(X_r)
            )

            tree = DecisionTreeClassifier(
                max_depth=config["max_depth"],
                min_samples_leaf=MIN_SAMPLES_LEAF,
                random_state=SEED + t,
            )
            tree.fit(X_r[bootstrap_idx], y_r[bootstrap_idx])

            pred = tree.predict(X_test).astype(int)
            individual_predictions[t] = pred
            vote_counts[np.arange(n_test), pred] += 1

            # Leaf occupancy diagnostics.
            is_leaf = tree.tree_.children_left == -1
            leaf_samples = tree.tree_.n_node_samples[is_leaf]

            leaf_counts.append(len(leaf_samples))
            singleton_fractions.append(
                float(np.mean(leaf_samples == 1))
                if len(leaf_samples) else 0.0
            )
            leaf_occupancies.append(
                float(np.mean(leaf_samples))
                if len(leaf_samples) else 0.0
            )

        predictions[ci, r] = vote_counts / N_TREES

        mean_leaf_count[ci, r] = np.mean(leaf_counts)
        singleton_leaf_fraction[ci, r] = np.mean(
            singleton_fractions
        )
        mean_leaf_occupancy[ci, r] = np.mean(
            leaf_occupancies
        )

        # Expected pairwise disagreement between two trees:
        # 1 - sum of squared class prediction fractions.
        class_fractions = np.zeros(
            (n_test, n_classes), dtype=np.float32
        )

        for k in range(n_classes):
            class_fractions[:, k] = np.mean(
                individual_predictions == k, axis=0
            )

        tree_disagreement[ci, r] = np.mean(
            1.0 - np.sum(class_fractions ** 2, axis=1)
        )

    completed = r + 1
    atomic_checkpoint(completed)
    print(
        f"Completed resample {completed}/{N_RESAMPLES}; "
        "checkpoint saved."
    )


# ------------------------------------------------------------
# Bias squared, variance, and test accuracy
# ------------------------------------------------------------
target = one_hot(y_test)
summary = {}
accuracy_rows = []

for ci, config in enumerate(CONFIGS):
    pred_vectors = predictions[ci]
    mean_prediction = pred_vectors.mean(axis=0)

    # Squared Euclidean distance between mean vote vector
    # and one-hot true label, averaged across test images.
    bias_squared = float(np.mean(np.sum(
        (mean_prediction - target) ** 2, axis=1
    )))

    # Average squared Euclidean deviation from the mean
    # prediction vector across resamples and test images.
    variance = float(np.mean(np.sum(
        (pred_vectors - mean_prediction[None, :, :]) ** 2,
        axis=2,
    )))

    accuracies = np.array([
        np.mean(
            np.argmax(pred_vectors[r], axis=1) == y_test
        )
        for r in range(N_RESAMPLES)
    ])

    summary[config["label"]] = {
        "max_depth": config["max_depth"],
        "depth_interpretation": config["label"],
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "n_trees": N_TREES,
        "n_resamples": N_RESAMPLES,
        "bias_squared": bias_squared,
        "variance": variance,
        "mean_test_accuracy": float(accuracies.mean()),
        "std_test_accuracy": float(accuracies.std()),
        "mean_leaf_count_per_tree": float(
            np.nanmean(mean_leaf_count[ci])
        ),
        "mean_singleton_leaf_fraction": float(
            np.nanmean(singleton_leaf_fraction[ci])
        ),
        "mean_samples_per_leaf": float(
            np.nanmean(mean_leaf_occupancy[ci])
        ),
        "mean_tree_disagreement": float(
            np.nanmean(tree_disagreement[ci])
        ),
    }

    for r, accuracy in enumerate(accuracies):
        accuracy_rows.append({
            "depth_setting": config["label"],
            "max_depth": config["max_depth"],
            "resample": r + 1,
            "test_accuracy": float(accuracy),
            "mean_leaf_count_per_tree": float(
                mean_leaf_count[ci, r]
            ),
            "singleton_leaf_fraction": float(
                singleton_leaf_fraction[ci, r]
            ),
            "mean_samples_per_leaf": float(
                mean_leaf_occupancy[ci, r]
            ),
            "tree_disagreement": float(
                tree_disagreement[ci, r]
            ),
        })


# ------------------------------------------------------------
# Save JSON and CSV
# ------------------------------------------------------------
output = {
    "seed": SEED,
    "n_resamples": N_RESAMPLES,
    "n_trees": N_TREES,
    "test_set_fixed": True,
    "resampling": "six draws with replacement per identity",
    "selected_q1_max_depth_parameter": None,
    "selected_tree_observed_depth": selected_effective_depth,
    "depth_caps": {
        config["label"]: config["max_depth"]
        for config in CONFIGS
    },
    "depth_interpretation": (
        "Q1 selected max_depth=None. The observed depth of that "
        "tree fitted on the original training split is used as "
        "the middle depth cap. The other settings use a shallower "
        "and a much deeper cap."
    ),
    "bias_variance_definition": (
        "Bias squared is mean squared Euclidean distance between "
        "the mean normalized vote vector and one-hot target. "
        "Variance is mean squared Euclidean deviation of each "
        "resample's vote vector from the per-test-example mean."
    ),
    "tree_disagreement_definition": (
        "Mean over fixed test images of one minus the sum of "
        "squared class prediction fractions across individual trees."
    ),
    "summary": summary,
    "per_resample": accuracy_rows,
}

json_path = RESULTS / "bagging_depth_bias_variance.json"
with json_path.open("w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

csv_path = RESULTS / "bagging_depth_bias_variance.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(accuracy_rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(accuracy_rows)


# ------------------------------------------------------------
# Figure for Q3(b)-(c)
# ------------------------------------------------------------
plot_metrics = [
    ("bias_squared", "Squared bias"),
    ("variance", "Prediction variance"),
    ("mean_test_accuracy", "Mean test accuracy"),
    ("mean_leaf_count_per_tree", "Mean leaves per tree"),
    ("mean_singleton_leaf_fraction", "Singleton-leaf fraction"),
    ("mean_tree_disagreement", "Tree disagreement"),
]

fig, axes = plt.subplots(2, 3, figsize=(13, 7))

for ax, (metric, title) in zip(axes.flat, plot_metrics):
    values = [summary[label][metric] for label in labels]
    ax.bar(labels, values)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=18)
    ax.grid(axis="y", alpha=0.25)

fig.suptitle("Q3(b)-(c): Bagging depth sensitivity (100 trees)")
fig.tight_layout()

figure_path = FIGURES / "q3b_depth_bias_variance.png"
fig.savefig(figure_path, dpi=200, bbox_inches="tight")
plt.close(fig)


# ------------------------------------------------------------
# Print final summary
# ------------------------------------------------------------
print("\n=== Q3(b)-(c): Bagging depth comparison ===")

for label, values in summary.items():
    print(
        f"{label:24s} "
        f"depth={values['max_depth']:>3} "
        f"bias^2={values['bias_squared']:.6f} "
        f"variance={values['variance']:.6f} "
        f"accuracy={values['mean_test_accuracy']:.4f} "
        f"leaves={values['mean_leaf_count_per_tree']:.2f} "
        f"disagreement={values['mean_tree_disagreement']:.4f}"
    )

print(f"\nSaved: {json_path}")
print(f"Saved: {csv_path}")
print(f"Saved: {figure_path}")
print(f"Checkpoint retained at: {CHECKPOINT_FILE}")
