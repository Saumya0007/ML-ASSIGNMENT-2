
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

SEED = 26007
N_TREES = 200
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 4

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load dataset and fixed split
# --------------------------------------------------
dataset = fetch_olivetti_faces(shuffle=False)
images = dataset.images
X = images.reshape(len(images), -1)
y = dataset.target

split = np.load(RESULTS_DIR / "split.npz")
train_idx = split["train_indices"]
test_idx = split["test_indices"]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]
classes = np.unique(y_train)

# --------------------------------------------------
# Train the frozen single tree
# --------------------------------------------------
single_tree = DecisionTreeClassifier(
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=SEED,
)
single_tree.fit(X_train, y_train)

single_pred = single_tree.predict(X_test)

# Leaf confidence: fraction of samples in the predicted leaf
# belonging to the predicted class.
leaf_ids = single_tree.apply(X_test)
leaf_confidence = []

for j, leaf_id in enumerate(leaf_ids):
    node_class_counts = single_tree.tree_.value[
        leaf_id, 0, :
    ]
    total = node_class_counts.sum()
    predicted_class_index = np.where(
        single_tree.classes_ == single_pred[j]
    )[0][0]

    confidence = (
        node_class_counts[predicted_class_index] / total
        if total > 0 else 0.0
    )
    leaf_confidence.append(float(confidence))

leaf_confidence = np.asarray(leaf_confidence)

# --------------------------------------------------
# Reproduce the sequential bootstrap ensemble
# --------------------------------------------------
rng = np.random.default_rng(SEED)
tree_predictions = []

for i in range(N_TREES):
    bootstrap_idx = rng.integers(
        0, len(X_train), size=len(X_train)
    )

    tree = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=SEED + i,
    )
    tree.fit(X_train[bootstrap_idx], y_train[bootstrap_idx])
    tree_predictions.append(tree.predict(X_test))

tree_predictions = np.asarray(tree_predictions)

# --------------------------------------------------
# Majority votes and vote fractions
# --------------------------------------------------
vote_counts = np.zeros(
    (len(X_test), len(classes)), dtype=int
)

for j in range(len(X_test)):
    for k, class_id in enumerate(classes):
        vote_counts[j, k] = np.sum(
            tree_predictions[:, j] == class_id
        )

winner_indices = np.argmax(vote_counts, axis=1)
bagged_pred = classes[winner_indices]

bagging_confidence = (
    vote_counts[np.arange(len(X_test)), winner_indices]
    / N_TREES
)

# --------------------------------------------------
# Identify rescued, regressed, and shared errors
# --------------------------------------------------
single_correct = single_pred == y_test
bagged_correct = bagged_pred == y_test

rescued = (~single_correct) & bagged_correct
regressed = single_correct & (~bagged_correct)
both_wrong = (~single_correct) & (~bagged_correct)
both_correct = single_correct & bagged_correct

# Pick two rescued and two both-wrong examples.
rescued_indices = np.flatnonzero(rescued)[:2]
both_wrong_indices = np.flatnonzero(both_wrong)[:2]

if len(rescued_indices) < 2 or len(both_wrong_indices) < 2:
    raise RuntimeError(
        "Could not find two examples in each requested category."
    )

selected = [
    ("Corrected by bagging", int(j))
    for j in rescued_indices
] + [
    ("Both models wrong", int(j))
    for j in both_wrong_indices
]

# --------------------------------------------------
# Confidence summaries on the clean test set
# --------------------------------------------------
summary = {
    "seed": SEED,
    "n_trees": N_TREES,
    "max_depth": MAX_DEPTH,
    "min_samples_leaf": MIN_SAMPLES_LEAF,
    "single_tree_clean_test_accuracy": float(
        accuracy_score(y_test, single_pred)
    ),
    "bagging_clean_test_accuracy": float(
        accuracy_score(y_test, bagged_pred)
    ),
    "single_tree_mean_leaf_confidence": float(
        leaf_confidence.mean()
    ),
    "bagging_mean_winning_vote_fraction": float(
        bagging_confidence.mean()
    ),
    "rescued_by_bagging_count": int(rescued.sum()),
    "regressed_under_bagging_count": int(regressed.sum()),
    "both_wrong_count": int(both_wrong.sum()),
    "both_correct_count": int(both_correct.sum()),
    "rescued_test_positions": np.flatnonzero(rescued).tolist(),
    "regressed_test_positions": np.flatnonzero(regressed).tolist(),
    "both_wrong_test_positions": np.flatnonzero(both_wrong).tolist(),
}

# --------------------------------------------------
# Detailed examples with all class vote counts
# --------------------------------------------------
examples = []

for category, j in selected:
    votes_for_face = {
        str(int(class_id)): int(vote_counts[j, k])
        for k, class_id in enumerate(classes)
        if vote_counts[j, k] > 0
    }

    examples.append({
        "category": category,
        "test_position": j,
        "dataset_index": int(test_idx[j]),
        "true_identity": int(y_test[j]),
        "single_tree_prediction": int(single_pred[j]),
        "single_tree_leaf_confidence": float(leaf_confidence[j]),
        "bagging_prediction": int(bagged_pred[j]),
        "bagging_winning_vote_count": int(
            vote_counts[j, winner_indices[j]]
        ),
        "bagging_winning_vote_fraction": float(
            bagging_confidence[j]
        ),
        "bagging_vote_counts_by_identity": votes_for_face,
        "single_tree_correct": bool(single_correct[j]),
        "bagging_correct": bool(bagged_correct[j]),
    })

summary["selected_examples"] = examples

# --------------------------------------------------
# Save JSON summary and detailed CSV
# --------------------------------------------------
with (RESULTS_DIR / "error_analysis.json").open(
    "w", encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

import csv

csv_path = RESULTS_DIR / "q1d_face_examples.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
     fieldnames=[
    "category",
    "test_position",
    "dataset_index",
    "true_identity",
    "single_tree_prediction",
    "single_tree_leaf_confidence",
    "bagging_prediction",
    "bagging_winning_vote_count",
    "bagging_winning_vote_fraction",
    "bagging_vote_counts_by_identity",
    "single_tree_correct",
    "bagging_correct",
],
    )
    writer.writeheader()

    for example in examples:
        row = example.copy()
        row["bagging_vote_counts_by_identity"] = json.dumps(
            row["bagging_vote_counts_by_identity"]
        )
        writer.writerow(row)

# --------------------------------------------------
# Plot the four selected faces
# --------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(12, 4))

for ax, (category, j) in zip(axes, selected):
    ax.imshow(images[test_idx[j]], cmap="gray")
    ax.set_title(
        f"{category}\n"
        f"True: {y_test[j]}\n"
        f"Tree: {single_pred[j]} "
        f"({leaf_confidence[j]:.2f})\n"
        f"Bag: {bagged_pred[j]} "
        f"({bagging_confidence[j]:.2f})",
        fontsize=8,
    )
    ax.axis("off")

fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "q1d_four_face_examples.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

print(json.dumps(summary, indent=2))
print(f"\nSaved: {RESULTS_DIR / 'error_analysis.json'}")
print(f"Saved: {csv_path}")
print(f"Saved: {FIGURES_DIR / 'q1d_four_face_examples.png'}")
