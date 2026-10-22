
from pathlib import Path
import csv
import json
import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# ============================================================
# Q2(c) and Q2(d): Bagging vs. SAMME boosting
# ============================================================
SEED = 26007
CHECKPOINTS = [25, 50, 100]
BAGGING_FINAL_TREES = 175
BAGGING_MAX_TREES = 200
BOOSTING_FINAL_ROUNDS = 100
BOOSTING_DEPTH = 4

# Preserve the original scripts' different base-tree settings.
BAGGING_MIN_LEAF = 4
BOOSTING_MIN_LEAF = 1

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def majority_vote(prediction_matrix, classes):
    """
    prediction_matrix shape: (number_of_learners, number_of_samples).
    Ties are resolved by np.argmax over sorted class labels.
    """
    counts = np.stack(
        [(prediction_matrix == cls).sum(axis=0) for cls in classes],
        axis=0,
    )
    winner_positions = np.argmax(counts, axis=0)
    vote_fractions = counts / prediction_matrix.shape[0]
    return classes[winner_positions], vote_fractions.T


def masked_test_images(images):
    """Zero rows/columns 24:39 inclusive of each test image."""
    masked = images.copy()
    masked[:, 24:40, 24:40] = 0.0
    return masked.reshape(len(masked), -1)


# ============================================================
# Load dataset and the one saved split
# ============================================================
print("Loading Olivetti Faces dataset...")
data = fetch_olivetti_faces(shuffle=False)

images = data.images
X = images.reshape(len(images), -1).astype(np.float64)
y = data.target.astype(int)

split_path = RESULTS / "split.npz"
if not split_path.exists():
    raise FileNotFoundError(
        "results/split.npz is missing. Run src/data_setup.py first."
    )

split = np.load(split_path)
train_idx = split["train_indices"]
val_idx = split["val_indices"]
test_idx = split["test_indices"]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

X_test_masked = masked_test_images(images[test_idx])

classes = np.unique(y_train)
n_classes = len(classes)

print(f"Train: {len(y_train)}, validation: {len(y_val)}, test: {len(y_test)}")
print(f"Classes: {n_classes}")
print(f"Seed: {SEED}")


# ============================================================
# Q2(c): Bagging, matching src/bagging.py
# ============================================================
print("\nTraining bagging ensemble...")

# This must match the original script: one continuous RNG sequence.
bootstrap_rng = np.random.default_rng(SEED)

bagging_models = []
bagging_train_predictions = []
bagging_val_predictions = []
bagging_test_predictions = []
bagging_masked_predictions = []

bagging_rows = []
bagging_checkpoints = {}
bagging_cumulative_seconds = {}

bagging_start = time.perf_counter()
n_train = len(y_train)

for tree_idx in range(BAGGING_MAX_TREES):
    # Same bootstrap sampling convention as the original bagging.py.
    bootstrap_indices = bootstrap_rng.choice(
        n_train, size=n_train, replace=True
    )

    tree = DecisionTreeClassifier(
        max_depth=None,
        min_samples_leaf=BAGGING_MIN_LEAF,
        random_state=SEED + tree_idx,
    )

    tree.fit(X_train[bootstrap_indices], y_train[bootstrap_indices])
    bagging_models.append(tree)

    # Cache each tree's predictions to make checkpoint voting consistent.
    bagging_train_predictions.append(tree.predict(X_train))
    bagging_val_predictions.append(tree.predict(X_val))
    bagging_test_predictions.append(tree.predict(X_test))
    bagging_masked_predictions.append(tree.predict(X_test_masked))

    n_trees = tree_idx + 1
    elapsed = time.perf_counter() - bagging_start

    if n_trees in CHECKPOINTS or n_trees == BAGGING_FINAL_TREES:
        train_matrix = np.asarray(bagging_train_predictions)
        val_matrix = np.asarray(bagging_val_predictions)
        test_matrix = np.asarray(bagging_test_predictions)
        masked_matrix = np.asarray(bagging_masked_predictions)

        train_pred, _ = majority_vote(train_matrix, classes)
        val_pred, val_votes = majority_vote(val_matrix, classes)
        test_pred, test_votes = majority_vote(test_matrix, classes)
        masked_pred, masked_votes = majority_vote(masked_matrix, classes)

        bagging_checkpoints[n_trees] = {
            "train": train_pred,
            "val": val_pred,
            "test": test_pred,
            "masked": masked_pred,
            "test_vote_fractions": test_votes,
            "masked_vote_fractions": masked_votes,
        }
        bagging_cumulative_seconds[n_trees] = elapsed

        row = {
            "model": "Bagging",
            "learners": n_trees,
            "training_seconds": elapsed,
            "train_accuracy": accuracy(y_train, train_pred),
            "validation_accuracy": accuracy(y_val, val_pred),
            "clean_test_accuracy": accuracy(y_test, test_pred),
            "masked_test_accuracy": accuracy(y_test, masked_pred),
        }
        bagging_rows.append(row)

        print(
            f"Bagging {n_trees:3d}: "
            f"time={elapsed:.2f}s, "
            f"train={row['train_accuracy']:.4f}, "
            f"val={row['validation_accuracy']:.4f}, "
            f"clean={row['clean_test_accuracy']:.4f}, "
            f"masked={row['masked_test_accuracy']:.4f}"
        )

bagging_total_seconds = time.perf_counter() - bagging_start


# ============================================================
# Q2(c): Scratch multiclass SAMME, matching boosting_grid_search.py
# ============================================================
class ScratchSAMME:
    """
    Multiclass SAMME.

    Uses weighted classification error and:
        alpha = log((1-error)/error) + log(K-1)

    Weak learners whose error is no better than random guessing
    are rejected, matching the original boosting grid script.
    """

    def __init__(
        self,
        max_depth,
        n_estimators,
        seed=SEED,
        min_samples_leaf=BOOSTING_MIN_LEAF,
    ):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.seed = seed
        self.min_samples_leaf = min_samples_leaf

        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        self.round_training_seconds_ = []
        self.sample_weight_history_ = []
        self.training_prediction_history_ = []

        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes_local = len(self.classes_)
        sample_weights = np.full(len(y), 1.0 / len(y))

        for round_idx in range(self.n_estimators):
            round_start = time.perf_counter()

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.seed + round_idx,
            )
            tree.fit(X, y, sample_weight=sample_weights)

            predictions = tree.predict(X)
            incorrect = predictions != y

            error = float(np.sum(sample_weights * incorrect))
            random_error_threshold = 1.0 - 1.0 / n_classes_local

            if error >= random_error_threshold:
                print(
                    f"SAMME stopped at round {round_idx + 1}: "
                    f"weighted error {error:.6f} >= "
                    f"{random_error_threshold:.6f}"
                )
                break

            error = float(np.clip(error, 1e-12, 1.0 - 1e-12))
            alpha = np.log((1.0 - error) / error)
            alpha += np.log(n_classes_local - 1)

            sample_weights *= np.exp(alpha * incorrect)
            sample_weights /= sample_weights.sum()

            self.estimators_.append(tree)
            self.estimator_weights_.append(float(alpha))
            self.estimator_errors_.append(error)

            self.round_training_seconds_.append(
                time.perf_counter() - round_start
            )
            self.sample_weight_history_.append(sample_weights.copy())
            self.training_prediction_history_.append(predictions.copy())

        return self

    def decision_function(self, X, n_estimators=None):
        n = (
            len(self.estimators_)
            if n_estimators is None
            else min(n_estimators, len(self.estimators_))
        )

        if n == 0:
            raise RuntimeError("SAMME has no accepted weak learners.")

        scores = np.zeros((len(X), len(self.classes_)), dtype=float)

        for tree, alpha in zip(
            self.estimators_[:n],
            self.estimator_weights_[:n],
        ):
            pred = tree.predict(X)
            for j, cls in enumerate(self.classes_):
                scores[:, j] += alpha * (pred == cls)

        return scores

    def predict(self, X, n_estimators=None):
        scores = self.decision_function(X, n_estimators)
        return self.classes_[np.argmax(scores, axis=1)]

    def normalized_winning_scores(self, X, n_estimators=None):
        """
        Normalized weighted-vote scores, used as a confidence proxy.
        These are not calibrated probabilities.
        """
        scores = self.decision_function(X, n_estimators)
        total_weight = np.sum(
            self.estimator_weights_[:(
                len(self.estimators_)
                if n_estimators is None
                else min(n_estimators, len(self.estimators_))
            )]
        )
        if total_weight <= 0:
            return np.zeros(len(X))
        return np.max(scores, axis=1) / total_weight


print("\nTraining SAMME boosting...")
boost_start = time.perf_counter()

boost = ScratchSAMME(
    max_depth=BOOSTING_DEPTH,
    n_estimators=BOOSTING_FINAL_ROUNDS,
    seed=SEED,
    min_samples_leaf=BOOSTING_MIN_LEAF,
).fit(X_train, y_train)

boost_total_seconds = time.perf_counter() - boost_start

print(
    f"SAMME training time: {boost_total_seconds:.2f}s "
    f"for {len(boost.estimators_)} accepted rounds"
)


# ============================================================
# Evaluate SAMME at 25, 50, and 100 rounds
# ============================================================
boosting_rows = []
boosting_checkpoints = {}
boosting_cumulative_seconds = {}

for requested_rounds in CHECKPOINTS:
    if requested_rounds > len(boost.estimators_):
        print(
            f"SAMME {requested_rounds}: unavailable; "
            f"only {len(boost.estimators_)} rounds accepted"
        )
        continue

    elapsed = sum(boost.round_training_seconds_[:requested_rounds])
    boosting_cumulative_seconds[requested_rounds] = elapsed

    train_pred = boost.predict(X_train, requested_rounds)
    val_pred = boost.predict(X_val, requested_rounds)
    test_pred = boost.predict(X_test, requested_rounds)
    masked_pred = boost.predict(X_test_masked, requested_rounds)

    boosting_checkpoints[requested_rounds] = {
        "train": train_pred,
        "val": val_pred,
        "test": test_pred,
        "masked": masked_pred,
        "test_winning_score": boost.normalized_winning_scores(
            X_test, requested_rounds
        ),
        "masked_winning_score": boost.normalized_winning_scores(
            X_test_masked, requested_rounds
        ),
    }

    row = {
        "model": "SAMME",
        "learners": requested_rounds,
        "training_seconds": elapsed,
        "train_accuracy": accuracy(y_train, train_pred),
        "validation_accuracy": accuracy(y_val, val_pred),
        "clean_test_accuracy": accuracy(y_test, test_pred),
        "masked_test_accuracy": accuracy(y_test, masked_pred),
    }
    boosting_rows.append(row)

    print(
        f"SAMME   {requested_rounds:3d}: "
        f"time={elapsed:.2f}s, "
        f"train={row['train_accuracy']:.4f}, "
        f"val={row['validation_accuracy']:.4f}, "
        f"clean={row['clean_test_accuracy']:.4f}, "
        f"masked={row['masked_test_accuracy']:.4f}"
    )


# ============================================================
# Selected models for Q2(d)
# ============================================================
# The original saved grid search selects depth 4, 100 rounds.
# The original bagging results select 175 trees by validation.
bagging_selected = bagging_checkpoints[BAGGING_FINAL_TREES]

boost_selected_rounds = min(
    BOOSTING_FINAL_ROUNDS, len(boost.estimators_)
)
if boost_selected_rounds == 0:
    raise RuntimeError("No valid SAMME rounds were trained.")

bagging_selected_val = bagging_selected["val"]
bagging_selected_test = bagging_selected["test"]
bagging_selected_masked = bagging_selected["masked"]

boost_selected_val = boost.predict(X_val, boost_selected_rounds)
boost_selected_test = boost.predict(X_test, boost_selected_rounds)
boost_selected_masked = boost.predict(X_test_masked, boost_selected_rounds)

# Single-tree baseline selected by validation in Q1(a).
with (RESULTS / "tree_baseline.json").open("r", encoding="utf-8") as f:
    tree_config = json.load(f)["best_config"]

tree = DecisionTreeClassifier(
    max_depth=tree_config["max_depth"],
    min_samples_leaf=tree_config["min_samples_leaf"],
    random_state=SEED,
)
tree_start = time.perf_counter()
tree.fit(X_train, y_train)
tree_training_seconds = time.perf_counter() - tree_start

tree_val = tree.predict(X_val)
tree_test = tree.predict(X_test)
tree_masked = tree.predict(X_test_masked)

# Confidence proxy: fraction of training samples in the winning leaf.
# For a decision tree, this is a useful leaf-purity confidence measure.
def tree_leaf_confidence(model, X):
    leaf_ids = model.apply(X)
    leaf_confidence = {}
    for node_id in np.unique(leaf_ids):
        leaf_confidence[node_id] = float(
            np.max(model.tree_.value[node_id][0])
            / np.sum(model.tree_.value[node_id][0])
        )
    return np.array([leaf_confidence[node_id] for node_id in leaf_ids])


tree_test_conf = tree_leaf_confidence(tree, X_test)
tree_masked_conf = tree_leaf_confidence(tree, X_test_masked)

bagging_test_conf = np.max(
    bagging_selected["test_vote_fractions"], axis=1
)
bagging_masked_conf = np.max(
    bagging_selected["masked_vote_fractions"], axis=1
)

boost_test_conf = boost.normalized_winning_scores(
    X_test, boost_selected_rounds
)
boost_masked_conf = boost.normalized_winning_scores(
    X_test_masked, boost_selected_rounds
)

selected_model_rows = [
    {
        "model": "Decision tree",
        "configuration": (
            f"depth={tree_config['max_depth']}, "
            f"min_leaf={tree_config['min_samples_leaf']}"
        ),
        "training_seconds": tree_training_seconds,
        "validation_accuracy": accuracy(y_val, tree_val),
        "clean_test_accuracy": accuracy(y_test, tree_test),
        "masked_test_accuracy": accuracy(y_test, tree_masked),
        "mean_clean_winning_score": float(tree_test_conf.mean()),
        "mean_masked_winning_score": float(tree_masked_conf.mean()),
    },
    {
        "model": "Bagging",
        "configuration": (
            f"trees={BAGGING_FINAL_TREES}, "
            f"depth=None, min_leaf={BAGGING_MIN_LEAF}"
        ),
        "training_seconds": bagging_total_seconds,
        "validation_accuracy": accuracy(y_val, bagging_selected_val),
        "clean_test_accuracy": accuracy(y_test, bagging_selected_test),
        "masked_test_accuracy": accuracy(y_test, bagging_selected_masked),
        "mean_clean_winning_score": float(bagging_test_conf.mean()),
        "mean_masked_winning_score": float(bagging_masked_conf.mean()),
    },
    {
        "model": "SAMME",
        "configuration": (
            f"depth={BOOSTING_DEPTH}, rounds={boost_selected_rounds}, "
            f"min_leaf={BOOSTING_MIN_LEAF}"
        ),
        "training_seconds": boost_total_seconds,
        "validation_accuracy": accuracy(y_val, boost_selected_val),
        "clean_test_accuracy": accuracy(y_test, boost_selected_test),
        "masked_test_accuracy": accuracy(y_test, boost_selected_masked),
        "mean_clean_winning_score": float(boost_test_conf.mean()),
        "mean_masked_winning_score": float(boost_masked_conf.mean()),
    },
]


# ============================================================
# Per-image predictions and disagreement analysis
# ============================================================
disagreement_rows = []

for i in range(len(y_test)):
    true_label = int(y_test[i])

    disagreement_rows.append({
        "test_position": int(i),
        "original_dataset_index": int(test_idx[i]),
        "true_label": true_label,
        "tree_clean_prediction": int(tree_test[i]),
        "bagging_clean_prediction": int(bagging_selected_test[i]),
        "boosting_clean_prediction": int(boost_selected_test[i]),
        "tree_masked_prediction": int(tree_masked[i]),
        "bagging_masked_prediction": int(bagging_selected_masked[i]),
        "boosting_masked_prediction": int(boost_selected_masked[i]),
        "tree_correct_clean": bool(tree_test[i] == true_label),
        "bagging_correct_clean": bool(bagging_selected_test[i] == true_label),
        "boosting_correct_clean": bool(boost_selected_test[i] == true_label),
        "tree_correct_masked": bool(tree_masked[i] == true_label),
        "bagging_correct_masked": bool(
            bagging_selected_masked[i] == true_label
        ),
        "boosting_correct_masked": bool(
            boost_selected_masked[i] == true_label
        ),
        "bagging_boosting_disagree_clean": bool(
            bagging_selected_test[i] != boost_selected_test[i]
        ),
        "bagging_boosting_disagree_masked": bool(
            bagging_selected_masked[i] != boost_selected_masked[i]
        ),
    })

save_csv(RESULTS / "q2_prediction_disagreements.csv", disagreement_rows)

# Find example positions satisfying the Q1(d)/Q2(d) case categories.
rescued_positions = [
    i for i in range(len(y_test))
    if bagging_selected_test[i] == y_test[i]
    and tree_test[i] != y_test[i]
]
both_wrong_positions = [
    i for i in range(len(y_test))
    if tree_test[i] != y_test[i]
    and bagging_selected_test[i] != y_test[i]
]
different_mask_response_positions = [
    i for i in range(len(y_test))
    if (
        bagging_selected_test[i] == boost_selected_test[i]
        and bagging_selected_masked[i] != boost_selected_masked[i]
    )
    or (
        bagging_selected_test[i] != boost_selected_test[i]
        and bagging_selected_masked[i] == boost_selected_masked[i]
    )
]


# ============================================================
# SAMME prediction history: hardest examples / sample weights
# ============================================================
weight_history_rows = []

if boost.sample_weight_history_:
    final_weights = boost.sample_weight_history_[-1]
    highest_weight_indices = np.argsort(final_weights)[::-1][:10]

    for rank, train_position in enumerate(highest_weight_indices, start=1):
        weight_history_rows.append({
            "rank": rank,
            "training_position": int(train_position),
            "original_dataset_index": int(train_idx[train_position]),
            "true_label": int(y_train[train_position]),
            "final_sample_weight": float(final_weights[train_position]),
            "final_round_prediction": int(
                boost.training_prediction_history_[-1][train_position]
            ),
            "misclassified_in_final_round": bool(
                boost.training_prediction_history_[-1][train_position]
                != y_train[train_position]
            ),
        })

save_csv(RESULTS / "q2_samme_high_weight_examples.csv", weight_history_rows)


# ============================================================
# Save Q2(c) checkpoint table and Q2(d) summary
# ============================================================
dynamics_rows = bagging_rows + boosting_rows
save_csv(RESULTS / "q2c_dynamics.csv", dynamics_rows)
save_csv(RESULTS / "q2d_selected_models.csv", selected_model_rows)

summary = {
    "seed": SEED,
    "split_file": "results/split.npz",
    "bagging_config": {
        "n_estimators_selected": BAGGING_FINAL_TREES,
        "max_depth": None,
        "min_samples_leaf": BAGGING_MIN_LEAF,
        "training_seconds_for_200_trees": bagging_total_seconds,
    },
    "samme_config": {
        "max_depth": BOOSTING_DEPTH,
        "min_samples_leaf": BOOSTING_MIN_LEAF,
        "requested_rounds": BOOSTING_FINAL_ROUNDS,
        "actual_rounds": len(boost.estimators_),
        "training_seconds": boost_total_seconds,
    },
    "selected_model_metrics": selected_model_rows,
    "clean_prediction_disagreement_count": int(
        np.sum(bagging_selected_test != boost_selected_test)
    ),
    "masked_prediction_disagreement_count": int(
        np.sum(bagging_selected_masked != boost_selected_masked)
    ),
    "tree_correct_bagging_wrong_clean_positions": [
        i for i in range(len(y_test))
        if tree_test[i] == y_test[i]
        and bagging_selected_test[i] != y_test[i]
    ],
    "bagging_correct_tree_wrong_clean_positions": rescued_positions,
    "tree_and_bagging_both_wrong_clean_positions": both_wrong_positions,
    "different_mask_response_positions": different_mask_response_positions,
    "clean_confusion_matrix_tree": confusion_matrix(
        y_test, tree_test, labels=classes
    ).tolist(),
    "clean_confusion_matrix_bagging": confusion_matrix(
        y_test, bagging_selected_test, labels=classes
    ).tolist(),
    "clean_confusion_matrix_samme": confusion_matrix(
        y_test, boost_selected_test, labels=classes
    ).tolist(),
    "masked_confusion_matrix_tree": confusion_matrix(
        y_test, tree_masked, labels=classes
    ).tolist(),
    "masked_confusion_matrix_bagging": confusion_matrix(
        y_test, bagging_selected_masked, labels=classes
    ).tolist(),
    "masked_confusion_matrix_samme": confusion_matrix(
        y_test, boost_selected_masked, labels=classes
    ).tolist(),
}

with (RESULTS / "q2c_d_summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)


# ============================================================
# Q2(c) figures: accuracy and training time vs. learners
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for model_name in ["Bagging", "SAMME"]:
    rows = [row for row in dynamics_rows if row["model"] == model_name]
    if not rows:
        continue

    axes[0].plot(
        [row["learners"] for row in rows],
        [row["train_accuracy"] for row in rows],
        marker="o",
        label=model_name,
    )
    axes[1].plot(
        [row["learners"] for row in rows],
        [row["validation_accuracy"] for row in rows],
        marker="o",
        label=model_name,
    )
    axes[2].plot(
        [row["learners"] for row in rows],
        [row["training_seconds"] for row in rows],
        marker="o",
        label=model_name,
    )

axes[0].set_title("Training accuracy")
axes[1].set_title("Validation accuracy")
axes[2].set_title("Cumulative training time")

for ax in axes:
    ax.set_xlabel("Number of learners")
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[0].set_ylabel("Accuracy")
axes[1].set_ylabel("Accuracy")
axes[0].set_ylim(0, 1)
axes[1].set_ylim(0, 1)
axes[2].set_ylabel("Seconds")

fig.tight_layout()
fig.savefig(FIGURES / "q2c_learning_dynamics.png", dpi=200)
plt.close(fig)


# ============================================================
# Q2(d) figure: selected-model clean vs. masked accuracy
# ============================================================
model_names = [row["model"] for row in selected_model_rows]
clean_scores = [row["clean_test_accuracy"] for row in selected_model_rows]
masked_scores = [row["masked_test_accuracy"] for row in selected_model_rows]

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - width / 2, clean_scores, width, label="Clean test")
ax.bar(x + width / 2, masked_scores, width, label="Masked test")
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.set_title("Clean vs. Masked Test Accuracy")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "q2d_clean_vs_masked.png", dpi=200)
plt.close(fig)


# ============================================================
# Final console summary
# ============================================================
print("\nQ2(c): checkpoint comparison")
for row in dynamics_rows:
    print(
        f"{row['model']:8s} B/T={row['learners']:3d} "
        f"time={row['training_seconds']:.2f}s "
        f"train={row['train_accuracy']:.4f} "
        f"val={row['validation_accuracy']:.4f} "
        f"clean={row['clean_test_accuracy']:.4f} "
        f"masked={row['masked_test_accuracy']:.4f}"
    )

print("\nQ2(d): selected models")
for row in selected_model_rows:
    print(
        f"{row['model']:15s} "
        f"val={row['validation_accuracy']:.4f} "
        f"clean={row['clean_test_accuracy']:.4f} "
        f"masked={row['masked_test_accuracy']:.4f} "
        f"mean winning score clean="
        f"{row['mean_clean_winning_score']:.4f}, masked="
        f"{row['mean_masked_winning_score']:.4f}"
    )

print("\nSaved files:")
print("  results/q2c_dynamics.csv")
print("  results/q2c_d_summary.json")
print("  results/q2d_selected_models.csv")
print("  results/q2_prediction_disagreements.csv")
print("  results/q2_samme_high_weight_examples.csv")
print("  results/figures/q2c_learning_dynamics.png")
print("  results/figures/q2d_clean_vs_masked.png")
