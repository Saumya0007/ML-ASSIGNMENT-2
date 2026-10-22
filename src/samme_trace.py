
from pathlib import Path
import json

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

SEED = 26007
N_SAMPLES = 10
N_ROUNDS = 3
MAX_DEPTH = 1

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load the same dataset and saved training split used by the project.
dataset = fetch_olivetti_faces(shuffle=False)
X = dataset.images.reshape(len(dataset.images), -1)
y = dataset.target

split = np.load(RESULTS_DIR / "split.npz")
train_idx = split["train_indices"]

X_train_all = X[train_idx]
y_train_all = y[train_idx]

# Select 10 deterministic training samples, spread across identities.
rng = np.random.default_rng(SEED)
selected_positions = []

for identity in np.unique(y_train_all):
    candidates = np.flatnonzero(y_train_all == identity)
    selected_positions.append(int(rng.choice(candidates)))

# Ten identities, ten examples total; no test data is used.
selected_positions = np.asarray(selected_positions[:N_SAMPLES])
X_trace = X_train_all[selected_positions]
y_trace = y_train_all[selected_positions]

classes = np.unique(y_train_all)
n_classes = len(classes)
n_samples = len(y_trace)

weights = np.full(n_samples, 1.0 / n_samples)
trace = []

print(f"Trace samples: {n_samples}")
print(f"Classes in full training set: {n_classes}")
print("Initial weights:", np.round(weights, 6).tolist())

for round_number in range(1, N_ROUNDS + 1):
    weights_before = weights.copy()

    learner = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        min_samples_leaf=1,
        random_state=SEED + round_number - 1,
    )
    learner.fit(X_trace, y_trace, sample_weight=weights_before)

    predictions = learner.predict(X_trace)
    incorrect = predictions != y_trace
    error = float(np.sum(weights_before * incorrect))

    # SAMME's weak-learner acceptance threshold.
    threshold = 1.0 - 1.0 / n_classes

    if error >= threshold:
        print(
            f"Round {round_number} stopped: weighted error "
            f"{error:.6f} >= threshold {threshold:.6f}"
        )
        break

    # Multiclass SAMME learner weight.
    clipped_error = np.clip(error, 1e-12, 1.0 - 1e-12)
    alpha = (
        np.log((1.0 - clipped_error) / clipped_error)
        + np.log(n_classes - 1)
    )

    # SAMME multiplies misclassified sample weights by exp(alpha).
    weights_after = weights_before * np.exp(alpha * incorrect)
    weights_after /= weights_after.sum()

    sample_rows = []
    for i in range(n_samples):
        sample_rows.append({
            "sample_number": i + 1,
            "training_position": int(selected_positions[i]),
            "true_label": int(y_trace[i]),
            "predicted_label": int(predictions[i]),
            "correct": bool(not incorrect[i]),
            "weight_before": float(weights_before[i]),
            "weight_after": float(weights_after[i]),
        })

    record = {
        "round": round_number,
        "max_depth": MAX_DEPTH,
        "weighted_error": error,
        "misclassified_count": int(incorrect.sum()),
        "learner_weight_alpha": float(alpha),
        "weight_sum_before": float(weights_before.sum()),
        "weight_sum_after": float(weights_after.sum()),
        "samples": sample_rows,
    }
    trace.append(record)

    print(f"\nRound {round_number}")
    print(f"  Weighted error: {error:.6f}")
    print(f"  Misclassified: {incorrect.sum()}/{n_samples}")
    print(f"  SAMME alpha: {alpha:.6f}")
    print(f"  Updated weight sum: {weights_after.sum():.6f}")
    print("  Sample | True | Pred | Weight before | Weight after")

    for row in sample_rows:
        print(
            f"  {row['sample_number']:>6} |"
            f" {row['true_label']:>4} |"
            f" {row['predicted_label']:>4} |"
            f" {row['weight_before']:.6f} |"
            f" {row['weight_after']:.6f}"
        )

    weights = weights_after

output = {
    "seed": SEED,
    "trace_sample_count": n_samples,
    "selected_training_positions": selected_positions.tolist(),
    "max_depth": MAX_DEPTH,
    "n_classes": n_classes,
    "rounds_completed": len(trace),
    "round_trace": trace,
}

output_path = RESULTS_DIR / "samme_trace.json"
with output_path.open("w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved trace to: {output_path}")
