
from pathlib import Path
import csv
import json
import time

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

SEED = 26007
DEPTH = 4
MIN_LEAF = 1
CHECKPOINTS = [25, 50, 100]

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# Load the same dataset and fixed split used by the assignment.
data = fetch_olivetti_faces(shuffle=False)
X = data.images.reshape(len(data.images), -1)
y = data.target.astype(int)

split = np.load(RESULTS / "split.npz")
train_idx = split["train_indices"]
val_idx = split["val_indices"]
test_idx = split["test_indices"]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

masked_images = data.images[test_idx].copy()
masked_images[:, 24:40, 24:40] = 0.0
X_masked = masked_images.reshape(len(test_idx), -1)

classes = np.unique(y_train)
K = len(classes)
weights = np.full(len(y_train), 1.0 / len(y_train))

estimators = []
alphas = []
round_errors = []
round_seconds = []
rows = []
start_all = time.perf_counter()


def scores_for(X_eval, n):
    scores = np.zeros((len(X_eval), K))
    for tree, alpha in zip(estimators[:n], alphas[:n]):
        pred = tree.predict(X_eval)
        for j, cls in enumerate(classes):
            scores[:, j] += alpha * (pred == cls)
    return scores


def predict_for(X_eval, n):
    return classes[np.argmax(scores_for(X_eval, n), axis=1)]


for round_number in range(1, max(CHECKPOINTS) + 1):
    start_round = time.perf_counter()

    tree = DecisionTreeClassifier(
        max_depth=DEPTH,
        min_samples_leaf=MIN_LEAF,
        random_state=SEED + round_number - 1,
    )
    tree.fit(X_train, y_train, sample_weight=weights)

    train_pred = tree.predict(X_train)
    incorrect = train_pred != y_train
    error = float(np.sum(weights * incorrect) / np.sum(weights))

    if error >= 1.0 - 1.0 / K:
        print(f"Stopped at round {round_number}: error={error:.8f}")
        break

    error = float(np.clip(error, 1e-12, 1.0 - 1e-12))
    alpha = float(np.log((1.0 - error) / error) + np.log(K - 1))

    weights *= np.exp(alpha * incorrect)
    weights /= weights.sum()

    estimators.append(tree)
    alphas.append(alpha)
    round_errors.append(error)
    round_seconds.append(time.perf_counter() - start_round)

    if round_number not in CHECKPOINTS:
        continue

    train_hat = predict_for(X_train, round_number)
    val_hat = predict_for(X_val, round_number)
    test_hat = predict_for(X_test, round_number)
    masked_hat = predict_for(X_masked, round_number)

    clean_scores = scores_for(X_test, round_number)
    masked_scores = scores_for(X_masked, round_number)

    clean_winning_score = (
        np.max(clean_scores, axis=1) / np.sum(alphas[:round_number])
    )
    masked_winning_score = (
        np.max(masked_scores, axis=1) / np.sum(alphas[:round_number])
    )

    elapsed = sum(round_seconds[:round_number])
    row = {
        "model": "SAMME",
        "learners": round_number,
        "training_seconds": elapsed,
        "train_accuracy": float(accuracy_score(y_train, train_hat)),
        "validation_accuracy": float(accuracy_score(y_val, val_hat)),
        "clean_test_accuracy": float(accuracy_score(y_test, test_hat)),
        "masked_test_accuracy": float(accuracy_score(y_test, masked_hat)),
        "last_weighted_error": error,
        "last_learner_alpha": alpha,
    }
    rows.append(row)

    np.savez_compressed(
        RESULTS / f"samme_predictions_{round_number}.npz",
        test_indices=test_idx,
        y_test=y_test,
        clean_prediction=test_hat,
        masked_prediction=masked_hat,
        clean_winning_score=clean_winning_score,
        masked_winning_score=masked_winning_score,
        clean_class_scores=clean_scores,
        masked_class_scores=masked_scores,
        estimator_weights=np.asarray(alphas),
    )

    print(
        f"Rounds={round_number:3d} "
        f"train={row['train_accuracy']:.4f} "
        f"val={row['validation_accuracy']:.4f} "
        f"clean={row['clean_test_accuracy']:.4f} "
        f"masked={row['masked_test_accuracy']:.4f}"
    )

elapsed_total = time.perf_counter() - start_all

save_csv(RESULTS / "samme_corrected_dynamics.csv", rows)

output = {
    "seed": SEED,
    "max_depth": DEPTH,
    "min_samples_leaf": MIN_LEAF,
    "accepted_rounds": len(estimators),
    "total_wall_time_seconds": elapsed_total,
    "checkpoint_results": rows,
    "confidence_note": (
        "Normalized winning weighted-vote score; not a calibrated probability."
    ),
    "prediction_files": [
        f"results/samme_predictions_{n}.npz"
        for n in CHECKPOINTS if n <= len(estimators)
    ],
}

with (RESULTS / "samme_corrected_summary.json").open(
    "w", encoding="utf-8"
) as f:
    json.dump(output, f, indent=2)

print("\nSaved corrected SAMME-only outputs:")
print("  results/samme_corrected_dynamics.csv")
print("  results/samme_corrected_summary.json")
print("  results/samme_predictions_25.npz")
print("  results/samme_predictions_50.npz")
print("  results/samme_predictions_100.npz")
print("Existing bagging and comparison files were not modified.")
