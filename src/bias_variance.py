
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

SEED = 26007
N_RESAMPLES = 30

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset and reuse the saved split.
dataset = fetch_olivetti_faces()
X = dataset.images.reshape(len(dataset.images), -1)
y = dataset.target

split = np.load(RESULTS_DIR / "split.npz")
train_idx = split["train_indices"]
test_idx = split["test_indices"]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

classes = np.unique(y_train)
rng = np.random.default_rng(SEED)

# Each row stores predictions from one bootstrap-trained tree.
all_predictions = []
resample_results = []

for r in range(N_RESAMPLES):
    bootstrap_idx = rng.integers(
        0, len(X_train), size=len(X_train)
    )

    model = DecisionTreeClassifier(
        max_depth=None,
        min_samples_leaf=4,
        random_state=SEED + r,
    )

    model.fit(X_train[bootstrap_idx], y_train[bootstrap_idx])
    predictions = model.predict(X_test)
    all_predictions.append(predictions)

    accuracy = accuracy_score(y_test, predictions)

    resample_results.append({
        "resample": r + 1,
        "test_accuracy": float(accuracy),
        "bootstrap_unique_samples": int(
            np.unique(bootstrap_idx).size
        ),
    })

    print(
        f"Resample {r + 1:02d}/{N_RESAMPLES}: "
        f"test accuracy={accuracy:.4f}"
    )

predictions = np.asarray(all_predictions)
# Shape: (30 resamples, 80 test images)

# Mean accuracy across independently trained models.
mean_accuracy = float(
    np.mean([r["test_accuracy"] for r in resample_results])
)
std_accuracy = float(
    np.std([r["test_accuracy"] for r in resample_results])
)

# Per-image prediction disagreement across resamples.
# This is a classification analogue of prediction instability.
per_image_disagreement = np.mean(
    predictions != predictions[0:1, :],
    axis=0,
)

# For each test image, estimate the fraction of model pairs
# that disagree on its predicted identity.
per_image_pairwise_disagreement = []
for j in range(len(y_test)):
    votes = predictions[:, j]
    pairwise = np.mean(votes[:, None] != votes[None, :])
    per_image_pairwise_disagreement.append(float(pairwise))

mean_pairwise_disagreement = float(
    np.mean(per_image_pairwise_disagreement)
)

# Majority-vote prediction across the 30 fitted trees.
ensemble_predictions = []
for j in range(len(y_test)):
    counts = np.array([
        np.sum(predictions[:, j] == c) for c in classes
    ])
    ensemble_predictions.append(classes[np.argmax(counts)])

ensemble_predictions = np.asarray(ensemble_predictions)
ensemble_accuracy = float(
    accuracy_score(y_test, ensemble_predictions)
)

# Per-image empirical distribution of predicted classes.
# Entropy is a complementary measure of prediction uncertainty.
per_image_entropy = []
for j in range(len(y_test)):
    _, counts = np.unique(predictions[:, j], return_counts=True)
    probabilities = counts / N_RESAMPLES
    entropy = -np.sum(probabilities * np.log(probabilities))
    per_image_entropy.append(float(entropy))

mean_entropy = float(np.mean(per_image_entropy))

summary = {
    "seed": SEED,
    "n_resamples": N_RESAMPLES,
    "model": "DecisionTreeClassifier",
    "max_depth": None,
    "min_samples_leaf": 4,
    "test_images": len(y_test),
    "mean_single_tree_test_accuracy": mean_accuracy,
    "std_single_tree_test_accuracy": std_accuracy,
    "majority_vote_accuracy": ensemble_accuracy,
    "mean_pairwise_prediction_disagreement": (
        mean_pairwise_disagreement
    ),
    "mean_prediction_entropy_nats": mean_entropy,
    "resample_results": resample_results,
}

json_path = RESULTS_DIR / "bias_variance.json"
with json_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

csv_path = RESULTS_DIR / "bias_variance_resamples.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "resample",
            "test_accuracy",
            "bootstrap_unique_samples",
        ],
    )
    writer.writeheader()
    writer.writerows(resample_results)

# Plot accuracy across the 30 resamples.
fig, ax = plt.subplots(figsize=(9, 5))
rounds = [r["resample"] for r in resample_results]
accuracies = [r["test_accuracy"] for r in resample_results]

ax.plot(rounds, accuracies, marker="o", linewidth=1)
ax.axhline(
    mean_accuracy,
    linestyle="--",
    label=f"Mean = {mean_accuracy:.3f}",
)
ax.set_xlabel("Bootstrap resample")
ax.set_ylabel("Clean test accuracy")
ax.set_title("Decision Tree Performance Across 30 Resamples")
ax.set_xticks(range(1, N_RESAMPLES + 1, 2))
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

plot_path = FIGURES_DIR / "bias_variance_resamples.png"
fig.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("\n=== Bias–variance experiment summary ===")
print(f"Mean individual-tree accuracy: {mean_accuracy:.4f}")
print(f"Standard deviation:            {std_accuracy:.4f}")
print(f"30-tree majority-vote accuracy: {ensemble_accuracy:.4f}")
print(
    "Mean pairwise prediction disagreement: "
    f"{mean_pairwise_disagreement:.4f}"
)
print(f"Mean prediction entropy (nats): {mean_entropy:.4f}")
print(f"\nSaved: {json_path}")
print(f"Saved: {csv_path}")
print(f"Saved: {plot_path}")
