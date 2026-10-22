
"""Q2(c): Inspect a high-weight training face and its SAMME history.

Run from the project root:
    python src/q2c_high_weight_face.py
"""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

SEED = 26007
DEPTH = 4
N_ROUNDS = 100

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Load the same dataset and fixed split used by the assignment.
data = fetch_olivetti_faces(shuffle=False)
X = data.images.reshape(len(data.images), -1)
images = data.images
y = data.target.astype(int)

split = np.load(RESULTS / "split.npz")
train_idx = split["train_indices"]
X_train, y_train = X[train_idx], y[train_idx]

classes = np.unique(y_train)
K = len(classes)
n = len(y_train)
weights = np.full(n, 1.0 / n)

# Store each training image's weight and weak-learner prediction.
weight_history = []
prediction_history = []
error_history = []
alpha_history = []

for t in range(1, N_ROUNDS + 1):
    tree = DecisionTreeClassifier(
        max_depth=DEPTH,
        min_samples_leaf=1,
        random_state=SEED + t - 1,
    )
    tree.fit(X_train, y_train, sample_weight=weights)

    pred = tree.predict(X_train)
    incorrect = pred != y_train
    error = float(np.sum(weights * incorrect) / weights.sum())

    # SAMME requires the weak learner to perform better than random.
    if error >= 1.0 - 1.0 / K:
        print(
            f"Stopped before round {t}: "
            f"weighted error={error:.6f}"
        )
        break

    error = float(np.clip(error, 1e-12, 1.0 - 1e-12))
    alpha = float(
        np.log((1.0 - error) / error) + np.log(K - 1)
    )

    # SAMME weight update and normalization.
    weights *= np.exp(alpha * incorrect)
    weights /= weights.sum()

    weight_history.append(weights.copy())
    prediction_history.append(pred.copy())
    error_history.append(error)
    alpha_history.append(alpha)

if not weight_history:
    raise RuntimeError("No valid SAMME rounds were completed.")

weight_history = np.asarray(weight_history)
prediction_history = np.asarray(prediction_history)

# Select the training image with the highest weight observed
# across all completed rounds.
sample_position, round_position = np.unravel_index(
    np.argmax(weight_history), weight_history.shape
)

dataset_index = int(train_idx[sample_position])
true_label = int(y_train[sample_position])
peak_weight = float(
    weight_history[round_position, sample_position]
)

print("\nHighest-weight training face")
print(f"Training position: {sample_position}")
print(f"Original dataset index: {dataset_index}")
print(f"True identity label: {true_label}")
print(f"Peak weight: {peak_weight:.6f}")
print(f"Round of peak weight: {round_position + 1}")

# Save the selected face's prediction and weight history.
csv_path = RESULTS / "q2c_high_weight_face_history.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "round",
        "true_label",
        "predicted_label",
        "correct",
        "sample_weight_after_update",
        "weighted_learner_error",
        "learner_alpha",
    ])

    for r in range(len(weight_history)):
        prediction = int(prediction_history[r, sample_position])
        writer.writerow([
            r + 1,
            true_label,
            prediction,
            prediction == true_label,
            f"{weight_history[r, sample_position]:.8f}",
            f"{error_history[r]:.8f}",
            f"{alpha_history[r]:.8f}",
        ])

# Prepare plot data.
rounds = np.arange(1, len(weight_history) + 1)
face_weights = weight_history[:, sample_position]
face_predictions = prediction_history[:, sample_position]
correct = face_predictions == true_label

# Three-panel figure.
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

# Panel 1: selected training face.
axes[0].imshow(
    images[dataset_index],
    cmap="gray",
    vmin=0,
    vmax=1,
)
axes[0].set_title(
    f"Selected training face\n"
    f"Index {dataset_index}, identity {true_label}"
)
axes[0].axis("off")

# Panel 2: sample weight over boosting rounds.
axes[1].plot(
    rounds,
    face_weights,
    marker=".",
    markersize=3,
)
axes[1].axvline(
    round_position + 1,
    linestyle="--",
    label=f"Peak: round {round_position + 1}",
)
axes[1].set_title("Sample weight after each round")
axes[1].set_xlabel("Boosting round")
axes[1].set_ylabel("Normalized sample weight")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Panel 3: swapped axes.
# X-axis = predicted identity; Y-axis = boosting round.
axes[2].scatter(
    face_predictions[correct],
    rounds[correct],
    marker="o",
    label="Correct",
    alpha=0.8,
)
axes[2].scatter(
    face_predictions[~correct],
    rounds[~correct],
    marker="x",
    label="Incorrect",
    alpha=0.8,
)
axes[2].axvline(
    true_label,
    linestyle="--",
    label=f"True label: {true_label}",
)
axes[2].set_title("Weak-learner predictions")
axes[2].set_xlabel("Predicted identity label")
axes[2].set_ylabel("Boosting round")
axes[2].set_ylim(0, len(rounds) + 1)
axes[2].grid(True, alpha=0.3)
axes[2].legend(fontsize=8)

fig.tight_layout()

plot_path = FIGURES / "q2c_high_weight_face.png"
fig.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print("\nSaved:")
print(f"  {csv_path.relative_to(ROOT)}")
print(f"  {plot_path.relative_to(ROOT)}")
