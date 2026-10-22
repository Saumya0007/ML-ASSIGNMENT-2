"""Q2(b): SAMME depth/round grid search and selected-model dynamics.

Place this file in src/ and run from the project root:
    python .\src\samme_grid_dynamics.py

Uses the existing results/split.npz, seed 26007, Olivetti raw pixels,
and the same multiclass SAMME update rule as correct_samme_comparison.py.
Model selection uses clean validation accuracy only.
"""
from pathlib import Path
import csv
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

SEED = 26007
DEPTHS = [1, 2, 3, 4]
CHECKPOINTS = [10, 25, 50, 100]
MIN_LEAF = 1

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

def save_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

# Load the exact fixed split already used by this project.
data = fetch_olivetti_faces(shuffle=False)
X = data.images.reshape(len(data.images), -1)
y = data.target.astype(int)
split_path = RESULTS / "split.npz"
if not split_path.exists():
    raise FileNotFoundError(
        f"Missing {split_path}. Run the project's data/split setup first."
    )
split = np.load(split_path)
train_idx, val_idx = split["train_indices"], split["val_indices"]
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
classes = np.unique(y_train)
K = len(classes)
n_train = len(y_train)

grid_rows = []
dynamics_by_model = {}
started = time.perf_counter()

for depth in DEPTHS:
    weights = np.full(n_train, 1.0 / n_train, dtype=float)
    estimators, alphas = [], []
    dynamics = []
    stopped_early = False

    for t in range(1, max(CHECKPOINTS) + 1):
        tree = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=MIN_LEAF,
            random_state=SEED + t - 1,
        )
        tree.fit(X_train, y_train, sample_weight=weights)
        pred_train = tree.predict(X_train)
        incorrect = pred_train != y_train
        error = float(np.sum(weights * incorrect) / np.sum(weights))

        # Standard SAMME weak-learner validity condition.
        if error >= 1.0 - 1.0 / K:
            stopped_early = True
            print(f"Depth {depth}: stopped before round {t}; "
                  f"weighted error={error:.6f} is not below 1-1/K.")
            break

        error = float(np.clip(error, 1e-12, 1.0 - 1e-12))
        alpha = float(np.log((1.0 - error) / error) + np.log(K - 1))

        # SAMME update: misclassified examples are upweighted, then normalize.
        weights *= np.exp(alpha * incorrect)
        weights /= weights.sum()

        estimators.append(tree)
        alphas.append(alpha)

        # Multiclass SAMME prediction by alpha-weighted class votes.
        train_scores = np.zeros((len(X_train), K))
        val_scores = np.zeros((len(X_val), K))
        for fitted_tree, fitted_alpha in zip(estimators, alphas):
            train_pred = fitted_tree.predict(X_train)
            val_pred = fitted_tree.predict(X_val)
            for j, cls in enumerate(classes):
                train_scores[:, j] += fitted_alpha * (train_pred == cls)
                val_scores[:, j] += fitted_alpha * (val_pred == cls)
        train_hat = classes[np.argmax(train_scores, axis=1)]
        val_hat = classes[np.argmax(val_scores, axis=1)]

        sorted_weights = np.sort(weights)[::-1]
        dynamics.append({
            "depth": depth,
            "round": t,
            "weighted_training_error": error,
            "learner_alpha": alpha,
            "max_sample_weight": float(weights.max()),
            "top10_sample_weight_sum": float(sorted_weights[:10].sum()),
            "train_accuracy": float(accuracy_score(y_train, train_hat)),
            "clean_validation_accuracy": float(accuracy_score(y_val, val_hat)),
            "weight_sum": float(weights.sum()),
        })

    dynamics_by_model[depth] = dynamics
    for checkpoint in CHECKPOINTS:
        available = [r for r in dynamics if r["round"] == checkpoint]
        if available:
            d = available[0]
            grid_rows.append({
                "depth": depth,
                "rounds": checkpoint,
                "clean_validation_accuracy": d["clean_validation_accuracy"],
                "train_accuracy": d["train_accuracy"],
                "weighted_training_error": d["weighted_training_error"],
                "reached_checkpoint": True,
            })
        else:
            grid_rows.append({
                "depth": depth,
                "rounds": checkpoint,
                "clean_validation_accuracy": "",
                "train_accuracy": "",
                "weighted_training_error": "",
                "reached_checkpoint": False,
            })

# Save all grid outcomes, including checkpoints that were unreachable.
save_csv(RESULTS / "samme_q2b_grid.csv", grid_rows)
all_dynamics = [r for depth in DEPTHS for r in dynamics_by_model[depth]]
save_csv(RESULTS / "samme_q2b_all_round_dynamics.csv", all_dynamics)

eligible = [r for r in grid_rows if r["reached_checkpoint"]]
if not eligible:
    raise RuntimeError("No requested checkpoint was reached by any model.")

# Selection is strictly by clean validation accuracy. Stable tie-break:
# fewer rounds, then shallower depth, to prefer the simpler model.
best = sorted(
    eligible,
    key=lambda r: (
        -float(r["clean_validation_accuracy"]),
        int(r["rounds"]),
        int(r["depth"]),
    ),
)[0]
selected_depth, selected_rounds = int(best["depth"]), int(best["rounds"])
selected_dynamics = [
    r for r in dynamics_by_model[selected_depth]
    if r["round"] <= selected_rounds
]
if not any(r["round"] == selected_rounds for r in selected_dynamics):
    raise RuntimeError("Selected checkpoint is missing from its round dynamics.")

# Plot four requested traces for the selected depth/round configuration.
rounds = [r["round"] for r in selected_dynamics]
series = [
    ("weighted_training_error", "Weighted training error"),
    ("learner_alpha", r"Learner weight $\alpha_t$"),
    ("max_sample_weight", "Maximum sample weight"),
    ("top10_sample_weight_sum", "Total weight of 10 heaviest images"),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
for ax, (key, title) in zip(axes.flat, series):
    ax.plot(rounds, [r[key] for r in selected_dynamics], marker="o", markersize=2.5)
    ax.set_title(title)
    ax.set_xlabel("Boosting round")
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
fig.suptitle(
    f"SAMME dynamics: selected depth={selected_depth}, "
    f"rounds={selected_rounds} (clean validation selection)"
)
fig.tight_layout()
fig.savefig(FIGURES / "q2b_samme_selected_dynamics.png", dpi=200, bbox_inches="tight")
plt.close(fig)

summary = {
    "seed": SEED,
    "depths_searched": DEPTHS,
    "round_checkpoints": CHECKPOINTS,
    "min_samples_leaf": MIN_LEAF,
    "selection_metric": "clean_validation_accuracy",
    "tie_break": "fewer rounds, then shallower depth",
    "selected_depth": selected_depth,
    "selected_rounds": selected_rounds,
    "selected_clean_validation_accuracy": float(best["clean_validation_accuracy"]),
    "grid_results_csv": "results/samme_q2b_grid.csv",
    "all_round_dynamics_csv": "results/samme_q2b_all_round_dynamics.csv",
    "selected_dynamics_plot": "results/figures/q2b_samme_selected_dynamics.png",
    "elapsed_seconds": time.perf_counter() - started,
    "notes": (
        "Weighted training error is the weak learner's weighted error epsilon_t "
        "before the round's sample-weight update. Alpha and sample-weight traces "
        "are recorded after fitting each weak learner; sample weights are normalized "
        "after every update."
    ),
}
with (RESULTS / "samme_q2b_summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nQ2(b) grid results (clean validation accuracy only):")
for row in sorted(eligible, key=lambda r: (int(r["depth"]), int(r["rounds"]))):
    print(f"depth={int(row['depth'])}, rounds={int(row['rounds']):3d}, "
          f"validation={float(row['clean_validation_accuracy']):.4f}")
print(f"\nSelected model: depth={selected_depth}, rounds={selected_rounds}, "
      f"clean validation accuracy={float(best['clean_validation_accuracy']):.4f}")
print("Saved:")
print("  results/samme_q2b_grid.csv")
print("  results/samme_q2b_all_round_dynamics.csv")
print("  results/samme_q2b_summary.json")
print("  results/figures/q2b_samme_selected_dynamics.png")
