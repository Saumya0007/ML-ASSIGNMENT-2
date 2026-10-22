
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# Authoritative SAMME checkpoint metrics.
summary = read_json(RESULTS / "samme_corrected_summary.json")
checkpoints = {
    int(row["learners"]): row
    for row in summary["checkpoint_results"]
}

# Load the saved split and true test labels.
split = np.load(RESULTS / "split.npz")
test_idx = split["test_indices"]

# Keep all existing bagging rows; replace only SAMME metrics.
dynamics = read_csv(RESULTS / "q2c_dynamics.csv")
for row in dynamics:
    if row["model"] != "SAMME":
        continue

    n = int(row["learners"])
    if n not in checkpoints:
        raise ValueError(f"Missing corrected SAMME checkpoint: {n}")

    corrected = checkpoints[n]
    for key in (
        "training_seconds",
        "train_accuracy",
        "validation_accuracy",
        "clean_test_accuracy",
        "masked_test_accuracy",
    ):
        row[key] = str(corrected[key])

write_csv(RESULTS / "q2c_dynamics.csv", dynamics)

# Load corrected 100-round per-image predictions and score arrays.
pred_file = np.load(RESULTS / "samme_predictions_100.npz")
y_test = pred_file["y_test"]
clean_pred = pred_file["clean_prediction"]
masked_pred = pred_file["masked_prediction"]
clean_scores = pred_file["clean_class_scores"]
masked_scores = pred_file["masked_class_scores"]

# Mean winning score is a normalized weighted-vote confidence proxy,
# NOT a calibrated probability.
clean_winning = pred_file["clean_winning_score"]
masked_winning = pred_file["masked_winning_score"]

selected = read_csv(RESULTS / "q2d_selected_models.csv")
for row in selected:
    if row["model"] != "SAMME":
        continue

    corrected = checkpoints[100]
    row["configuration"] = "depth=4, rounds=100, min_leaf=1"
    for key in (
        "training_seconds",
        "validation_accuracy",
        "clean_test_accuracy",
        "masked_test_accuracy",
    ):
        row[key] = str(corrected[key])

    row["mean_clean_winning_score"] = str(float(clean_winning.mean()))
    row["mean_masked_winning_score"] = str(float(masked_winning.mean()))

write_csv(RESULTS / "q2d_selected_models.csv", selected)

# Synchronize saved per-test-image boosting predictions while preserving
# the tree and bagging predictions already in the CSV.
disagreement_path = RESULTS / "q2_prediction_disagreements.csv"
if disagreement_path.exists():
    disagreements = read_csv(disagreement_path)

    if len(disagreements) != len(y_test):
        raise ValueError(
            "Disagreement CSV does not match the corrected test set size."
        )

    for i, row in enumerate(disagreements):
        true_label = int(y_test[i])
        row["original_dataset_index"] = str(int(test_idx[i]))
        row["true_label"] = str(true_label)
        row["boosting_clean_prediction"] = str(int(clean_pred[i]))
        row["boosting_masked_prediction"] = str(int(masked_pred[i]))
        row["boosting_correct_clean"] = str(bool(clean_pred[i] == true_label))
        row["boosting_correct_masked"] = str(bool(masked_pred[i] == true_label))
        row["bagging_boosting_disagree_clean"] = str(
            int(row["bagging_clean_prediction"]) != int(clean_pred[i])
        )
        row["bagging_boosting_disagree_masked"] = str(
            int(row["bagging_masked_prediction"]) != int(masked_pred[i])
        )

    write_csv(disagreement_path, disagreements)

# Synchronize summary metrics and SAMME confusion matrices.
summary_path = RESULTS / "q2c_d_summary.json"
if summary_path.exists():
    q2_summary = read_json(summary_path)
    q2_summary["samme_config"]["max_depth"] = 4
    q2_summary["samme_config"]["min_samples_leaf"] = 1
    q2_summary["samme_config"]["actual_rounds"] = 100
    q2_summary["samme_config"]["training_seconds"] = summary[
        "total_wall_time_seconds"
    ]

    for row in q2_summary.get("selected_model_metrics", []):
        if row["model"] == "SAMME":
            corrected = checkpoints[100]
            for key in (
                "training_seconds",
                "validation_accuracy",
                "clean_test_accuracy",
                "masked_test_accuracy",
            ):
                row[key] = corrected[key]
            row["mean_clean_winning_score"] = float(clean_winning.mean())
            row["mean_masked_winning_score"] = float(masked_winning.mean())

    # Recover class labels from the saved split's test labels.
    classes = np.unique(y_test)
    q2_summary["clean_confusion_matrix_samme"] = confusion_matrix(
        y_test, clean_pred, labels=classes
    ).tolist()
    q2_summary["masked_confusion_matrix_samme"] = confusion_matrix(
        y_test, masked_pred, labels=classes
    ).tolist()

    if "bagging_clean_prediction" in disagreements[0]:
        bag_clean = np.array([
            int(r["bagging_clean_prediction"]) for r in disagreements
        ])
        bag_masked = np.array([
            int(r["bagging_masked_prediction"]) for r in disagreements
        ])
        q2_summary["clean_prediction_disagreement_count"] = int(
            np.sum(bag_clean != clean_pred)
        )
        q2_summary["masked_prediction_disagreement_count"] = int(
            np.sum(bag_masked != masked_pred)
        )

    write_json(summary_path, q2_summary)

# Regenerate Q2(c) dynamics plot from the synchronized CSV.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for model in ("Bagging", "SAMME"):
    rows = [r for r in dynamics if r["model"] == model]
    if not rows:
        continue

    axes[0].plot(
        [int(r["learners"]) for r in rows],
        [float(r["train_accuracy"]) for r in rows],
        marker="o", label=model
    )
    axes[1].plot(
        [int(r["learners"]) for r in rows],
        [float(r["validation_accuracy"]) for r in rows],
        marker="o", label=model
    )
    axes[2].plot(
        [int(r["learners"]) for r in rows],
        [float(r["training_seconds"]) for r in rows],
        marker="o", label=model
    )

for ax, title, ylabel in zip(
    axes,
    ("Training accuracy", "Validation accuracy", "Cumulative training time"),
    ("Accuracy", "Accuracy", "Seconds"),
):
    ax.set_title(title)
    ax.set_xlabel("Number of learners")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[0].set_ylim(0, 1)
axes[1].set_ylim(0, 1)
fig.tight_layout()
fig.savefig(FIGURES / "q2c_learning_dynamics.png", dpi=200)
plt.close(fig)

# Regenerate Q2(d) clean/masked plot from the synchronized selected table.
names = [r["model"] for r in selected]
clean = [float(r["clean_test_accuracy"]) for r in selected]
masked = [float(r["masked_test_accuracy"]) for r in selected]
x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - width / 2, clean, width, label="Clean test")
ax.bar(x + width / 2, masked, width, label="Masked test")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.set_title("Clean vs. Masked Test Accuracy")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "q2d_clean_vs_masked.png", dpi=200)
plt.close(fig)

print("Q2 artifacts synchronized from saved corrected SAMME outputs.")
print("No models were retrained.")
print("\nUpdated SAMME checkpoints:")
for n in (25, 50, 100):
    r = checkpoints[n]
    print(
        f"{n:3d} rounds: train={r['train_accuracy']:.4f}, "
        f"val={r['validation_accuracy']:.4f}, "
        f"clean={r['clean_test_accuracy']:.4f}, "
        f"masked={r['masked_test_accuracy']:.4f}"
    )
print("\nReview these files:")
print("  results/q2c_dynamics.csv")
print("  results/q2d_selected_models.csv")
print("  results/q2c_d_summary.json")
print("  results/q2_prediction_disagreements.csv")
print("  results/figures/q2c_learning_dynamics.png")
print("  results/figures/q2d_clean_vs_masked.png")
