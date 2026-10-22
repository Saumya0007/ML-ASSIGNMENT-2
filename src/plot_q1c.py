
import json
from pathlib import Path

import matplotlib.pyplot as plt


# ==================================================
# Paths and required ensemble sizes
# ==================================================
ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "results" / "bagging.json"
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = [1, 5, 10, 25, 50, 100, 200]


# ==================================================
# Load bagging results
# ==================================================
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {INPUT_FILE}. "
        "Run src/bagging.py first."
    )

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    results = json.load(file)

all_results = results.get("all_results", [])

if not all_results:
    raise ValueError(
        "No 'all_results' entries found in results/bagging.json."
    )

# Map each saved checkpoint to its result row
by_b = {
    int(row["n_estimators"]): row
    for row in all_results
    if "n_estimators" in row
}

missing = [b for b in CHECKPOINTS if b not in by_b]

if missing:
    raise ValueError(
        f"Missing checkpoints: {missing}. "
        "Update and rerun bagging.py so it saves diagnostics "
        "for all seven required B values."
    )

rows = [by_b[b] for b in CHECKPOINTS]


# ==================================================
# Validate required diagnostic metrics
# ==================================================
required_fields = [
    "validation_accuracy",
    "oob_accuracy",
    "pairwise_disagreement",
    "mean_vote_margin",
]

for b, row in zip(CHECKPOINTS, rows):
    for field in required_fields:
        if field not in row or row[field] is None:
            raise ValueError(
                f"Missing '{field}' for B={b}. "
                "Recalculate and save this diagnostic in bagging.py."
            )


# ==================================================
# Print the selected results
# ==================================================
print("\nQ1(c): Bagging diagnostics")
print("-" * 82)
print(
    f"{'B':>5} {'Validation':>14} {'OOB accuracy':>14} "
    f"{'Disagreement':>15} {'Vote margin':>14}"
)
print("-" * 82)

for b, row in zip(CHECKPOINTS, rows):
    print(
        f"{b:>5} "
        f"{100 * float(row['validation_accuracy']):>13.2f}% "
        f"{100 * float(row['oob_accuracy']):>13.2f}% "
        f"{100 * float(row['pairwise_disagreement']):>14.2f}% "
        f"{float(row['mean_vote_margin']):>14.4f}"
    )

print("-" * 82)


# ==================================================
# Select final B using validation accuracy only
# ==================================================
best_val = max(
    float(row["validation_accuracy"])
    for row in rows
)

tied_best = [
    b
    for b, row in zip(CHECKPOINTS, rows)
    if float(row["validation_accuracy"]) == best_val
]

# Tie-break: choose the larger ensemble
selected_B = max(tied_best)

print(f"\nBest validation accuracy: {100 * best_val:.2f}%")
print(f"Selected B: {selected_B}")
print("Selection rule: validation accuracy; ties go to larger B.")


# ==================================================
# Create ONE combined figure with EXACTLY 3 panels
# ==================================================
B = CHECKPOINTS

plot_settings = [
    (
        "oob_accuracy",
        "Out-of-Bag Accuracy",
        "OOB accuracy",
    ),
    (
        "pairwise_disagreement",
        "Pairwise Disagreement",
        "Average pairwise disagreement",
    ),
    (
        "mean_vote_margin",
        "Mean Vote Margin",
        "Winning fraction - runner-up fraction",
    ),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

for ax, (field, title, ylabel) in zip(axes, plot_settings):
    values = [float(row[field]) for row in rows]

    ax.plot(B, values, marker="o", linewidth=2)
    ax.axvline(
        selected_B,
        linestyle="--",
        linewidth=1.2,
        label=f"Selected B={selected_B}",
    )

    ax.set_xscale("log")
    ax.set_xticks(B)
    ax.set_xticklabels([str(b) for b in B])

    ax.set_xlabel("Number of trees (B)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

fig.tight_layout()

figure_path = FIGURE_DIR / "q1c_bagging_diagnostics.png"

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(f"\nSaved combined 3-panel figure:\n{figure_path}")


# ==================================================
# Generate LaTeX table
# ==================================================
def format_percent(value):
    return f"{100 * float(value):.2f}\\%"


table_lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\small",
    r"\caption{Bagging diagnostics for the required ensemble sizes.}",
    r"\label{tab:bagging_diagnostics}",
    r"\begin{tabular}{rcccc}",
    r"\hline",
    r"$B$ & Validation & OOB accuracy & Disagreement & Vote margin \\",
    r"\hline",
]

for b, row in zip(CHECKPOINTS, rows):
    validation = format_percent(row["validation_accuracy"])
    oob = format_percent(row["oob_accuracy"])
    disagreement = format_percent(row["pairwise_disagreement"])
    margin = f"{float(row['mean_vote_margin']):.4f}"

    table_lines.append(
        f"{b} & {validation} & {oob} & "
        f"{disagreement} & {margin} " + r"\\"
    )

table_lines.extend([
    r"\hline",
    r"\end{tabular}",
    r"\end{table}",
])

table_path = RESULTS_DIR / "q1c_diagnostics_table.tex"

table_path.write_text(
    "\n".join(table_lines) + "\n",
    encoding="utf-8",
)

print(f"Saved LaTeX table:\n{table_path}")
