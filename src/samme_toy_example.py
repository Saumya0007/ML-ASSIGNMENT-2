"""One complete, reproducible SAMME round on a tiny three-class dataset.

Run from the project root:
    python .\\src\\samme_toy_example.py

This is a demonstration only; it does not load or retrain the face dataset.
"""
import numpy as np

# Six samples from three classes; predictions are fixed to make the round
# completely reproducible and easy to inspect.
y_true = np.array(["A", "A", "B", "B", "C", "C"])
weak_predictions = np.array(["A", "A", "B", "C", "C", "A"])

classes = np.unique(y_true)
K = len(classes)
n_samples = len(y_true)
weights = np.full(n_samples, 1.0 / n_samples)

incorrect = weak_predictions != y_true
weighted_error = float(np.sum(weights * incorrect))

# SAMME requires the weak learner to perform better than random guessing.
if not 0.0 < weighted_error < 1.0 - 1.0 / K:
    raise ValueError(
        f"Weak learner error {weighted_error:.6f} is not in "
        f"(0, 1 - 1/K) for K={K} classes."
    )

# Multiclass SAMME learner weight.
alpha = float(
    np.log((1.0 - weighted_error) / weighted_error) + np.log(K - 1)
)

# Increase weights of misclassified samples, then normalize.
unnormalized_weights = weights * np.exp(alpha * incorrect)
updated_weights = unnormalized_weights / unnormalized_weights.sum()

print("SAMME toy example: one complete round")
print(f"Classes (K): {classes.tolist()}")
print("\nSample | True class | Prediction | Initial weight | Correct?")
for i, (truth, pred, weight, wrong) in enumerate(
    zip(y_true, weak_predictions, weights, incorrect), start=1
):
    print(
        f"{i:>6} | {truth:^10} | {pred:^10} | "
        f"{weight:>14.6f} | {'No' if wrong else 'Yes'}"
    )

print(f"\nWeighted error epsilon_t: {weighted_error:.6f}")
print(f"Alpha_t:                  {alpha:.6f}")
print(f"exp(alpha_t):             {np.exp(alpha):.6f}")

print("\nSample | Unnormalized updated weight | Normalized updated weight")
for i, (raw, normalized) in enumerate(
    zip(unnormalized_weights, updated_weights), start=1
):
    print(f"{i:>6} | {raw:>27.6f} | {normalized:>25.6f}")

print(f"\nSum of normalized weights: {updated_weights.sum():.6f}")
