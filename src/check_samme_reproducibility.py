
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

SEED = 26007
DEPTH = 4
CHECKPOINTS = [25, 50, 100]

# Load the same dataset and saved split.
data = fetch_olivetti_faces(shuffle=False)
X = data.images.reshape(len(data.images), -1)
y = data.target

split = np.load("results/split.npz")
train_idx = split["train_indices"]
val_idx = split["val_indices"]
test_idx = split["test_indices"]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

masked_images = data.images[test_idx].copy()
masked_images[:, 24:40, 24:40] = 0
X_masked = masked_images.reshape(len(test_idx), -1)

classes = np.unique(y_train)
K = len(classes)
weights = np.full(len(y_train), 1.0 / len(y_train))

estimators = []
alphas = []

for round_number in range(1, max(CHECKPOINTS) + 1):
    tree = DecisionTreeClassifier(
        max_depth=DEPTH,
        min_samples_leaf=1,
        random_state=SEED + round_number - 1,
    )
    tree.fit(X_train, y_train, sample_weight=weights)

    pred = tree.predict(X_train)
    incorrect = pred != y_train
    error = float(np.sum(weights * incorrect) / np.sum(weights))

    if error >= 1.0 - 1.0 / K:
        print(f"Stopped at round {round_number}; error={error:.8f}")
        break

    error = float(np.clip(error, 1e-12, 1.0 - 1e-12))
    alpha = np.log((1.0 - error) / error) + np.log(K - 1)

    weights *= np.exp(alpha * incorrect)
    weights /= weights.sum()

    estimators.append(tree)
    alphas.append(float(alpha))

    if round_number in CHECKPOINTS:
        def predict_prefix(X_eval, n):
            scores = np.zeros((len(X_eval), K))
            for estimator, a in zip(estimators[:n], alphas[:n]):
                p = estimator.predict(X_eval)
                for j, cls in enumerate(classes):
                    scores[:, j] += a * (p == cls)
            return classes[np.argmax(scores, axis=1)]

        print(f"\nRounds: {round_number}")
        print(f"  Accepted learners: {len(estimators)}")
        print(f"  Train: {accuracy_score(y_train, predict_prefix(X_train, round_number)):.4f}")
        print(f"  Val:   {accuracy_score(y_val, predict_prefix(X_val, round_number)):.4f}")
        print(f"  Test:  {accuracy_score(y_test, predict_prefix(X_test, round_number)):.4f}")
        print(f"  Masked:{accuracy_score(y_test, predict_prefix(X_masked, round_number)):.4f}")
        print(f"  Last weighted error: {error:.8f}")
        print(f"  Last alpha: {alpha:.8f}")
