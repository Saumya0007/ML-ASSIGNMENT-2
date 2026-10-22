from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_olivetti_faces

# Reproducible configuration
SEED = 26007
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load Olivetti Faces
dataset = fetch_olivetti_faces()
images = dataset.images
labels = dataset.target

# Use raw flattened pixels
X = images.reshape(len(images), -1)
y = labels

# Split each identity independently
rng = np.random.default_rng(SEED)

train_indices = []
val_indices = []
test_indices = []

for identity in np.unique(y):
    identity_indices = np.flatnonzero(y == identity)
    shuffled = rng.permutation(identity_indices)

    train_indices.extend(shuffled[:6])
    val_indices.extend(shuffled[6:8])
    test_indices.extend(shuffled[8:10])

train_indices = np.array(train_indices, dtype=int)
val_indices = np.array(val_indices, dtype=int)
test_indices = np.array(test_indices, dtype=int)

# Save the split for reuse in every experiment
np.savez(
    OUTPUT_DIR / "split.npz",
    train_indices=train_indices,
    val_indices=val_indices,
    test_indices=test_indices,
)

# Verify split sizes and disjointness
assert len(train_indices) == 240
assert len(val_indices) == 80
assert len(test_indices) == 80

assert len(np.unique(np.concatenate(
    [train_indices, val_indices, test_indices]
))) == 400

for identity in np.unique(y):
    assert np.sum(y[train_indices] == identity) == 6
    assert np.sum(y[val_indices] == identity) == 2
    assert np.sum(y[test_indices] == identity) == 2

print("Dataset shape:", X.shape)
print("Training samples:", len(train_indices))
print("Validation samples:", len(val_indices))
print("Test samples:", len(test_indices))
print("Split saved to:", OUTPUT_DIR / "split.npz")
print("All split checks passed.")