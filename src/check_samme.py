
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

SEED = 26007

data = fetch_olivetti_faces()
X = data.images.reshape(len(data.images), -1)
y = data.target

split = np.load("results/split.npz")
train_idx = split["train_indices"]

X_train = X[train_idx]
y_train = y[train_idx]

print("Training shape:", X_train.shape)
print("Number of classes:", len(np.unique(y_train)))
print("Class counts:", np.unique(y_train, return_counts=True)[1][:10])

for depth in [1, 2, 3, 4, None]:
    model = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=1,
        random_state=SEED,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_train)

    print(
        f"depth={depth}: "
        f"training accuracy={accuracy_score(y_train, pred):.4f}, "
        f"unique predictions={len(np.unique(pred))}, "
        f"first predictions={pred[:20].tolist()}"
    )
