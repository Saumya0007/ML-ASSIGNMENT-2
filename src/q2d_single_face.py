from pathlib import Path
import csv, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

SEED = 26007
BAGGING_TREES = 200
SAMME_DEPTH = 4
SAMME_MIN_LEAF = 1
SAMME_ROUNDS = 100
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

data = fetch_olivetti_faces(shuffle=False)
X = data.images.reshape(len(data.images), -1)
y = data.target.astype(int)
split = np.load(RESULTS / "split.npz")
train_idx, test_idx = split["train_indices"], split["test_indices"]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]
images = data.images[test_idx].copy()
masked_images = images.copy()
masked_images[:, 24:40, 24:40] = 0.0
X_masked = masked_images.reshape(len(test_idx), -1)
classes = np.unique(y_train)
K = len(classes)

# Single-tree baseline matching the project baseline.
single_tree = DecisionTreeClassifier(max_depth=None, min_samples_leaf=4, random_state=SEED)
single_tree.fit(X_train, y_train)

# 200-tree bagging ensemble using the seeded bootstrap scheme.
rng = np.random.default_rng(SEED)
bag_trees = []
for t in range(BAGGING_TREES):
    boot = rng.choice(len(train_idx), size=len(train_idx), replace=True)
    model = DecisionTreeClassifier(max_depth=None, min_samples_leaf=4, random_state=SEED + t)
    model.fit(X_train[boot], y_train[boot])
    bag_trees.append(model)

def bag_scores(X_eval):
    tree_preds = np.stack([m.predict(X_eval) for m in bag_trees], axis=0)
    scores = np.stack([(tree_preds == c).sum(axis=0) for c in classes], axis=1).astype(float)
    return scores, tree_preds

bag_clean_scores, bag_clean_tree_preds = bag_scores(X_test)
bag_mask_scores, bag_mask_tree_preds = bag_scores(X_masked)
bag_clean_pred = classes[bag_clean_scores.argmax(axis=1)]
bag_mask_pred = classes[bag_mask_scores.argmax(axis=1)]
bag_clean_conf = bag_clean_scores.max(axis=1) / BAGGING_TREES
bag_mask_conf = bag_mask_scores.max(axis=1) / BAGGING_TREES
tree_clean_pred, tree_mask_pred = single_tree.predict(X_test), single_tree.predict(X_masked)

# Correct multiclass SAMME implementation matching the supplied script.
weights = np.full(len(y_train), 1.0 / len(y_train))
samme_models, alphas = [], []
for t in range(1, SAMME_ROUNDS + 1):
    model = DecisionTreeClassifier(max_depth=SAMME_DEPTH, min_samples_leaf=SAMME_MIN_LEAF,
                                   random_state=SEED + t - 1)
    model.fit(X_train, y_train, sample_weight=weights)
    pred = model.predict(X_train)
    incorrect = pred != y_train
    err = float(np.sum(weights * incorrect) / np.sum(weights))
    if err >= 1.0 - 1.0 / K:
        break
    err = float(np.clip(err, 1e-12, 1.0 - 1e-12))
    alpha = float(np.log((1.0 - err) / err) + np.log(K - 1))
    weights *= np.exp(alpha * incorrect)
    weights /= weights.sum()
    samme_models.append(model)
    alphas.append(alpha)

def samme_scores(X_eval):
    scores = np.zeros((len(X_eval), len(classes)), dtype=float)
    for model, alpha in zip(samme_models, alphas):
        pred = model.predict(X_eval)
        for j, cls in enumerate(classes):
            scores[:, j] += alpha * (pred == cls)
    return scores

sc_clean, sc_mask = samme_scores(X_test), samme_scores(X_masked)
samme_clean_pred, samme_mask_pred = classes[sc_clean.argmax(axis=1)], classes[sc_mask.argmax(axis=1)]
alpha_sum = float(np.sum(alphas))
samme_clean_conf, samme_mask_conf = sc_clean.max(axis=1)/alpha_sum, sc_mask.max(axis=1)/alpha_sum
bag_changed, samme_changed = bag_clean_pred != bag_mask_pred, samme_clean_pred != samme_mask_pred

# Choose a real test face that illustrates different masking responses, if available.
candidates = np.where(bag_changed != samme_changed)[0]
rule = "one ensemble changes its prediction under masking while the other does not"
if len(candidates) == 0:
    candidates = np.where(bag_mask_pred != samme_mask_pred)[0]
    rule = "fallback: bagging and SAMME masked predictions differ"
if len(candidates) == 0:
    impact = np.abs(bag_mask_conf-bag_clean_conf) + np.abs(samme_mask_conf-samme_clean_conf)
    candidates = np.array([int(np.argmax(impact))])
    rule = "fallback: largest combined change in winning-score confidence"
i = int(candidates[0])
global_index, true_label = int(test_idx[i]), int(y_test[i])
tree_flip_fraction = float(np.mean(bag_clean_tree_preds[:, i] != bag_mask_tree_preds[:, i]))
bag_disagreement_clean = float(1 - np.max(bag_clean_scores[i])/BAGGING_TREES)
bag_disagreement_mask = float(1 - np.max(bag_mask_scores[i])/BAGGING_TREES)
top_weight_local = int(np.argmax(weights))

record = {
    "seed": SEED, "selected_test_position": i, "selected_global_dataset_index": global_index,
    "true_identity_label": true_label, "selection_rule": rule,
    "bagging": {"trees": BAGGING_TREES, "clean_prediction": int(bag_clean_pred[i]),
        "masked_prediction": int(bag_mask_pred[i]), "clean_winning_vote_fraction": float(bag_clean_conf[i]),
        "masked_winning_vote_fraction": float(bag_mask_conf[i]),
        "prediction_changed_by_mask": bool(bag_changed[i]),
        "tree_prediction_flip_fraction_under_mask": tree_flip_fraction,
        "clean_vote_disagreement": bag_disagreement_clean, "masked_vote_disagreement": bag_disagreement_mask},
    "samme": {"depth": SAMME_DEPTH, "rounds_accepted": len(samme_models),
        "clean_prediction": int(samme_clean_pred[i]), "masked_prediction": int(samme_mask_pred[i]),
        "clean_normalized_winning_score": float(samme_clean_conf[i]),
        "masked_normalized_winning_score": float(samme_mask_conf[i]),
        "prediction_changed_by_mask": bool(samme_changed[i]),
        "confidence_note": "Normalized winning weighted-vote score, not a calibrated probability."},
    "highest_final_samme_training_weight": {"global_dataset_index": int(train_idx[top_weight_local]),
        "identity_label": int(y_train[top_weight_local]), "weight": float(weights[top_weight_local]),
        "note": "This is a training face, not the selected test face."},
    "test_accuracies": {
        "single_tree_clean": float(accuracy_score(y_test, tree_clean_pred)),
        "single_tree_masked": float(accuracy_score(y_test, tree_mask_pred)),
        "bagging_200_clean": float(accuracy_score(y_test, bag_clean_pred)),
        "bagging_200_masked": float(accuracy_score(y_test, bag_mask_pred)),
        "samme_clean": float(accuracy_score(y_test, samme_clean_pred)),
        "samme_masked": float(accuracy_score(y_test, samme_mask_pred))}
}
(RESULTS / "q2d_single_face_example.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 5))
axes[0].imshow(images[i], cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Original test face")
axes[1].imshow(masked_images[i], cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Same face: central 16×16 masked")
for ax in axes: ax.axis("off")
caption = (f"True identity: {true_label} | Dataset index: {global_index}\n"
 f"Bagging (200): {int(bag_clean_pred[i])} → {int(bag_mask_pred[i])}; "
 f"winning vote {bag_clean_conf[i]:.3f} → {bag_mask_conf[i]:.3f}\n"
 f"SAMME ({len(samme_models)} rounds): {int(samme_clean_pred[i])} → {int(samme_mask_pred[i])}; "
 f"normalized winning score {samme_clean_conf[i]:.3f} → {samme_mask_conf[i]:.3f}\n"
 f"Bagging trees whose prediction changes under mask: {tree_flip_fraction:.1%}")
fig.suptitle("Q2(d): one test face under central occlusion")
fig.text(0.5, 0.035, caption, ha="center", va="bottom", fontsize=9)
fig.tight_layout(rect=[0, 0.20, 1, 0.90])
fig.savefig(FIGURES / "q2d_single_face_example.png", dpi=200, bbox_inches="tight")
plt.close(fig)

with (RESULTS / "q2d_single_face_example.csv").open("w", newline="", encoding="utf-8") as f:
    fields = ["global_dataset_index","true_identity_label","bagging_clean","bagging_masked",
              "bagging_vote_clean","bagging_vote_masked","samme_clean","samme_masked",
              "samme_score_clean","samme_score_masked","bagging_tree_flip_fraction","selection_rule"]
    writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
    writer.writerow({"global_dataset_index":global_index,"true_identity_label":true_label,
        "bagging_clean":int(bag_clean_pred[i]),"bagging_masked":int(bag_mask_pred[i]),
        "bagging_vote_clean":f"{bag_clean_conf[i]:.6f}","bagging_vote_masked":f"{bag_mask_conf[i]:.6f}",
        "samme_clean":int(samme_clean_pred[i]),"samme_masked":int(samme_mask_pred[i]),
        "samme_score_clean":f"{samme_clean_conf[i]:.6f}","samme_score_masked":f"{samme_mask_conf[i]:.6f}",
        "bagging_tree_flip_fraction":f"{tree_flip_fraction:.6f}","selection_rule":rule})
print("Saved results/q2d_single_face_example.json")
print("Saved results/q2d_single_face_example.csv")
print("Saved results/figures/q2d_single_face_example.png")
print(json.dumps(record, indent=2))
