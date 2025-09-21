#!/usr/bin/env python3
"""
Multiclass classification with Perceptron and Logistic Regression + fairness metrics
using a real-life dataset (UCI Wine dataset via sklearn).

- Loads real dataset (13 features, 3 classes).
- Splits into train/validation/test.
- Trains Perceptron (OvR) and multinomial Logistic Regression.
- Reports accuracy, macro-F1, classification reports, and confusion matrices.
- Visualizes results with PCA 2D projection.
- Demonstrates fairness metrics with synthetic sensitive attribute correlated to target.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import itertools

RNG = np.random.RandomState(42)

# -----------------------------------------------------------------------------
# 1) Load real dataset: Wine (3 classes, 13 features)
# -----------------------------------------------------------------------------
wine = load_wine()
X, y = wine.data, wine.target
print(wine.DESCR)
X = StandardScaler().fit_transform(X)

# Train/val/test split
X_temp, X_te, y_temp, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RNG)
X_tr, X_va, y_tr, y_va = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RNG)

# -----------------------------------------------------------------------------
# 2) Models: Perceptron and Multinomial Logistic Regression
# -----------------------------------------------------------------------------
percep = Perceptron(max_iter=1000, tol=1e-3, random_state=RNG)
logreg = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=RNG)

percep.fit(X_tr, y_tr)
logreg.fit(X_tr, y_tr)

yp_te_p = percep.predict(X_te)
yp_te_l = logreg.predict(X_te)

# -----------------------------------------------------------------------------
# 3) Evaluation
# -----------------------------------------------------------------------------
print("Perceptron — Test Accuracy:", accuracy_score(y_te, yp_te_p))
print("Perceptron — Macro-F1:", f1_score(y_te, yp_te_p, average="macro"))

print("LogReg — Test Accuracy:", accuracy_score(y_te, yp_te_l))
print("LogReg — Macro-F1:", f1_score(y_te, yp_te_l, average="macro"))

print("\nClassification report (LogReg/Test):\n")
print(classification_report(y_te, yp_te_l, digits=3))

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, y_pred, title in zip(axes, [yp_te_p, yp_te_l], ["Perceptron", "LogReg"]):
    C = confusion_matrix(y_te, y_pred)
    im = ax.imshow(C, interpolation="nearest", cmap="Blues")
    ax.set_title(f"{title} Confusion")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(wine.target_names)))
    ax.set_yticks(range(len(wine.target_names)))
    for i, j in itertools.product(range(C.shape[0]), range(C.shape[1])):
        ax.text(j, i, C[i, j], ha="center", va="center", color="red")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 4) Visualization in 2D using PCA
# -----------------------------------------------------------------------------
pca = PCA(n_components=2)
X2d = pca.fit_transform(X)

fig, ax = plt.subplots(figsize=(6, 5))
scatter = ax.scatter(X2d[:, 0], X2d[:, 1], c=y, cmap="viridis", edgecolor="k", s=40)
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)
ax.set_title("Wine dataset (PCA projection)")
plt.show()

# -----------------------------------------------------------------------------
# 5) Fairness demonstration with synthetic sensitive attribute
# -----------------------------------------------------------------------------
# Define binary sensitive attribute correlated with whether alcohol content > median.
median_alcohol = np.median(wine.data[:, 0])
s = (wine.data[:, 0] > median_alcohol).astype(int)

# Binarize label: class 0 vs rest
y_bin = (y == 0).astype(int)

# Train/test split for binary fairness analysis
Xb_tr, Xb_te, yb_tr, yb_te, sb_tr, sb_te = train_test_split(X, y_bin, s, test_size=0.3, stratify=y_bin, random_state=RNG)

bin_logreg = LogisticRegression(max_iter=1000, random_state=RNG)
bin_logreg.fit(Xb_tr, yb_tr)
ypb_te = bin_logreg.predict(Xb_te)

# Fairness metrics
acc = accuracy_score(yb_te, ypb_te)
macro_f1 = f1_score(yb_te, ypb_te, average="macro")

def rate_positive(yhat):
    return yhat.mean() if len(yhat) else np.nan

def tpr(y_true, yhat):
    mask = y_true == 1
    if mask.sum() == 0:
        return np.nan
    return (yhat[mask] == 1).mean()

def spd(yhat, s):
    return rate_positive(yhat[s == 1]) - rate_positive(yhat[s == 0])

def eod(y_true, yhat, s):
    return tpr(y_true[s == 1], yhat[s == 1]) - tpr(y_true[s == 0], yhat[s == 0])

print("\nBinary fairness analysis (class 0 vs rest)")
print("Accuracy:", acc)
print("Macro-F1:", macro_f1)
print("SPD:", spd(ypb_te, sb_te))
print("EOD:", eod(yb_te, ypb_te, sb_te))

fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(["s=0", "s=1"], [rate_positive(ypb_te[sb_te==0]), rate_positive(ypb_te[sb_te==1])])
ax.set_ylim(0, 1)
ax.set_title("Positive prediction rates by sensitive group")
plt.show()
