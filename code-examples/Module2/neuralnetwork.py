#!/usr/bin/env python3
"""
NLP neural network, end-to-end, with simple visuals.

What you get
- Real text dataset: 20 Newsgroups (3 classes for speed)
- Vectorization: TF–IDF (bag-of-words)
- Models: Perceptron (baseline), Logistic Regression (baseline), 1-hidden-layer MLP
- Charts: training loss curve, validation accuracy per epoch, confusion matrix,
          PCA scatter of documents colored by class
- Notes inline tie back to class intuition about "latent space" via nonlinearity

Run
  $ python nlp_nn_simple.py

Requirements
  pip install scikit-learn matplotlib numpy

"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

RNG = 7
np.random.seed(RNG)

# -----------------------------------------------------------------------------
# 1) Load a real NLP dataset: pick 3 topical classes for clarity and speed
# -----------------------------------------------------------------------------
CATS = [
    "rec.sport.baseball",
    "sci.med",
    "talk.politics.misc",
]
train = fetch_20newsgroups(subset="train", categories=CATS, remove=("headers","quotes"))
test  = fetch_20newsgroups(subset="test",  categories=CATS, remove=("headers","quotes"))

# -----------------------------------------------------------------------------
# 2) Vectorize with TF–IDF. Keep a modest vocab to stay fast and interpretable
# -----------------------------------------------------------------------------
VECT = TfidfVectorizer(min_df=3, max_df=0.6, ngram_range=(1,2), stop_words="english")
Xtr = VECT.fit_transform(train.data)
Xte = VECT.transform(test.data)
ytr = train.target
yte = test.target
class_names = train.target_names

# We will also hold out a validation set from training docs for live monitoring
Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.2, random_state=RNG, stratify=ytr)

# -----------------------------------------------------------------------------
# 3) Baselines: Perceptron and Logistic Regression
# -----------------------------------------------------------------------------
percep = Perceptron(max_iter=1000, tol=1e-3, random_state=RNG)
logreg = LogisticRegression(max_iter=200, multi_class="auto", n_jobs=None)
percep.fit(Xtr, ytr)
logreg.fit(Xtr, ytr)

base_preds = {
    "Perceptron": percep.predict(Xte),
    "LogReg": logreg.predict(Xte),
}

# -----------------------------------------------------------------------------
# 4) A simple neural network for text: MLP with one hidden layer
#    Nonlinearity maps sparse BoW features to a learned latent space Z, then a
#    linear classifier acts in Z (exactly the class intuition).
# -----------------------------------------------------------------------------
mlp = MLPClassifier(
    hidden_layer_sizes=(128,),
    activation="tanh",      # matches lecture intuition; try 'relu' too
    solver="adam",
    alpha=1e-4,
    learning_rate_init=5e-4,
    batch_size=64,
    max_iter=1,             # we will manual-loop for live val accuracy
    random_state=RNG,
    warm_start=True,
)

EPOCHS = 15
va_acc_hist = []
tr_acc_hist = []
for epoch in range(EPOCHS):
    mlp.fit(Xtr, ytr)
    yva_hat = mlp.predict(Xva)
    ytr_hat = mlp.predict(Xtr)
    va_acc_hist.append(accuracy_score(yva, yva_hat))
    tr_acc_hist.append(accuracy_score(ytr, ytr_hat))

# Final test evaluation
pred_mlp = mlp.predict(Xte)

# -----------------------------------------------------------------------------
# 5) Report numbers
# -----------------------------------------------------------------------------
print("Test accuracy:")
for name, yhat in base_preds.items():
    print(f"  {name:10s}: {accuracy_score(yte, yhat):.3f}")
print(f"  {'MLP':10s}: {accuracy_score(yte, pred_mlp):.3f}")

# -----------------------------------------------------------------------------
# 6) Visuals
# -----------------------------------------------------------------------------
plt.figure(figsize=(6,3.2))
plt.plot(tr_acc_hist, label="Train acc")
plt.plot(va_acc_hist, label="Val acc")
plt.title("MLP accuracy over epochs")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,3.2))
plt.plot(mlp.loss_curve_)
plt.title("MLP training loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.tight_layout()
plt.show()

# Confusion matrix for the MLP on the real test set
cm = confusion_matrix(yte, pred_mlp)
fig, ax = plt.subplots(figsize=(5.5,4.5))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", values_format="d")
plt.title("MLP confusion matrix (test)")
plt.tight_layout()
plt.show()

# PCA visualization of documents in 2D (unsupervised), colored by their class
pca = PCA(n_components=2, random_state=RNG)
# For PCA speed, project a subset if very large
Nvis = min(2000, Xte.shape[0])
idx = np.random.RandomState(RNG).choice(Xte.shape[0], size=Nvis, replace=False)
Xte_2d = pca.fit_transform(Xte[idx].toarray())
yte_sub = yte[idx]
plt.figure(figsize=(6,5))
for c in range(len(class_names)):
    mask = (yte_sub == c)
    plt.scatter(Xte_2d[mask,0], Xte_2d[mask,1], s=12, label=class_names[c], edgecolor="none")
plt.title("20NG (3 classes) — PCA of TF–IDF vectors")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(markerscale=1.5, frameon=True)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 7) Inspect top n-grams per class using logistic regression weights for intuition
# -----------------------------------------------------------------------------
def top_features_per_class(clf, vectorizer, k=12):
    terms = np.array(vectorizer.get_feature_names_out())
    W = clf.coef_  # shape [C, V]
    for c, cname in enumerate(class_names):
        topk = np.argsort(W[c])[-k:][::-1]
        print(f"\nTop features for class '{cname}':")
        print(", ".join(terms[topk]))

print("\nMost indicative n-grams per class (LogReg):")
top_features_per_class(logreg, VECT, k=12)

# -----------------------------------------------------------------------------
# 8) What to tweak (experiments)
# -----------------------------------------------------------------------------
print("\nTry:")
print("- activation='relu' and larger hidden_layer_sizes, watch loss and val accuracy")
print("- add dropout-ish effect via alpha (L2) and compare overfitting")
print("- use bigram-only or unigram-only ngrams to see feature effects")
print("- increase classes to 4-5 to stress the capacity and training time")
