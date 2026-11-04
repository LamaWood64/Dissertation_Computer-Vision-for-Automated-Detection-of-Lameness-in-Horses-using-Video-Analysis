#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 18:18:52 2026

@author: anastasia
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT = Path("")
INPUT = PROJECT / "predictions/features.csv"
OUTPUT = PROJECT / "classifier_svm"
OUTPUT.mkdir(parents=True, exist_ok=True)
MODEL_FILE = OUTPUT / "svm.pkl"

df = pd.read_csv(INPUT)
X = df.drop(columns=["video", "label"])
y = df["label"].map({
    "Healthy": 0,
    "Lame": 1,})

feature_names = X.columns
X = X.values
y = y.values

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=42,)),])

accuracies = []
precisions = []
recalls = []
f1_scores = []

all_true = []
all_pred = []
all_prob = []

fold = 1

for train_idx, test_idx in cv.split(X, y):

    print(f"\n Fold {fold} \n")

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    pipeline.fit(
        X_train,
        y_train,
    )
    print("\nClasses:")
    print(pipeline.classes_)
    
    y_pred = pipeline.predict(X_test)

    y_score = pipeline.decision_function(X_test)

    all_true.extend(y_test)
    all_pred.extend(y_pred)
    all_prob.extend(y_score)

    acc = accuracy_score(
        y_test,
        y_pred,
    )

    prec = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    rec = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)

    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")

    fold += 1

print("\n Cross validation results \n")

print(f"Accuracy : {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
print(f"Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
print(f"Recall   : {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
print(f"F1-score : {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")

pipeline.fit(
    X,
    y,
)

joblib.dump(
    pipeline,
    MODEL_FILE,
)

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
    ],
    "Mean": [
        np.mean(accuracies),
        np.mean(precisions),
        np.mean(recalls),
        np.mean(f1_scores),
    ],
    "Std": [
        np.std(accuracies),
        np.std(precisions),
        np.std(recalls),
        np.std(f1_scores),
    ],
})

metrics.to_csv(
    OUTPUT / "cross_validation_results.csv",
    index=False,
)

ConfusionMatrixDisplay.from_predictions(
    all_true,
    all_pred,
    display_labels=["Healthy", "Lame"],
    cmap="Blues",
)

print(pipeline.classes_)

plt.title("Confusion matrix")

plt.tight_layout()

plt.savefig(
    OUTPUT / "confusion_matrix.png",
    dpi=300,
)

plt.close()

fpr, tpr, _ = roc_curve(
    all_true,
    all_prob,
)

roc_auc = auc(
    fpr,
    tpr,
)

plt.figure(figsize=(6, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {roc_auc:.3f}",
)

plt.plot(
    [0, 1],
    [0, 1],
    "--",
)

plt.xlabel("False positive rate")

plt.ylabel("True positive rate")

plt.title("Roc curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT / "roc_curve.png",
    dpi=300,
)

plt.close()

print(f"\nRoc AUC: {roc_auc:.3f}")

print("\nModel saved to:")
print(MODEL_FILE)

print("\nResults saved to:")
print(OUTPUT)
