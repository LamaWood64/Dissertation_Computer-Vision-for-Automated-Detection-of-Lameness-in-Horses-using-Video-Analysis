#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 18:01:45 2026

@author: anastasia
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,)
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel


PROJECT = Path("/Users/anastasia/Desktop/HorseLamenessProject")

INPUT = PROJECT / "predictions/features.csv"

OUTPUT = PROJECT / "classifier"
OUTPUT.mkdir(parents=True, exist_ok=True)

MODEL_FILE = OUTPUT / "random_forest.pkl"

TOP_FEATURES = 20

df = pd.read_csv(INPUT)

y = df["label"].map({
    "Healthy": 0,
    "Lame": 1,
})

limb_prefixes = (
    "lf_knee",
    "rf_knee",
    "lh_knee",
    "rh_knee",
    "lf_hoof",
    "rf_hoof",
    "lh_hoof",
    "rh_hoof",
)

limb_features = [
    column
    for column in df.columns
    if column.startswith(limb_prefixes)
]

X = df[limb_features]

feature_names = X.columns

X = X.to_numpy()
y = y.to_numpy()

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

accuracies = []
precisions = []
recalls = []
f1_scores = []

feature_importance = np.zeros(len(feature_names))

all_true = []
all_pred = []
all_prob = []

fold = 1

for train_idx, test_idx in cv.split(X, y):

    print(f"\n========== Fold {fold} ==========\n")

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    selector = SelectFromModel(
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
        ),
        max_features=TOP_FEATURES,
        threshold=-np.inf,
    )

    selector.fit(X_train, y_train)

    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
    )

    model.fit(
        X_train_selected,
        y_train,
    )

    y_pred = model.predict(X_test_selected)

    y_prob = model.predict_proba(
        X_test_selected,
    )[:, 1]

    all_true.extend(y_test)
    all_pred.extend(y_pred)
    all_prob.extend(y_prob)

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

    selected = selector.get_support()

    feature_importance[selected] += model.feature_importances_

    print(f"Selected features: {selected.sum()}")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")

    fold += 1

feature_importance /= cv.get_n_splits()

print("\n========== Cross validation results ==========\n")

print(f"Accuracy : {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
print(f"Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
print(f"Recall   : {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
print(f"F1-score : {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")

selector = SelectFromModel(
    RandomForestClassifier(
        n_estimators=300,
        random_state=42,
    ),
    max_features=TOP_FEATURES,
    threshold=-np.inf,
)

selector.fit(X, y)

X_selected = selector.transform(X)

selected_features = feature_names[
    selector.get_support()
]

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
)

model.fit(
    X_selected,
    y,
)

joblib.dump(
    model,
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

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance,
})

importance = importance.sort_values(
    by="Importance",
    ascending=False,
)

importance.to_csv(
    OUTPUT / "feature_importance.csv",
    index=False,
)

top20 = importance.head(20)

plt.figure(figsize=(10, 8))

plt.barh(
    top20["Feature"],
    top20["Importance"],
)

plt.gca().invert_yaxis()

plt.xlabel("Feature importance")

plt.title("Top 20 most important features")

plt.tight_layout()

plt.savefig(
    OUTPUT / "feature_importance.png",
    dpi=300,
)

plt.close()

ConfusionMatrixDisplay.from_predictions(
    all_true,
    all_pred,
    display_labels=["Healthy", "Lame"],
    cmap="Blues",
)

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

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_selected)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

shap.summary_plot(
    shap_values,
    X_selected,
    feature_names=selected_features,
    show=False,
)

plt.tight_layout()

plt.savefig(
    OUTPUT / "shap_summary.png",
    dpi=300,
)

plt.close()

shap.summary_plot(
    shap_values,
    X_selected,
    feature_names=selected_features,
    plot_type="bar",
    show=False,
)

plt.tight_layout()

plt.savefig(
    OUTPUT / "shap_bar.png",
    dpi=300,
)

plt.close()

print(f"\nRoc AUC: {roc_auc:.3f}")

print("\n========== Top 20 features ==========\n")

print(top20)

print("\nModel saved to:")
print(MODEL_FILE)

print("\nResults saved to:")
print(OUTPUT)