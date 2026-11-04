#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 17:56:50 2026

@author: anastasia
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind


PROJECT = Path("/Users/anastasia/Desktop/HorseLamenessProject")

INPUT = PROJECT / "predictions/features.csv"

OUTPUT = PROJECT / "predictions/feature_analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(INPUT)

print("\n========== DATASET ==========\n")

print(df.shape)

print("\nColumns:", len(df.columns))

print("\nHealthy:", (df["label"] == "Healthy").sum())
print("Lame   :", (df["label"] == "Lame").sum())


print("\n========== MISSING VALUES ==========\n")

missing = df.isna().sum()

missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values.")
else:
    print(missing)


features = df.drop(columns=["video", "label"])


print("\n========== DESCRIPTIVE STATISTICS ==========\n")

print(features.describe())


variance = features.var().sort_values(ascending=False)

variance.to_csv(
    OUTPUT / "feature_variance.csv"
)


plt.figure(figsize=(12, 6))

variance.head(20).plot.bar()

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    OUTPUT / "top20_variance.png",
    dpi=300
)

plt.close()


corr = features.corr()

plt.figure(figsize=(14, 12))

plt.imshow(corr)

plt.colorbar()

plt.tight_layout()

plt.savefig(
    OUTPUT / "correlation_matrix.png",
    dpi=300
)

plt.close()


healthy = df[df.label == "Healthy"]

lame = df[df.label == "Lame"]


results = []


for feature in features.columns:

    x1 = healthy[feature]

    x2 = lame[feature]

    t_stat, p_value = ttest_ind(
        x1,
        x2,
        equal_var=False,
        nan_policy="omit"
    )

    mean1 = x1.mean()
    mean2 = x2.mean()

    pooled_std = np.sqrt(
        (
            x1.std() ** 2 +
            x2.std() ** 2
        ) / 2
    )

    if pooled_std == 0:
        effect = 0
    else:
        effect = abs(mean1 - mean2) / pooled_std

    results.append({

        "feature": feature,

        "healthy_mean": mean1,

        "lame_mean": mean2,

        "difference": abs(mean1 - mean2),

        "cohens_d": effect,

        "p_value": p_value

    })


stats = pd.DataFrame(results)

stats = stats.sort_values(
    by=[
        "cohens_d",
        "difference"
    ],
    ascending=False
)

stats.to_csv(
    OUTPUT / "feature_statistics.csv",
    index=False
)


print("\n========== TOP 20 FEATURES ==========\n")

print(
    stats[
        [
            "feature",
            "cohens_d",
            "p_value"
        ]
    ].head(20)
)


TOP = 12


for feature in stats.feature.head(TOP):

    plt.figure(figsize=(6, 4))

    data = [
        healthy[feature],
        lame[feature]
    ]

    plt.boxplot(
        data,
        tick_labels=[
            "Healthy",
            "Lame"
        ]
    )

    plt.title(feature)

    plt.tight_layout()

    plt.savefig(
        OUTPUT / f"{feature}.png",
        dpi=300
    )

    plt.close()


counts = df.label.value_counts()

plt.figure(figsize=(5, 4))

counts.plot.bar()

plt.tight_layout()

plt.savefig(
    OUTPUT / "class_distribution.png",
    dpi=300
)

plt.close()


print()

print("=" * 60)

print("Feature exploration completed successfully.")

print("=" * 60)

print()

print(f"Dataset size : {df.shape}")

print(f"Features     : {features.shape[1]}")

print()

print("Top discriminative features:")

print(
    stats[
        [
            "feature",
            "cohens_d",
            "p_value"
        ]
    ].head(10)
)

print()

print("Saved to:")

print(OUTPUT)