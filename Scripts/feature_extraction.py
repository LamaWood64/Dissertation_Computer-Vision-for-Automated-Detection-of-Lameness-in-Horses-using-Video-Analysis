#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 16:55:19 2026

@author: anastasia
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path("")
INPUT_DIR = PROJECT / "predictions/smoothed"
OUTPUT_FILE = PROJECT / "predictions/features.csv"
KEYPOINTS = [
    "nose",
    "neck",
    "withers",
    "pelvis",
    "lf_knee",
    "rf_knee",
    "lh_knee",
    "rh_knee",
    "lf_hoof",
    "rf_hoof",
    "lh_hoof",
    "rh_hoof",]

def extract_features(signal: pd.Series) -> dict:

    signal = signal.dropna().to_numpy(dtype=float)

    if len(signal) < 2:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "range": np.nan,
            "mean_velocity": np.nan,
            "std_velocity": np.nan,
            "max_velocity": np.nan,}

    velocity = np.diff(signal)
    features = {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "min": np.min(signal),
        "max": np.max(signal),
        "median": np.median(signal),
        "range": np.max(signal) - np.min(signal),
        "mean_velocity": np.mean(np.abs(velocity)),
        "std_velocity": np.std(velocity),
        "max_velocity": np.max(np.abs(velocity)),}

    return features

rows = []
files = sorted(INPUT_DIR.glob("*_keypoints_smoothed.csv"))

for file in files:
    print(f"Processing: {file.name}")
    df = pd.read_csv(file)
    row = {}
    video_name = file.stem.replace("_keypoints_smoothed", "")
    row["video"] = video_name
    row["label"] = df["label"].iloc[0]

    for kp in KEYPOINTS:
        for axis in ["x", "y"]:
            column = f"{kp}_smooth_{axis}"

            if column not in df.columns:
                print(f"Warning: {column} not found.")
                continue

            features = extract_features(df[column])
            for feature_name, value in features.items():
                row[f"{kp}_{axis}_{feature_name}"] = value
    rows.append(row)
    
features_df = pd.DataFrame(rows)
features_df = features_df.sort_values("video").reset_index(drop=True)
features_df.to_csv(OUTPUT_FILE, index=False)

print()
print("=" * 60)
print("Feature extraction completed successfully.")
print("=" * 60)
print()

print(f"Videos processed: {len(features_df)}")
print(f"Features per video: {features_df.shape[1] - 2}")
print()

print("Output file:")
print(OUTPUT_FILE)
print()

print("Preview:")
print(features_df.head())
