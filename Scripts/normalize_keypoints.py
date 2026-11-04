#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:42:46 2026

@author: anastasia
"""

from pathlib import Path
import pandas as pd

PROJECT = Path("/Users/anastasia/Desktop/HorseLamenessProject")

INPUT_DIR = PROJECT / "predictions/keypoints"
OUTPUT_DIR = PROJECT / "predictions/normalized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KEYPOINTS = [
    "lh_hoof",
    "lh_knee",
    "rh_hoof",
    "rh_knee",
    "rf_hoof",
    "rf_knee",
    "lf_hoof",
    "lf_knee",
    "withers",
    "pelvis",
    "neck",
    "nose",
]

files = sorted(INPUT_DIR.glob("*_keypoints.csv"))

print(f"Found {len(files)} files")

for file in files:

    print(f"\nProcessing: {file.name}")

    df = pd.read_csv(file)

    df["body_center_x"] = (
        df["withers_x"] + df["pelvis_x"]
    ) / 2

    df["body_center_y"] = (
        df["withers_y"] + df["pelvis_y"]
    ) / 2

    for kp in KEYPOINTS:

        df[f"{kp}_rel_x"] = (
            df[f"{kp}_x"] - df["body_center_x"]
        )

        df[f"{kp}_rel_y"] = (
            df[f"{kp}_y"] - df["body_center_y"]
        )

    output_file = OUTPUT_DIR / file.name.replace(
        "_keypoints.csv",
        "_keypoints_normalized.csv",
    )

    df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")

print("\nDone.")