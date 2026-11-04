#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:56:56 2026

@author: anastasia
"""

from pathlib import Path
import pandas as pd
from scipy.signal import savgol_filter

PROJECT = Path("/")

INPUT_DIR = PROJECT / "predictions/normalized"
OUTPUT_DIR = PROJECT / "predictions/smoothed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_LENGTH = 11
POLYORDER = 2

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
    "nose",]

files = sorted(INPUT_DIR.glob("*_keypoints_normalized.csv"))

print(f"Found {len(files)} files")

for file in files:

    print(f"\nProcessing: {file.name}")

    df = pd.read_csv(file)

    for kp in KEYPOINTS:

        df[f"{kp}_smooth_x"] = savgol_filter(
            df[f"{kp}_rel_x"],
            window_length=WINDOW_LENGTH,
            polyorder=POLYORDER,
            mode="interp",
        )

        df[f"{kp}_smooth_y"] = savgol_filter(
            df[f"{kp}_rel_y"],
            window_length=WINDOW_LENGTH,
            polyorder=POLYORDER,
            mode="interp",
        )

    output_file = OUTPUT_DIR / file.name.replace(
        "_keypoints_normalized.csv",
        "_keypoints_smoothed.csv",
    )

    df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")

print("\nDone.")
