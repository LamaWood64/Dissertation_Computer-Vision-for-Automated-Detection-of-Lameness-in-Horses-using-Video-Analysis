#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 16:41:50 2026

@author: anastasia
"""

from pathlib import Path
import numpy as np
import pandas as pd

# =============================================================================
# Paths
# =============================================================================

PROJECT = Path("/Users/anastasia/Desktop/HorseLamenessProject")

PEAKS_FILE = PROJECT / "predictions/test_peaks.csv"

OUTPUT_FILE = PROJECT / "predictions/gait_features.csv"

FPS = 30  # change if your videos use another frame rate

# =============================================================================
# Load
# =============================================================================

peaks = pd.read_csv(PEAKS_FILE)

# =============================================================================
# Hooves
# =============================================================================

HOOFS = [
    "lf_hoof",
    "rf_hoof",
    "lh_hoof",
    "rh_hoof",
]

features = []

# =============================================================================
# Process each hoof
# =============================================================================

for hoof in HOOFS:

    hoof_data = peaks[peaks["hoof"] == hoof].copy()

    maxima = (
        hoof_data[hoof_data["type"] == "peak"]
        .sort_values("frame")
        .reset_index(drop=True)
    )

    minima = (
        hoof_data[hoof_data["type"] == "valley"]
        .sort_values("frame")
        .reset_index(drop=True)
    )

    step_durations = []
    amplitudes = []

    # -------------------------------------------------------------
    # Each gait cycle = Peak -> Valley -> Peak
    # -------------------------------------------------------------

    for i in range(len(maxima) - 1):

        peak1 = maxima.iloc[i]
        peak2 = maxima.iloc[i + 1]

        valley = minima[
            (minima["frame"] > peak1["frame"]) &
            (minima["frame"] < peak2["frame"])
        ]

        if len(valley) != 1:
            continue

        valley = valley.iloc[0]

        duration_frames = peak2["frame"] - peak1["frame"]

        amplitude = (
            (peak1["value"] + peak2["value"]) / 2
            - valley["value"]
        )

        step_durations.append(duration_frames / FPS)

        amplitudes.append(amplitude)

    step_durations = np.asarray(step_durations)
    amplitudes = np.asarray(amplitudes)

    features.append({

        "hoof": hoof,

        "num_steps": len(step_durations),

        "mean_step_duration_s":
            np.mean(step_durations) if len(step_durations) else np.nan,

        "std_step_duration_s":
            np.std(step_durations) if len(step_durations) else np.nan,

        "mean_amplitude":
            np.mean(amplitudes) if len(amplitudes) else np.nan,

        "std_amplitude":
            np.std(amplitudes) if len(amplitudes) else np.nan,

        "max_amplitude":
            np.max(amplitudes) if len(amplitudes) else np.nan,

        "min_amplitude":
            np.min(amplitudes) if len(amplitudes) else np.nan,

    })

# =============================================================================
# Save
# =============================================================================

features = pd.DataFrame(features)

features.to_csv(OUTPUT_FILE, index=False)

print()
print(features)
print()
print("Saved to:")
print(OUTPUT_FILE)