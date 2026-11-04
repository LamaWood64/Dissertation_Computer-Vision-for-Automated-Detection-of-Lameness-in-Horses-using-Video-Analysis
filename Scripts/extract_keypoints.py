#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:29:26 2026

@author: anastasia
"""

from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO


PROJECT = Path("/Users/anastasia/Desktop/HorseLamenessProject")

MODEL = PROJECT / "runs/horse_pose_12kp/weights/best.pt"

VIDEO_DIRS = {
    "Healthy": PROJECT / "dataset/videos/Videos No Lamleness",
    "Lame": PROJECT / "dataset/videos/Videos lamleness",
}

OUTPUT_DIR = PROJECT / "predictions/keypoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL)

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

columns = ["frame"]

for kp in KEYPOINTS:
    columns.extend([
        f"{kp}_x",
        f"{kp}_y",
        f"{kp}_conf",
    ])

videos = []

for label, directory in VIDEO_DIRS.items():
    for i, video in enumerate(sorted(directory.glob("*.mov")), start=1):
        videos.append((video, label, i))

print(f"Found {len(videos)} videos")

for video, label, number in videos:
    
    print(f"\nProcessing: {video.name}")

    cap = cv2.VideoCapture(str(video))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Frames : {frame_count}")
    print(f"FPS    : {fps:.2f}")

    rows = []
    frame_id = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        result = model(frame, verbose=False)[0]

        row = [frame_id]

        if result.keypoints is not None and len(result.keypoints) > 0:

            xy = result.keypoints.xy.cpu().numpy()[0]
            conf = result.keypoints.conf.cpu().numpy()[0]

            for i in range(len(KEYPOINTS)):
                row.extend([
                    float(xy[i][0]),
                    float(xy[i][1]),
                    float(conf[i]),
                ])

        else:

            for _ in KEYPOINTS:
                row.extend([None, None, None])

        rows.append(row)

        frame_id += 1

    cap.release()

    df = pd.DataFrame(rows, columns=columns)
    df["label"] = label

    output_file = OUTPUT_DIR / f"{label}_{number:02d}_keypoints.csv"

    df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")

print("\nDone.")