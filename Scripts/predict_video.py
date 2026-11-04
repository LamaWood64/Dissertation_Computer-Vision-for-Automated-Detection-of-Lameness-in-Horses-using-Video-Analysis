#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:28:23 2026

@author: anastasia
"""

from pathlib import Path
from ultralytics import YOLO

PROJECT = Path("/Users/anastasia/Desktop/HorseLamenessProject")

MODEL = PROJECT / "runs/horse_pose_12kp/weights/best.pt"
VIDEO = PROJECT / "videos/test1.mp4"
OUTPUT = PROJECT / "results"

model = YOLO(MODEL)

model.predict(
    source=VIDEO,

    imgsz=640,
    device="mps",

    save=True,
    save_txt=False,
    save_conf=False,

    show=False,

    project=OUTPUT,
    name="pose_prediction",
    exist_ok=True,

    verbose=True,
)

print("Done.")