#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 2026

@author: anastasia
"""

from ultralytics import YOLO

PROJECT = ""

model = YOLO("yolo11n-pose.pt")

results = model.train(
    data=f"{PROJECT}/dataset/data.yaml",

    epochs=200,
    patience=30,

    imgsz=640,
    batch=16,

    device="mps",
    workers=4,
    cache=True,

    optimizer="auto",

    project=f"{PROJECT}/runs",
    name="horse_pose_12kp",
    exist_ok=False,

    seed=42,
    deterministic=True,

    save=True,
    save_period=10,

    plots=True,
    verbose=True
)
