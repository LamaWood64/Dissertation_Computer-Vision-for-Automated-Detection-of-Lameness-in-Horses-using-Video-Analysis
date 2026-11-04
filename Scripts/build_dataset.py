#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 20:02:52 2026

@author: anastasia
"""

import shutil
from pathlib import Path

PROJECT = Path("")
OUTPUT = PROJECT / "dataset"

NUM_KEYPOINTS = 12

EXPORTS = [
    {
        "images": PROJECT / "original_frames/Healthy/trot1-10",
        "labels": PROJECT / "cvat_exports/DisserHorseH1/labels/train",
        "class": "Healthy",
        "first_video": 1,
    },
    {
        "images": PROJECT / "original_frames/Healthy/trot11-15",
        "labels": PROJECT / "cvat_exports/DisserHorseH2/labels/train",
        "class": "Healthy",
        "first_video": 11,
    },
    {
        "images": PROJECT / "original_frames/Lame/trotsL_1-10",
        "labels": PROJECT / "cvat_exports/DisserHorseL/labels/train",
        "class": "Lame",
        "first_video": 1,
    },
]

TRAIN_VIDEOS = set(range(1, 13))
VAL_VIDEOS = {13, 14}
TEST_VIDEOS = {15}

EXPECTED_VALUES = 5 + 14 * 3

if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

for split in ("train", "val", "test"):
    (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)

stats = {
    "train": 0,
    "val": 0,
    "test": 0,
}

errors = {
    "missing_images": 0,
    "invalid_labels": 0,
    "out_of_range": 0,
}

image_counter = 1


def get_video_number(filename, first_video):
    stem = Path(filename).stem

    if stem.startswith("frame_"):
        return first_video

    return int(stem.split("_")[0])


def validate_label(path):
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) != 1:
        return False, False

    values = list(map(float, lines[0].split()))

    if len(values) != EXPECTED_VALUES:
        return False, False

    for value in values[1:5]:
        if not 0 <= value <= 1:
            return True, False

    keypoints = values[5:]

    for i in range(0, len(keypoints), 3):
        x = keypoints[i]
        y = keypoints[i + 1]
        v = keypoints[i + 2]

        if not 0 <= x <= 1:
            return True, False

        if not 0 <= y <= 1:
            return True, False

        if v not in (0, 1, 2):
            return True, False

    return True, True


def prepare_label(path):
    with open(path) as f:
        values = f.readline().split()

    header = values[:5]
    keypoints = values[5:]

    if NUM_KEYPOINTS == 12:
        keypoints = keypoints[:36]
    elif NUM_KEYPOINTS == 14:
        keypoints = keypoints[:42]
    else:
        raise ValueError("NUM_KEYPOINTS must be either 12 or 14")

    return " ".join(header + keypoints) + "\n"

for export in EXPORTS:

    image_dir = export["images"]
    label_dir = export["labels"]

    if not image_dir.exists():
        raise FileNotFoundError(image_dir)

    if not label_dir.exists():
        raise FileNotFoundError(label_dir)

    print(f"\nProcessing {export['class']}")

    for label in sorted(label_dir.glob("*.txt")):

        image = image_dir / f"{label.stem}.jpg"

        if not image.exists():
            errors["missing_images"] += 1
            print(f"Missing image: {image.name}")
            continue

        valid_structure, valid_coordinates = validate_label(label)

        if not valid_structure:
            errors["invalid_labels"] += 1
            print(f"Invalid label: {label.name}")
            continue

        if not valid_coordinates:
            errors["out_of_range"] += 1
            print(f"Coordinates outside range: {label.name}")
            continue

        video = get_video_number(label.name, export["first_video"])

        if video in TRAIN_VIDEOS:
            split = "train"
        elif video in VAL_VIDEOS:
            split = "val"
        elif video in TEST_VIDEOS:
            split = "test"
        else:
            raise ValueError(f"Unknown video {video}")

        prefix = "H" if export["class"] == "Healthy" else "L"
        new_name = f"{prefix}_{image_counter:06d}"

        shutil.copy2(
            image,
            OUTPUT / "images" / split / f"{new_name}.jpg"
        )

        with open(
            OUTPUT / "labels" / split / f"{new_name}.txt",
            "w"
        ) as f:
            f.write(prepare_label(label))

        image_counter += 1
        stats[split] += 1


if NUM_KEYPOINTS == 12:
    flip_idx = [2, 3, 0, 1, 6, 7, 4, 5, 8, 9, 10, 11]
elif NUM_KEYPOINTS == 14:
    flip_idx = [2, 3, 0, 1, 6, 7, 4, 5, 8, 9, 10, 11, 12, 13]
else:
    raise ValueError("NUM_KEYPOINTS must be either 12 or 14")


yaml_text = f"""path: {OUTPUT}

train: images/train
val: images/val
test: images/test

kpt_shape: [{NUM_KEYPOINTS}, 3]

flip_idx: {flip_idx}

names:
  0: horse
"""

with open(OUTPUT / "data.yaml", "w") as f:
    f.write(yaml_text)


print("\n========== DATASET CREATED ==========")
print(f"Train: {stats['train']}")
print(f"Validation: {stats['val']}")
print(f"Test: {stats['test']}")
print(f"Total: {sum(stats.values())}")

print("\n========== CHECKS ==========")
print(f"Missing images: {errors['missing_images']}")
print(f"Invalid labels: {errors['invalid_labels']}")
print(f"Coordinates outside [0,1]: {errors['out_of_range']}")

if sum(errors.values()) == 0:
    print("\nDataset validation passed.")
else:
    print("\nDataset validation failed.")

print(f"\nSaved to:\n{OUTPUT}")
