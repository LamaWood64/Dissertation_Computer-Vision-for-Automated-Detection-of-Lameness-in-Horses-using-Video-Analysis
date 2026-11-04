from pathlib import Path

DATASET = Path("/Users/anastasia/Desktop/HorseLamenessProject/dataset")

def check_labels(labels_dir):
    bad = []

    for txt in sorted(labels_dir.glob("*.txt")):
        vals = list(map(float, txt.read_text().split()))

        invalid = []

        # bbox: cx cy w h
        for i in range(1, 5):
            if not (0 <= vals[i] <= 1):
                invalid.append(vals[i])

        # keypoints: x y v
        idx = 5
        while idx + 2 < len(vals):
            x = vals[idx]
            y = vals[idx + 1]
            # v = vals[idx + 2]  <-- НЕ проверяем!

            if not (0 <= x <= 1):
                invalid.append(x)

            if not (0 <= y <= 1):
                invalid.append(y)

            idx += 3

        if invalid:
            bad.append((txt.name, invalid))

    return bad


total = 0

for split in ["train", "val", "test"]:
    labels = DATASET / "labels" / split
    bad = check_labels(labels)

    print(f"\n{split}: {len(bad)} corrupt labels")

    for name, vals in bad:
        print(name)
        print(vals)

    total += len(bad)

print("\nTotal:", total)