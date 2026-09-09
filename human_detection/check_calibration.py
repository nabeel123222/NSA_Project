from pathlib import Path
import numpy as np

from features import extract_feature_vector


CALIBRATION_DIR = Path("calibration")
SETUPS = ["set_A", "set_B", "set_C"]
LABELS = ["empty", "present"]


def load_features(setup, label):
    folder = CALIBRATION_DIR / setup / label
    files = sorted(folder.glob("*.npy"))

    features = []

    for file in files:
        matrix = np.load(file)

        if matrix.shape != (50, 64):
            print(f"Skipping {file}: shape={matrix.shape}")
            continue

        if not np.isfinite(matrix).all():
            print(f"Skipping {file}: contains invalid values")
            continue

        vector = extract_feature_vector(matrix)

        if np.isfinite(vector).all():
            features.append(vector)

    return np.asarray(features)


print("=" * 70)
print("CALIBRATION FEATURE DIAGNOSTIC")
print("=" * 70)

all_empty = []
all_present = []

for setup in SETUPS:

    print()
    print("-" * 70)
    print(setup.upper())
    print("-" * 70)

    empty = load_features(setup, "empty")
    present = load_features(setup, "present")

    print(f"EMPTY samples   : {len(empty)}")
    print(f"PRESENT samples : {len(present)}")

    if len(empty) == 0 or len(present) == 0:
        continue

    empty_mean = np.mean(empty, axis=0)
    present_mean = np.mean(present, axis=0)

    separation = np.mean(np.abs(present_mean - empty_mean))

    print(f"Mean feature separation: {separation:.6f}")

    print(
        "Mean motion features - "
        f"EMPTY={np.mean(empty[:, 192:256]):.4f}, "
        f"PRESENT={np.mean(present[:, 192:256]):.4f}"
    )

    print(
        "Total motion score - "
        f"EMPTY={np.mean(empty[:, 257]):.4f}, "
        f"PRESENT={np.mean(present[:, 257]):.4f}"
    )

    all_empty.append(empty)
    all_present.append(present)


if all_empty and all_present:

    empty = np.vstack(all_empty)
    present = np.vstack(all_present)

    print()
    print("=" * 70)
    print("ALL CALIBRATION SETS")
    print("=" * 70)

    print(f"Total EMPTY   : {len(empty)}")
    print(f"Total PRESENT : {len(present)}")

    empty_mean = np.mean(empty, axis=0)
    present_mean = np.mean(present, axis=0)

    difference = np.abs(present_mean - empty_mean)

    print(
        f"Overall mean feature separation: "
        f"{np.mean(difference):.6f}"
    )

    top_indices = np.argsort(difference)[-10:][::-1]

    print()
    print("Top 10 class-separating features:")

    for index in top_indices:
        print(
            f"Feature {index:3d}: "
            f"EMPTY={empty_mean[index]:.4f} | "
            f"PRESENT={present_mean[index]:.4f} | "
            f"Difference={difference[index]:.4f}"
        )

print()
print("=" * 70)
print("DIAGNOSTIC COMPLETED")
print("=" * 70)
