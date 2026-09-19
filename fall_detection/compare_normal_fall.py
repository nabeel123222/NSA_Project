import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "human_detection"))

from fall_features import extract_two_receiver_fall_features


DATASET_DIR = Path(__file__).resolve().parent / "dataset"

NORMAL_DIR = DATASET_DIR / "normal"
FALL_DIR = DATASET_DIR / "fall"


FEATURE_NAMES = [
    "Friend Peak Motion",
    "Friend Motion Jerk",
    "Friend Stillness",
    "You Peak Motion",
    "You Motion Jerk",
    "You Stillness",
]


def load_dataset(folder, pattern):

    features = []
    names = []

    files = sorted(folder.glob(pattern))

    for file in files:

        try:

            data = np.load(file)

            friend_csi = data["friend_csi"]
            you_csi = data["you_csi"]

            friend_rate = float(data["friend_rate"])
            you_rate = float(data["you_rate"])

            friend_amp = np.abs(friend_csi)
            you_amp = np.abs(you_csi)

            n = min(
                len(friend_amp),
                len(you_amp)
            )

            if n < 30:
                continue

            friend_amp = friend_amp[:n]
            you_amp = you_amp[:n]

            f = extract_two_receiver_fall_features(
                friend_amp,
                you_amp,
                friend_rate,
                you_rate
            )

            if len(f) != 6:
                continue

            if not np.isfinite(f).all():
                continue

            features.append(f)
            names.append(file.name)

        except Exception as e:

            print(
                f"Skipping {file.name}: {e}"
            )

    return np.asarray(features), names


def print_statistics(title, X):

    print("\n" + "=" * 85)
    print(title)
    print("=" * 85)

    print(
        f"\n{'Feature':30s}"
        f"{'Mean':>12s}"
        f"{'Std':>12s}"
        f"{'Min':>12s}"
        f"{'Max':>12s}"
    )

    print("-" * 78)

    for i, name in enumerate(FEATURE_NAMES):

        values = X[:, i]

        print(
            f"{name:30s}"
            f"{np.mean(values):12.3f}"
            f"{np.std(values):12.3f}"
            f"{np.min(values):12.3f}"
            f"{np.max(values):12.3f}"
        )


def main():

    print("=" * 85)
    print("NSA - FALL DETECTION")
    print("NORMAL vs FALL FEATURE ANALYSIS")
    print("=" * 85)

    normal_X, normal_names = load_dataset(
        NORMAL_DIR,
        "normal_*.npz"
    )

    fall_X, fall_names = load_dataset(
        FALL_DIR,
        "fall_*.npz"
    )

    print("\nDataset counts:")
    print(f"  NORMAL valid samples : {len(normal_X)}")
    print(f"  FALL valid samples   : {len(fall_X)}")

    if len(normal_X) == 0 or len(fall_X) == 0:

        print("\nERROR: One dataset is empty.")
        return

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    print_statistics(
        "NORMAL FEATURES",
        normal_X
    )

    print_statistics(
        "FALL FEATURES",
        fall_X
    )

    # --------------------------------------------------
    # Difference between class means
    # --------------------------------------------------

    print("\n" + "=" * 85)
    print("NORMAL vs FALL - CLASS MEAN COMPARISON")
    print("=" * 85)

    print(
        f"\n{'Feature':30s}"
        f"{'NORMAL Mean':>15s}"
        f"{'FALL Mean':>15s}"
        f"{'Fall/Normal':>15s}"
    )

    print("-" * 78)

    for i, name in enumerate(FEATURE_NAMES):

        normal_mean = np.mean(normal_X[:, i])
        fall_mean = np.mean(fall_X[:, i])

        if abs(normal_mean) > 1e-12:

            ratio = fall_mean / normal_mean

        else:

            ratio = np.inf

        print(
            f"{name:30s}"
            f"{normal_mean:15.3f}"
            f"{fall_mean:15.3f}"
            f"{ratio:15.2f}"
        )

    # --------------------------------------------------
    # Simple separation score
    # --------------------------------------------------

    print("\n" + "=" * 85)
    print("FEATURE SEPARATION")
    print("=" * 85)

    print(
        "\nA larger value means the two class distributions "
        "are more separated."
    )

    print(
        f"\n{'Feature':30s}"
        f"{'Separation':>15s}"
    )

    print("-" * 50)

    for i, name in enumerate(FEATURE_NAMES):

        normal_mean = np.mean(normal_X[:, i])
        fall_mean = np.mean(fall_X[:, i])

        normal_std = np.std(normal_X[:, i])
        fall_std = np.std(fall_X[:, i])

        pooled_std = np.sqrt(
            (
                normal_std ** 2
                +
                fall_std ** 2
            ) / 2
        )

        if pooled_std > 1e-12:

            separation = (
                abs(fall_mean - normal_mean)
                / pooled_std
            )

        else:

            separation = 0.0

        print(
            f"{name:30s}"
            f"{separation:15.3f}"
        )

    # --------------------------------------------------
    # Display all FALL samples
    # --------------------------------------------------

    print("\n" + "=" * 85)
    print("FALL SAMPLE VALUES")
    print("=" * 85)

    for name, f in zip(fall_names, fall_X):

        print(
            f"\n{name}"
        )

        print(
            f"  Friend: "
            f"peak={f[0]:.3f}, "
            f"jerk={f[1]:.3f}, "
            f"stillness={f[2]:.3f}"
        )

        print(
            f"  You:    "
            f"peak={f[3]:.3f}, "
            f"jerk={f[4]:.3f}, "
            f"stillness={f[5]:.3f}"
        )

    print("\n" + "=" * 85)
    print("ANALYSIS COMPLETE")
    print("=" * 85)


if __name__ == "__main__":
    main()
