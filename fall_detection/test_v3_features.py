import numpy as np
from pathlib import Path

from fall_features_v3 import extract_two_receiver_features


BASE = Path(__file__).resolve().parent / "dataset"

NORMAL_DIR = BASE / "normal_set2"
FALL_DIR = BASE / "fall"

FEATURE_NAMES = [
    "Friend Peak Ratio",
    "Friend Event Energy",
    "Friend Post Drop",
    "You Peak Ratio",
    "You Event Energy",
    "You Post Drop",
]


def load_dataset(folder):

    features = []

    for file in sorted(folder.glob("*.npz")):

        data = np.load(file)

        friend_amp = np.abs(data["friend_csi"])
        you_amp = np.abs(data["you_csi"])

        n = min(
            len(friend_amp),
            len(you_amp)
        )

        if n < 30:
            continue

        f = extract_two_receiver_features(
            friend_amp[:n],
            you_amp[:n]
        )

        if np.isfinite(f).all():
            features.append(f)

    return np.asarray(features)


def main():

    print("=" * 80)
    print("NSA - FALL DETECTION")
    print("V3 FEATURE SEPARATION TEST")
    print("=" * 80)

    normal = load_dataset(NORMAL_DIR)
    fall = load_dataset(FALL_DIR)

    print("\nDataset:")
    print(f"  NORMAL Set 2 : {len(normal)}")
    print(f"  FALL         : {len(fall)}")

    print("\n" + "=" * 80)
    print("FEATURE STATISTICS")
    print("=" * 80)

    print(
        f"\n{'Feature':28s}"
        f"{'NORMAL Mean':>15s}"
        f"{'FALL Mean':>15s}"
        f"{'Separation':>15s}"
    )

    print("-" * 75)

    for i, name in enumerate(FEATURE_NAMES):

        normal_mean = np.mean(normal[:, i])
        fall_mean = np.mean(fall[:, i])

        normal_std = np.std(normal[:, i])
        fall_std = np.std(fall[:, i])

        pooled_std = np.sqrt(
            (
                normal_std ** 2
                + fall_std ** 2
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
            f"{name:28s}"
            f"{normal_mean:15.4f}"
            f"{fall_mean:15.4f}"
            f"{separation:15.3f}"
        )

    print("\n" + "=" * 80)
    print("V3 TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
