import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "human_detection"))

from .fall_features import extract_two_receiver_fall_features


BASE = Path(__file__).resolve().parent / "dataset"

SETS = {
    "NORMAL SET 1": BASE / "normal",
    "NORMAL SET 2": BASE / "normal_set2",
    "FALL": BASE / "fall",
}

FEATURE_NAMES = [
    "Friend Peak Motion",
    "Friend Motion Jerk",
    "Friend Stillness",
    "You Peak Motion",
    "You Motion Jerk",
    "You Stillness",
]


def load_set(folder):
    X = []

    for file in sorted(folder.glob("*.npz")):
        try:
            data = np.load(file)

            friend_amp = np.abs(data["friend_csi"])
            you_amp = np.abs(data["you_csi"])

            n = min(len(friend_amp), len(you_amp))

            if n < 30:
                continue

            friend_amp = friend_amp[:n]
            you_amp = you_amp[:n]

            features = extract_two_receiver_fall_features(
                friend_amp,
                you_amp,
                float(data["friend_rate"]),
                float(data["you_rate"])
            )

            if len(features) == 6 and np.isfinite(features).all():
                X.append(features)

        except Exception as e:
            print(f"Skipping {file.name}: {e}")

    return np.asarray(X)


def main():

    print("=" * 90)
    print("NSA - FALL DETECTION")
    print("NORMAL SET 1 vs NORMAL SET 2 vs FALL")
    print("=" * 90)

    datasets = {}

    for name, folder in SETS.items():

        X = load_set(folder)
        datasets[name] = X

        print(f"\n{name}")
        print(f"  Samples: {len(X)}")

    print("\n" + "=" * 90)
    print("FEATURE STATISTICS")
    print("=" * 90)

    print(
        f"\n{'Feature':25s}"
        f"{'NORM1 Mean':>15s}"
        f"{'NORM2 Mean':>15s}"
        f"{'FALL Mean':>15s}"
    )

    print("-" * 75)

    for i, feature in enumerate(FEATURE_NAMES):

        n1 = datasets["NORMAL SET 1"][:, i]
        n2 = datasets["NORMAL SET 2"][:, i]
        fall = datasets["FALL"][:, i]

        print(
            f"{feature:25s}"
            f"{np.mean(n1):15.3f}"
            f"{np.mean(n2):15.3f}"
            f"{np.mean(fall):15.3f}"
        )

    print("\n" + "=" * 90)
    print("NORMAL SET 1 vs NORMAL SET 2")
    print("=" * 90)

    for i, feature in enumerate(FEATURE_NAMES):

        n1 = datasets["NORMAL SET 1"][:, i]
        n2 = datasets["NORMAL SET 2"][:, i]

        print(
            f"\n{feature}"
        )

        print(
            f"  Set 1 : "
            f"mean={np.mean(n1):.3f}, "
            f"std={np.std(n1):.3f}, "
            f"min={np.min(n1):.3f}, "
            f"max={np.max(n1):.3f}"
        )

        print(
            f"  Set 2 : "
            f"mean={np.mean(n2):.3f}, "
            f"std={np.std(n2):.3f}, "
            f"min={np.min(n2):.3f}, "
            f"max={np.max(n2):.3f}"
        )

    print("\n" + "=" * 90)
    print("FALL SEPARATION FROM COMBINED NORMAL")
    print("=" * 90)

    normal = np.vstack([
        datasets["NORMAL SET 1"],
        datasets["NORMAL SET 2"]
    ])

    fall = datasets["FALL"]

    for i, feature in enumerate(FEATURE_NAMES):

        normal_mean = np.mean(normal[:, i])
        fall_mean = np.mean(fall[:, i])

        normal_std = np.std(normal[:, i])
        fall_std = np.std(fall[:, i])

        pooled_std = np.sqrt(
            (normal_std ** 2 + fall_std ** 2) / 2
        )

        if pooled_std > 1e-12:
            separation = (
                abs(fall_mean - normal_mean)
                / pooled_std
            )
        else:
            separation = 0.0

        print(
            f"{feature:25s} "
            f"separation = {separation:.3f}"
        )

    print("\n" + "=" * 90)
    print("DECISION")
    print("=" * 90)

    print(
        "\nWe will decide whether NORMAL Set 1 should be kept "
        "or removed only after reviewing these results."
    )


if __name__ == "__main__":
    main()
