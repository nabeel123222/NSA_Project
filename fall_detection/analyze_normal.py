import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "human_detection"))

from fall_features import extract_two_receiver_fall_features


DATASET_DIR = Path(__file__).resolve().parent / "dataset" / "normal"


FEATURE_NAMES = [
    "Friend Peak Motion Energy",
    "Friend Motion Jerk",
    "Friend Post-Event Stillness",
    "You Peak Motion Energy",
    "You Motion Jerk",
    "You Post-Event Stillness",
]


def main():

    files = sorted(DATASET_DIR.glob("normal_*.npz"))

    print("=" * 70)
    print("NSA - NORMAL DATASET ANALYSIS")
    print("=" * 70)

    print(f"\nDataset folder: {DATASET_DIR}")
    print(f"Samples found : {len(files)}")

    if not files:
        print("\nERROR: No NORMAL samples found.")
        return

    all_features = []
    valid_files = []

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
                print(
                    f"SKIP {file.name}: "
                    f"only {n} common frames"
                )
                continue

            friend_amp = friend_amp[:n]
            you_amp = you_amp[:n]

            features = extract_two_receiver_fall_features(
                friend_amp,
                you_amp,
                friend_rate,
                you_rate
            )

            if len(features) != 6:
                print(
                    f"SKIP {file.name}: "
                    f"feature count = {len(features)}"
                )
                continue

            if not np.isfinite(features).all():
                print(
                    f"SKIP {file.name}: "
                    "non-finite features"
                )
                continue

            all_features.append(features)
            valid_files.append(file.name)

        except Exception as e:

            print(
                f"SKIP {file.name}: {e}"
            )

    if not all_features:
        print("\nERROR: No valid samples.")
        return

    X = np.asarray(all_features)

    print("\n" + "=" * 70)
    print("FEATURE STATISTICS")
    print("=" * 70)

    print(f"\nValid samples: {len(X)}")

    print(
        "\n"
        f"{'Feature':35s}"
        f"{'Mean':>12s}"
        f"{'Std':>12s}"
        f"{'Min':>12s}"
        f"{'Max':>12s}"
    )

    print("-" * 83)

    for i, name in enumerate(FEATURE_NAMES):

        values = X[:, i]

        print(
            f"{name:35s}"
            f"{np.mean(values):12.4f}"
            f"{np.std(values):12.4f}"
            f"{np.min(values):12.4f}"
            f"{np.max(values):12.4f}"
        )

    print("\n" + "=" * 70)
    print("SAMPLE-BY-SAMPLE VALUES")
    print("=" * 70)

    for filename, features in zip(
        valid_files,
        X
    ):

        print(
            f"\n{filename}"
        )

        print(
            f"  Friend: "
            f"peak={features[0]:.3f}, "
            f"jerk={features[1]:.3f}, "
            f"stillness={features[2]:.3f}"
        )

        print(
            f"  You:    "
            f"peak={features[3]:.3f}, "
            f"jerk={features[4]:.3f}, "
            f"stillness={features[5]:.3f}"
        )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
