import numpy as np
from pathlib import Path

from fall_features import extract_two_receiver_fall_features


BASE_DIR = Path(__file__).resolve().parent

NORMAL_DIR = BASE_DIR / "dataset" / "test" / "normal"
FALL_DIR = BASE_DIR / "dataset" / "test" / "fall"

FEATURE_NAMES = [
    "friend_peak_motion_energy",
    "friend_motion_jerk",
    "friend_post_event_stillness",
    "you_peak_motion_energy",
    "you_motion_jerk",
    "you_post_event_stillness",
]


def load_features(file_path):
    data = np.load(file_path)

    return extract_two_receiver_fall_features(
        data["friend_csi"],
        data["you_csi"],
        float(data["friend_rate"]),
        float(data["you_rate"]),
    )


print("=" * 70)
print("NSA - FALL DETECTION")
print("UNSEEN NORMAL FEATURE ANALYSIS")
print("=" * 70)


normal_files = sorted(NORMAL_DIR.glob("normal_*.npz"))
fall_files = sorted(FALL_DIR.glob("fall_*.npz"))


normal_features = np.array(
    [load_features(f) for f in normal_files]
)

fall_features = np.array(
    [load_features(f) for f in fall_files]
)


print("\nUNSEEN NORMAL SAMPLES")
print("-" * 70)

for i, (name, features) in enumerate(
    zip([f.name for f in normal_files], normal_features),
    start=1
):
    print(f"\n{name}")

    for feature_name, value in zip(FEATURE_NAMES, features):
        print(f"  {feature_name:32s}: {value:.6f}")


print("\n" + "=" * 70)
print("FEATURE STATISTICS")
print("=" * 70)

print("\nUNSEEN NORMAL")
print("-" * 70)

for i, name in enumerate(FEATURE_NAMES):
    values = normal_features[:, i]

    print(
        f"{name:32s} "
        f"mean={np.mean(values):10.4f}  "
        f"std={np.std(values):10.4f}  "
        f"min={np.min(values):10.4f}  "
        f"max={np.max(values):10.4f}"
    )


print("\nUNSEEN FALL")
print("-" * 70)

for i, name in enumerate(FEATURE_NAMES):
    values = fall_features[:, i]

    print(
        f"{name:32s} "
        f"mean={np.mean(values):10.4f}  "
        f"std={np.std(values):10.4f}  "
        f"min={np.min(values):10.4f}  "
        f"max={np.max(values):10.4f}"
    )


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
