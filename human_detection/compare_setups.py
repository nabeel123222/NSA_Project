from pathlib import Path

import numpy as np

from features import extract_feature_vector


DATASET_DIR = Path("dataset")

SETUPS = ["setup_A", "setup_B", "setup_C"]

LABELS = {
    "empty": 0,
    "present": 1,
}


def load_features(setup, label_name):
    folder = DATASET_DIR / setup / label_name
    files = sorted(folder.glob("*.npy"))

    features = []

    for file_path in files:

        amp_matrix = np.load(file_path)

        if amp_matrix.shape != (50, 64):
            continue

        if not np.isfinite(amp_matrix).all():
            continue

        if np.count_nonzero(amp_matrix) == 0:
            continue

        vector = extract_feature_vector(
            amp_matrix,
            sample_rate_hz=180.0,
        )

        features.append(vector)

    return np.asarray(features, dtype=np.float64)


print("=" * 70)
print("CSI FEATURE DISTRIBUTION DIAGNOSTIC")
print("=" * 70)

data = {}

# ---------------------------------------------------------
# Load all setup/class features
# ---------------------------------------------------------

for setup in SETUPS:

    for label_name in LABELS:

        print(f"Loading {setup} - {label_name}...")

        data[(setup, label_name)] = load_features(
            setup,
            label_name,
        )

        print(
            "  Shape:",
            data[(setup, label_name)].shape
        )


# ---------------------------------------------------------
# Compare feature distributions
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE DISTRIBUTION COMPARISON")
print("=" * 70)

for label_name in LABELS:

    print("\n" + "-" * 70)
    print(f"CLASS: {label_name.upper()}")
    print("-" * 70)

    reference = data[("setup_A", label_name)]

    for setup in ["setup_B", "setup_C"]:

        current = data[(setup, label_name)]

        mean_difference = np.mean(
            np.abs(
                np.mean(current, axis=0)
                - np.mean(reference, axis=0)
            )
        )

        std_difference = np.mean(
            np.abs(
                np.std(current, axis=0)
                - np.std(reference, axis=0)
            )
        )

        print(
            f"Setup A vs {setup}: "
            f"mean_difference={mean_difference:.6f}, "
            f"std_difference={std_difference:.6f}"
        )


# ---------------------------------------------------------
# Find features most affected by Setup C
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TOP FEATURES AFFECTED BY SETUP C")
print("=" * 70)

for label_name in LABELS:

    setup_a = data[("setup_A", label_name)]
    setup_c = data[("setup_C", label_name)]

    a_mean = np.mean(setup_a, axis=0)
    c_mean = np.mean(setup_c, axis=0)

    differences = np.abs(c_mean - a_mean)

    top_indices = np.argsort(differences)[-10:][::-1]

    print(f"\n{label_name.upper()}:")

    for index in top_indices:

        print(
            f"Feature {index:3d}: "
            f"A_mean={a_mean[index]:.6f} | "
            f"C_mean={c_mean[index]:.6f} | "
            f"difference={differences[index]:.6f}"
        )


# ---------------------------------------------------------
# Compare EMPTY vs PRESENT inside each setup
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("EMPTY vs PRESENT SEPARATION")
print("=" * 70)

for setup in SETUPS:

    empty = data[(setup, "empty")]
    present = data[(setup, "present")]

    separation = np.mean(
        np.abs(
            np.mean(present, axis=0)
            - np.mean(empty, axis=0)
        )
    )

    print(
        f"{setup}: average feature separation = "
        f"{separation:.6f}"
    )


print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETED")
print("=" * 70)
