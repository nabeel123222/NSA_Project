from pathlib import Path

import numpy as np

from signal_processing import process_window


DATASET_DIR = Path("dataset")

SETUPS = ["setup_A", "setup_B", "setup_C"]
LABELS = ["empty", "present"]


def get_motion_energy(file_path):
    amp_matrix = np.load(file_path)

    if amp_matrix.shape != (50, 64):
        return None

    if not np.isfinite(amp_matrix).all():
        return None

    features = process_window(
        amp_matrix,
        sample_rate_hz=180.0,
    )

    # Feature positions 192–255 = motion energy
    return features.motion_energy


print("=" * 70)
print("MOTION ENERGY DIAGNOSTIC")
print("=" * 70)

for setup in SETUPS:

    for label in LABELS:

        folder = DATASET_DIR / setup / label
        files = sorted(folder.glob("*.npy"))

        all_energy = []

        for file_path in files:

            energy = get_motion_energy(file_path)

            if energy is not None:
                all_energy.append(energy)

        all_energy = np.asarray(all_energy)

        mean_energy = np.mean(all_energy, axis=0)
        std_energy = np.std(all_energy, axis=0)

        print("\n" + "-" * 70)
        print(f"{setup.upper()} - {label.upper()}")
        print("-" * 70)

        print(
            f"Overall mean motion energy : "
            f"{np.mean(mean_energy):.6f}"
        )

        print(
            f"Overall std motion energy  : "
            f"{np.mean(std_energy):.6f}"
        )

        print(
            f"Minimum feature mean       : "
            f"{np.min(mean_energy):.6f}"
        )

        print(
            f"Maximum feature mean       : "
            f"{np.max(mean_energy):.6f}"
        )


print("\n" + "=" * 70)
print("MOTION ENERGY DIAGNOSTIC COMPLETED")
print("=" * 70)
