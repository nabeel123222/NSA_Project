import numpy as np
from pathlib import Path


BASE = Path(__file__).resolve().parent / "dataset"

SETS = {
    "NORMAL SET 1": BASE / "normal",
    "NORMAL SET 2": BASE / "normal_set2",
    "FALL": BASE / "fall",
}


def analyze(name, folder):

    files = sorted(folder.glob("*.npz"))

    friend_means = []
    you_means = []

    friend_stds = []
    you_stds = []

    for file in files:

        data = np.load(file)

        friend_amp = np.abs(data["friend_csi"])
        you_amp = np.abs(data["you_csi"])

        friend_means.append(np.mean(friend_amp))
        you_means.append(np.mean(you_amp))

        friend_stds.append(np.std(friend_amp))
        you_stds.append(np.std(you_amp))

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Samples: {len(files)}"
    )

    print("\nFriend amplitude:")
    print(
        f"  Mean amplitude : "
        f"{np.mean(friend_means):.6f}"
    )
    print(
        f"  Std amplitude  : "
        f"{np.mean(friend_stds):.6f}"
    )
    print(
        f"  Min mean       : "
        f"{np.min(friend_means):.6f}"
    )
    print(
        f"  Max mean       : "
        f"{np.max(friend_means):.6f}"
    )

    print("\nYou amplitude:")
    print(
        f"  Mean amplitude : "
        f"{np.mean(you_means):.6f}"
    )
    print(
        f"  Std amplitude  : "
        f"{np.mean(you_stds):.6f}"
    )
    print(
        f"  Min mean       : "
        f"{np.min(you_means):.6f}"
    )
    print(
        f"  Max mean       : "
        f"{np.max(you_means):.6f}"
    )


def main():

    print("=" * 70)
    print("NSA - CSI AMPLITUDE SCALE CHECK")
    print("=" * 70)

    for name, folder in SETS.items():
        analyze(name, folder)


if __name__ == "__main__":
    main()
