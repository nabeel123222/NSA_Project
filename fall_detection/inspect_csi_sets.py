import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent / "dataset"

SETS = {
    "NORMAL SET 1": BASE / "normal",
    "NORMAL SET 2": BASE / "normal_set2",
    "FALL": BASE / "fall",
}


def analyze_set(name, folder):

    files = sorted(folder.glob("*.npz"))

    print("\n" + "=" * 90)
    print(name)
    print("=" * 90)

    print(
        f"{'Sample':15s}"
        f"{'Friend Mean':>15s}"
        f"{'Friend Std':>15s}"
        f"{'You Mean':>15s}"
        f"{'You Std':>15s}"
    )

    print("-" * 75)

    for file in files:

        data = np.load(file)

        friend = np.abs(data["friend_csi"])
        you = np.abs(data["you_csi"])

        print(
            f"{file.name:15s}"
            f"{np.mean(friend):15.3f}"
            f"{np.std(friend):15.3f}"
            f"{np.mean(you):15.3f}"
            f"{np.std(you):15.3f}"
        )


def main():

    print("=" * 90)
    print("NSA - RAW CSI PER-SAMPLE INSPECTION")
    print("=" * 90)

    for name, folder in SETS.items():
        analyze_set(name, folder)


if __name__ == "__main__":
    main()
