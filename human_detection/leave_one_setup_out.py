from pathlib import Path

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from features import extract_feature_vector


DATASET_DIR = Path("dataset")

LABELS = {
    "empty": 0,
    "present": 1,
}


def load_setup(setups):
    X = []
    y = []

    for setup in setups:

        print(f"\nLoading {setup.upper()}...")

        for label_name, label in LABELS.items():

            folder = DATASET_DIR / setup / label_name
            files = sorted(folder.glob("*.npy"))

            loaded = 0

            for file_path in files:

                try:
                    amp_matrix = np.load(file_path)

                    if amp_matrix.shape != (50, 64):
                        continue

                    if not np.isfinite(amp_matrix).all():
                        continue

                    if np.count_nonzero(amp_matrix) == 0:
                        continue

                    feature_vector = extract_feature_vector(
                        amp_matrix,
                        sample_rate_hz=180.0,
                    )

                    X.append(feature_vector)
                    y.append(label)

                    loaded += 1

                except Exception:
                    continue

            print(
                f"  {label_name.upper()}: {loaded}"
            )

    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.int64),
    )


def run_test(train_setups, test_setup):

    print("\n" + "=" * 70)
    print(
        f"TRAIN: {', '.join(train_setups)}  "
        f"→  TEST: {test_setup}"
    )
    print("=" * 70)

    X_train, y_train = load_setup(train_setups)
    X_test, y_test = load_setup([test_setup])

    print("\nTraining samples:", len(X_train))
    print("Testing samples :", len(X_test))

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", f"{accuracy * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return accuracy


print("=" * 70)
print("LEAVE-ONE-SETUP-OUT GENERALIZATION TEST")
print("=" * 70)

print("\nNo files will be modified.")
print("The saved human_detector.joblib will NOT be changed.")

results = {}

results["A+B -> C"] = run_test(
    ["setup_A", "setup_B"],
    "setup_C",
)

results["A+C -> B"] = run_test(
    ["setup_A", "setup_C"],
    "setup_B",
)

results["B+C -> A"] = run_test(
    ["setup_B", "setup_C"],
    "setup_A",
)

print("\n" + "=" * 70)
print("FINAL GENERALIZATION RESULTS")
print("=" * 70)

for test_name, accuracy in results.items():
    print(
        f"{test_name:12s} : "
        f"{accuracy * 100:.2f}%"
    )

print("=" * 70)
print("TEST COMPLETED")
print("=" * 70)
