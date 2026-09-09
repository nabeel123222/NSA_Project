"""
collect_data.py
---------------
Collects fresh CSI windows for the Random Forest human-detection model.

Classes:
    empty   -> room is empty
    present -> person is inside the detection zone

Each sample:
    50 CSI frames x 64 subcarriers

Before saving a sample, the collector verifies that:
    - the shape is exactly (50, 64)
    - values are finite
    - the CSI data is not completely zero

Each successfully saved sample gets:
    - its own uniquely-indexed .npy file
    - one appended row in metadata.csv (rate, mean, std, nonzero, elapsed)

Both the filename and the metadata write happen INSIDE the loop, computed
fresh every iteration - this is deliberate, to prevent a class of bug
where a filename or file handle gets reused across iterations and
silently overwrites the previous sample.
"""

from __future__ import annotations

import argparse
import csv
from email import parser
import os
import time

import numpy as np

from serial_reader import (
    CSISerialReader,
    frames_to_amplitude_matrix,
)


WINDOW_SIZE = 50
DEFAULT_PORT = "/dev/ttyACM0"
DATASET_DIR = "dataset"


def measure_sample_rate(frames) -> float:
    """Estimate CSI frame rate from frame timestamps."""
    timestamps = np.array([frame.timestamp for frame in frames], dtype=np.float64)

    if len(timestamps) < 2:
        return 0.0

    differences = np.diff(timestamps)
    differences = differences[differences > 0]

    if len(differences) == 0:
        return 0.0

    return float(1.0 / np.median(differences))


def validate_csi_window(amp_matrix: np.ndarray) -> tuple[bool, str]:
    """Validate one CSI amplitude window before saving."""
    if amp_matrix.shape != (WINDOW_SIZE, 64):
        return False, f"wrong shape {amp_matrix.shape}; expected (50, 64)"

    if not np.isfinite(amp_matrix).all():
        return False, "contains NaN or infinite values"

    nonzero = np.count_nonzero(amp_matrix)
    if nonzero == 0:
        return False, "ALL ZERO CSI DATA"

    return True, "OK"


def collect_data(
    label: str,
    num_samples: int,
    port: str,
    setup: str
):

    save_dir = os.path.join(
        DATASET_DIR,
        setup,
        label
    )
    os.makedirs(save_dir, exist_ok=True)

    metadata_path = os.path.join(save_dir, "metadata.csv")
    metadata_is_new = not os.path.exists(metadata_path)

    existing = [f for f in os.listdir(save_dir) if f.endswith(".npy")]
    start_index = len(existing)

    print("\n" + "=" * 60)
    print("NSA HUMAN DETECTION - DATA COLLECTION")
    print("=" * 60)
    print(f"Class       : {label}")
    print(f"Samples     : {num_samples}")
    print(f"Window      : {WINDOW_SIZE} frames")
    print(f"Save folder : {save_dir}")
    print(f"Start index : {start_index}  (existing files found: {len(existing)})")
    print(f"Serial port : {port}")
    print("=" * 60)

    if label == "empty":
        print("\nROOM MUST BE EMPTY.")
        print("Nobody should enter the detection zone.")
    else:
        print("\nPERSON PRESENT.")
        print("One person should remain inside the detection zone.")

    input("\nPress ENTER to start...")

    reader = CSISerialReader(port=port)

    collected = 0
    rejected = 0
    rate_samples = []

    # Metadata file opened ONCE in append mode, then every write below
    # uses this same open handle with an explicit flush - this avoids
    # any risk of accidentally reopening in "w" (truncate) mode later.
    with reader, open(metadata_path, "a", newline="") as meta_file:
        meta_writer = csv.writer(meta_file)
        if metadata_is_new:
            meta_writer.writerow(["filename", "rate_hz", "mean", "std", "nonzero", "elapsed_sec"])
            meta_file.flush()

        while collected < num_samples:
            try:
                print(f"\nWaiting for CSI window {collected + 1}/{num_samples}...")

                start_time = time.time()
                frames = reader.read_window(WINDOW_SIZE)
                elapsed = time.time() - start_time

                amp_matrix = frames_to_amplitude_matrix(frames)
                amp_matrix = np.asarray(amp_matrix, dtype=np.float32)

                valid, reason = validate_csi_window(amp_matrix)
                if not valid:
                    rejected += 1
                    print(f"REJECTED | {reason}")
                    continue

                rate = measure_sample_rate(frames)
                if rate > 0:
                    rate_samples.append(rate)

                # Fresh, unique index computed THIS iteration, based on
                # how many samples we've saved so far this run.
                sample_index = start_index + collected
                filename = os.path.join(save_dir, f"{label}_{sample_index:05d}.npy")

                # Fail loudly if this filename somehow already exists -
                # this should be structurally impossible now, but if it
                # ever happens again, this stops the run instead of
                # silently overwriting data.
                if os.path.exists(filename):
                    raise RuntimeError(
                        f"Refusing to overwrite existing file: {filename}"
                    )

                np.save(filename, amp_matrix)

                meta_writer.writerow([
                    os.path.basename(filename),
                    f"{rate:.4f}",
                    f"{amp_matrix.mean():.6f}",
                    f"{amp_matrix.std():.6f}",
                    int(np.count_nonzero(amp_matrix)),
                    f"{elapsed:.3f}",
                ])
                meta_file.flush()  # write to disk immediately, don't buffer

                collected += 1

                print(
                    f"SAVED | {collected}/{num_samples} | "
                    f"File={os.path.basename(filename)} | "
                    f"Shape={amp_matrix.shape} | "
                    f"Nonzero={np.count_nonzero(amp_matrix)} | "
                    f"Mean={amp_matrix.mean():.4f} | "
                    f"Std={amp_matrix.std():.4f} | "
                    f"Rate={rate:.1f} Hz | "
                    f"Time={elapsed:.2f}s"
                )

            except KeyboardInterrupt:
                print("\nCollection stopped by user.")
                break

            except Exception as e:
                print(f"\nERROR while collecting window: {e}")

    print("\n" + "=" * 60)
    print("COLLECTION COMPLETED")
    print("=" * 60)
    print(f"Class          : {label}")
    print(f"Saved samples  : {collected}")
    print(f"Rejected       : {rejected}")
    if rate_samples:
        print(f"Average CSI rate: {np.mean(rate_samples):.2f} Hz")
    print(f"Folder         : {save_dir}")

    # Final on-disk verification, printed so a mismatch is impossible to miss.
    actual_files = [f for f in os.listdir(save_dir) if f.endswith(".npy")]
    print(f"Actual .npy files on disk in {save_dir}: {len(actual_files)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Collect CSI data for Random Forest human detection."
    )
    parser.add_argument(
    "--label",
    required=True,
    choices=["empty", "present"]
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=100
    )

    parser.add_argument(
        "--port",
        default=DEFAULT_PORT
    )

    parser.add_argument(
        "--setup",
        required=True,
        choices=["setup_A", "setup_B", "setup_C"]
    )

    args = parser.parse_args()

    collect_data(
        label=args.label,
        num_samples=args.samples,
        port=args.port,
        setup=args.setup
    )


if __name__ == "__main__":
    main()