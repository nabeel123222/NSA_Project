"""
collect_calibration.py
----------------------
Collect local CSI calibration data for the current room/setup.

Classes:
    empty   = room is empty
    present = one person is inside the detection zone

Each sample:
    (50, 64) CSI amplitude matrix

Output:
    calibration/current_room/empty/
    calibration/current_room/present/
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from serial_reader import CSISerialReader, frames_to_amplitude_matrix


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

WINDOW_SIZE = 50
EXPECTED_SUBCARRIERS = 64

DEFAULT_PORT = "/dev/ttyACM0"

CALIBRATION_DIR = "calibration"


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def measure_sample_rate(frames):
    """Calculate CSI sample rate from frame timestamps."""

    if len(frames) < 2:
        return 0.0

    timestamps = np.array(
        [frame.timestamp for frame in frames],
        dtype=np.float64,
    )

    intervals = np.diff(timestamps)

    intervals = intervals[intervals > 0]

    if len(intervals) == 0:
        return 0.0

    median_interval = np.median(intervals)

    if median_interval <= 0:
        return 0.0

    return float(1.0 / median_interval)


def validate_csi_window(amp_matrix):
    """Check whether the CSI window is valid."""

    if amp_matrix.shape != (WINDOW_SIZE, EXPECTED_SUBCARRIERS):
        return False, (
            f"invalid shape {amp_matrix.shape}, "
            f"expected {(WINDOW_SIZE, EXPECTED_SUBCARRIERS)}"
        )

    if not np.isfinite(amp_matrix).all():
        return False, "contains NaN or Inf"

    if np.count_nonzero(amp_matrix) == 0:
        return False, "all values are zero"

    return True, ""


# ---------------------------------------------------------
# Existing sample count
# ---------------------------------------------------------

def get_existing_samples(save_dir, label):
    """Find the next available sample number."""

    os.makedirs(save_dir, exist_ok=True)

    files = [
        f
        for f in os.listdir(save_dir)
        if f.endswith(".npy")
    ]

    indexes = []

    for filename in files:

        try:
            number = int(
                filename.replace(
                    f"{label}_",
                    ""
                ).replace(
                    ".npy",
                    ""
                )
            )

            indexes.append(number)

        except ValueError:
            continue

    if not indexes:
        return 0

    return max(indexes) + 1


# ---------------------------------------------------------
# Collect calibration data
# ---------------------------------------------------------

def collect_calibration(
    label,
    num_samples,
    port,
    setup,
):
    """Collect calibration samples for one class."""

    save_dir = os.path.join(
        CALIBRATION_DIR,
        setup,
        label,
    )

    os.makedirs(save_dir, exist_ok=True)

    metadata_path = os.path.join(
        save_dir,
        "metadata.csv",
    )

    start_index = get_existing_samples(
        save_dir,
        label,
    )

    metadata_is_new = not os.path.exists(
        metadata_path
    )

    print("\n" + "=" * 60)
    print("NSA LOCAL ROOM CALIBRATION")
    print("=" * 60)

    print(f"Class          : {label.upper()}")
    print(f"Samples needed : {num_samples}")
    print(f"Serial port    : {port}")
    print(f"Window         : {WINDOW_SIZE} frames")
    print(f"Subcarriers    : {EXPECTED_SUBCARRIERS}")
    print(f"Save directory : {save_dir}")

    print("\nIMPORTANT:")
    
    if label == "empty":
        print("Make sure the detection zone is EMPTY.")
    else:
        print("One person must stand at the normal detection position.")

    input("\nPress ENTER when ready...")

    reader = CSISerialReader(
        port=port,
        baudrate=921600,
    )

    collected = 0
    rejected = 0
    rate_samples = []

    with reader, open(
        metadata_path,
        "a",
        newline="",
    ) as metadata_file:

        writer = csv.writer(metadata_file)

        if metadata_is_new:
            writer.writerow(
                [
                    "filename",
                    "rate_hz",
                    "shape",
                    "timestamp",
                ]
            )

        while collected < num_samples:

            try:

                frames = reader.read_window(
                    WINDOW_SIZE
                )

                amp_matrix = frames_to_amplitude_matrix(
                    frames
                )

                valid, reason = validate_csi_window(
                    amp_matrix
                )

                if not valid:

                    rejected += 1

                    print(
                        f"Rejected window "
                        f"#{rejected}: {reason}"
                    )

                    continue

                sample_rate = measure_sample_rate(
                    frames
                )

                sample_index = (
                    start_index + collected
                )

                filename = (
                    f"{label}_{sample_index:05d}.npy"
                )

                file_path = os.path.join(
                    save_dir,
                    filename,
                )

                if os.path.exists(file_path):
                    print(
                        f"WARNING: {file_path} "
                        f"already exists. Skipping."
                    )
                    continue

                np.save(
                    file_path,
                    amp_matrix,
                )

                writer.writerow(
                    [
                        filename,
                        f"{sample_rate:.2f}",
                        str(amp_matrix.shape),
                        time.time(),
                    ]
                )

                metadata_file.flush()

                collected += 1
                rate_samples.append(
                    sample_rate
                )

                print(
                    f"[{collected:03d}/{num_samples}] "
                    f"saved {filename} | "
                    f"rate={sample_rate:.2f} Hz"
                )

            except KeyboardInterrupt:

                print(
                    "\nCalibration collection "
                    "stopped by user."
                )
                break

            except Exception as e:

                rejected += 1

                print(
                    f"Rejected window: {e}"
                )

    print("\n" + "-" * 60)
    print("CALIBRATION COLLECTION COMPLETE")
    print("-" * 60)

    print(f"Class collected : {label.upper()}")
    print(f"Saved samples    : {collected}")
    print(f"Rejected windows : {rejected}")

    if rate_samples:

        print(
            f"Average rate    : "
            f"{np.mean(rate_samples):.2f} Hz"
        )

    print(f"Location        : {save_dir}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Collect local room calibration CSI data."
    )

    parser.add_argument(
        "label",
        choices=["empty", "present"],
        help="Calibration class.",
    )

    parser.add_argument(
        "--setup",
        choices=["set_A", "set_B", "set_C"],
        required=True,
        help="Calibration set.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples to collect.",
    )

    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        help="ESP32 serial port.",
    )

    args = parser.parse_args()

    collect_calibration(
        label=args.label,
        num_samples=args.samples,
        port=args.port,
        setup=args.setup,
    )


if __name__ == "__main__":
    main()
