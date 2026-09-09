"""
baseline.py
-----------
Builds the "empty room" reference: collects raw CSI while the room is
empty, processes it into motion-score windows, and saves the resulting
mean/std as the baseline that human_detector.py compares live data against.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import numpy as np

from config import BASELINE_FRAMES, WINDOW_SIZE
from serial_reader import frames_to_amplitude_matrix
from signal_processing import process_window


@dataclass
class BaselineStats:
    motion_score_mean: float
    motion_score_std: float
    motion_energy_mean: float
    motion_energy_std: float
    subcarrier_correlation_mean: float
    sample_rate_hz: float
    window_size: int
    num_windows_used: int

    def save(self, path: str = "../data/baseline_stats.json"):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str = "../data/baseline_stats.json") -> "BaselineStats":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def run_room_setup(reader, num_frames: int = BASELINE_FRAMES, window_size: int = WINDOW_SIZE, stride: int = 10) -> BaselineStats:
    """
    Collects `num_frames` raw CSI frames while the room is empty, slides a
    window of `window_size` across them, and computes motion-score
    statistics across all resulting windows.
    """
    print(f"Collecting {num_frames} empty-room frames...")
    csi_frames = reader.read_window(num_frames)  # List[CSIFrame]

    # Effective sample rate from actual frame timestamps, not wall-clock
    # around the whole call (more accurate - matches what stream_windows
    # will experience live).
    elapsed = csi_frames[-1].timestamp - csi_frames[0].timestamp
    sample_rate_hz = (len(csi_frames) - 1) / elapsed if elapsed > 0 else 100.0

    amp_matrix = frames_to_amplitude_matrix(csi_frames)  # (num_frames, num_subcarriers)

    motion_scores = []
    motion_energies = []
    correlations = []

    for start in range(0, len(amp_matrix) - window_size, stride):
        window = amp_matrix[start:start + window_size]
        features = process_window(window, sample_rate_hz)
        motion_scores.append(features.total_motion_score)
        motion_energies.append(float(np.mean(features.motion_energy)))
        correlations.append(features.subcarrier_correlation)

    stats = BaselineStats(
        motion_score_mean=float(np.mean(motion_scores)),
        motion_score_std=float(np.std(motion_scores)),
        motion_energy_mean=float(np.mean(motion_energies)),
        motion_energy_std=float(np.std(motion_energies)),
        subcarrier_correlation_mean=float(np.mean(correlations)),
        sample_rate_hz=sample_rate_hz,
        window_size=window_size,
        num_windows_used=len(motion_scores),
    )

    print(f"Baseline complete: mean={stats.motion_score_mean:.4f}, std={stats.motion_score_std:.4f}, "
          f"sample_rate={stats.sample_rate_hz:.2f}Hz, windows={stats.num_windows_used}")

    stats.save()
    return stats