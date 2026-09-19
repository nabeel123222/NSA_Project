"""
fall_features.py
----------------
Fall-specific feature extraction for the NSA project.

This module reuses the existing Phase 1 signal-processing pipeline
from human_detection and adds a small number of fall-specific features.

Existing pipeline:
    raw CSI
       ↓
    amplitude
       ↓
    process_window()
       ↓
    258 existing CSI features

Fall-specific features:
    1. Peak motion energy
    2. Motion-energy change (jerk-like feature)
    3. Post-event stillness ratio

The goal is to distinguish:

    NORMAL activity
            vs
          FALL
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# ------------------------------------------------------------
# Reuse existing human_detection processing
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HUMAN_DETECTION_DIR = PROJECT_ROOT / "human_detection"

if str(HUMAN_DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(HUMAN_DETECTION_DIR))

from signal_processing import process_window


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

EXPECTED_SUBCARRIERS = 64
FALL_WINDOW_SECONDS = 3.0

# Small value used to avoid division by zero.
EPSILON = 1e-12


# ------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------

def validate_amplitude_matrix(amp_matrix: np.ndarray) -> None:
    """
    Validate an amplitude matrix before processing.
    """

    amp_matrix = np.asarray(amp_matrix)

    if amp_matrix.ndim != 2:
        raise ValueError(
            f"Expected 2D amplitude matrix, got {amp_matrix.shape}"
        )

    if amp_matrix.shape[1] != EXPECTED_SUBCARRIERS:
        raise ValueError(
            f"Expected {EXPECTED_SUBCARRIERS} subcarriers, "
            f"got {amp_matrix.shape[1]}"
        )

    if amp_matrix.shape[0] < 30:
        raise ValueError(
            f"Not enough frames for fall analysis: "
            f"{amp_matrix.shape[0]}"
        )

    if not np.isfinite(amp_matrix).all():
        raise ValueError(
            "Amplitude matrix contains NaN or infinite values."
        )


# ------------------------------------------------------------
# Motion signal
# ------------------------------------------------------------

def calculate_motion_signal(amp_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate frame-to-frame CSI motion.

    The mean absolute change across subcarriers gives one
    motion value per frame.
    """

    frame_difference = np.diff(amp_matrix, axis=0)

    motion = np.mean(
        np.abs(frame_difference),
        axis=1
    )

    return motion


# ------------------------------------------------------------
# Peak motion energy
# ------------------------------------------------------------

def calculate_peak_motion_energy(
    motion_signal: np.ndarray,
) -> float:
    """
    Maximum short-term motion energy.

    A fall is expected to contain a strong transient movement,
    so the peak is useful in addition to average motion energy.
    """

    if len(motion_signal) == 0:
        return 0.0

    return float(
        np.max(motion_signal ** 2)
    )


# ------------------------------------------------------------
# Motion-energy change / jerk-like feature
# ------------------------------------------------------------

def calculate_motion_jerk(
    motion_signal: np.ndarray,
) -> float:
    """
    Measure the largest change in motion energy between
    consecutive portions of the window.

    The window is divided into three sections.

    A sudden event should produce a larger change than
    relatively smooth normal activity.
    """

    n = len(motion_signal)

    if n < 6:
        return 0.0

    third = n // 3

    section_1 = motion_signal[:third]
    section_2 = motion_signal[third:2 * third]
    section_3 = motion_signal[2 * third:]

    energies = np.array([
        np.mean(section_1 ** 2),
        np.mean(section_2 ** 2),
        np.mean(section_3 ** 2),
    ])

    changes = np.abs(
        np.diff(energies)
    )

    return float(
        np.max(changes)
    )


# ------------------------------------------------------------
# Post-event stillness
# ------------------------------------------------------------

def calculate_post_event_stillness(
    motion_signal: np.ndarray,
) -> float:
    """
    Estimate how still the signal becomes after the main
    motion event.

    The strongest motion point is treated as the event peak.
    The second half after that point is compared against
    the overall motion level.

    Returns a value between approximately 0 and 1:

        1.0 → very still after event
        0.0 → continued movement
    """

    n = len(motion_signal)

    if n < 10:
        return 0.0

    peak_index = int(
        np.argmax(motion_signal)
    )

    # Need enough samples after the peak.
    if peak_index >= n - 5:
        return 0.0

    post_event = motion_signal[
        peak_index + 1:
    ]

    if len(post_event) < 5:
        return 0.0

    baseline = float(
        np.median(motion_signal)
    )

    post_energy = float(
        np.mean(post_event ** 2)
    )

    baseline_energy = (
        baseline ** 2 + EPSILON
    )

    ratio = post_energy / baseline_energy

    # Convert high post-event motion into low stillness.
    stillness = 1.0 / (1.0 + ratio)

    return float(
        np.clip(stillness, 0.0, 1.0)
    )


# ------------------------------------------------------------
# Fall-specific features for one receiver
# ------------------------------------------------------------

def extract_fall_features(
    amp_matrix: np.ndarray,
    sample_rate_hz: float,
) -> np.ndarray:
    """
    Extract fall-specific features from one receiver.

    Returns:

        [
            peak_motion_energy,
            motion_jerk,
            post_event_stillness
        ]
    """

    amp_matrix = np.abs(
        np.asarray(amp_matrix)
    )

    validate_amplitude_matrix(
        amp_matrix
    )

    # Reuse the existing tested processing pipeline.
    processed = process_window(
        amp_matrix,
        sample_rate_hz
    )

    if processed is None:
        raise RuntimeError(
            "process_window() returned None."
        )

    # Calculate a simple frame-to-frame motion signal
    # from the validated amplitude data.
    motion_signal = calculate_motion_signal(
        amp_matrix
    )

    peak_energy = calculate_peak_motion_energy(
        motion_signal
    )

    jerk = calculate_motion_jerk(
        motion_signal
    )

    stillness = calculate_post_event_stillness(
        motion_signal
    )

    return np.array(
        [
            peak_energy,
            jerk,
            stillness,
        ],
        dtype=float,
    )


# ------------------------------------------------------------
# Combined two-ESP32 fall features
# ------------------------------------------------------------

def extract_two_receiver_fall_features(
    friend_amp_matrix: np.ndarray,
    you_amp_matrix: np.ndarray,
    friend_rate: float,
    you_rate: float,
) -> np.ndarray:
    """
    Extract fall-specific features from both ESP32 receivers.

    Output:

        3 features from Friend
        +
        3 features from You
        =
        6 fall-specific features
    """

    friend_features = extract_fall_features(
        friend_amp_matrix,
        friend_rate,
    )

    you_features = extract_fall_features(
        you_amp_matrix,
        you_rate,
    )

    features = np.concatenate(
        [
            friend_features,
            you_features,
        ]
    )

    if not np.isfinite(features).all():
        raise ValueError(
            "Fall feature vector contains NaN or infinite values."
        )

    return features


# ------------------------------------------------------------
# Feature names
# ------------------------------------------------------------

FALL_FEATURE_NAMES = [
    "friend_peak_motion_energy",
    "friend_motion_jerk",
    "friend_post_event_stillness",
    "you_peak_motion_energy",
    "you_motion_jerk",
    "you_post_event_stillness",
]


def get_feature_names() -> list[str]:
    """Return names of the six fall-specific features."""
    return FALL_FEATURE_NAMES.copy()
