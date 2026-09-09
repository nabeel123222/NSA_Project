"""
features.py
-----------
Converts one raw CSI amplitude window into a flat 1D feature vector for
the Random Forest human-detection model.

This file does NOT implement any signal processing itself - it is a thin
adapter between:

    raw amplitude matrix (50, 64)
              |
              v
    signal_processing.process_window()   <- existing, tested pipeline
              |
              v
    CSIFeatures.as_vector()              <- existing, already flattens
              |
              v
    1D numpy array  ->  RandomForestClassifier.fit()/.predict()

Reused unchanged from signal_processing.py:
    - Hampel filtering (outlier removal)
    - moving-average smoothing
    - band-pass motion isolation (0.5-10 Hz)
    - feature extraction (mean/std/MAD/motion energy/correlation)

Nothing here duplicates or replaces that pipeline - it is the ONLY
processing path this module uses.
"""

from __future__ import annotations

import numpy as np

from signal_processing import process_window


# Confirmed empirically: ESP32 CSI rate when kept close to the 2.4GHz
# hotspot. Used ONLY as a fallback if a caller doesn't supply a measured
# rate - always prefer passing the real, measured rate when you have it
# (e.g. from timestamps during collect_data.py / live_detector.py),
# since the band-pass filter's behavior depends on this value.
DEFAULT_SAMPLE_RATE_HZ = 101.0

EXPECTED_SHAPE = (50, 64)  # (num_frames, num_subcarriers)


def extract_feature_vector(
    amp_matrix: np.ndarray,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
) -> np.ndarray:
    """
    Turn one raw CSI amplitude window into a flat feature vector.

    Parameters
    ----------
    amp_matrix : np.ndarray, shape (50, 64)
        Raw amplitude window, as produced by
        serial_reader.frames_to_amplitude_matrix().
    sample_rate_hz : float
        The ACTUAL measured CSI frame rate for this capture session.
        Pass the real measured value when available - the band-pass
        filter's cutoff behavior depends on it (see signal_processing.py).

    Returns
    -------
    np.ndarray, shape (194,)
        Flat feature vector: [mean_amplitude(64), std_amplitude(64),
        mad_amplitude(64), motion_energy(64)... wait, see note below]
        Exact length is whatever CSIFeatures.as_vector() produces -
        this function does not reshape or truncate it.

    Raises
    ------
    ValueError
        If amp_matrix isn't a 2D array shaped (50, 64). Catching this
        here, at the boundary, means a malformed window fails loudly
        and immediately instead of silently producing a garbage
        feature vector that RandomForestClassifier would happily
        train on without complaint - which is exactly the kind of bug
        that produced the corrupted all-zero PRESENT samples before.
    """
    amp_matrix = np.asarray(amp_matrix)

    if amp_matrix.ndim != 2:
        raise ValueError(
            f"Expected a 2D amplitude matrix, got shape {amp_matrix.shape}"
        )

    if amp_matrix.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Expected amplitude matrix of shape {EXPECTED_SHAPE}, "
            f"got {amp_matrix.shape}. Check the window size used when "
            f"calling frames_to_amplitude_matrix()."
        )

    if not np.isfinite(amp_matrix).all():
        raise ValueError(
            "amp_matrix contains NaN or infinite values - check the "
            "serial capture, this window should not be used for "
            "training or prediction."
        )

    features = process_window(amp_matrix, sample_rate_hz)
    vector = features.as_vector()

    return vector
