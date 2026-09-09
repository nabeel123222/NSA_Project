"""
signal_processing.py
---------------------
Turns a raw amplitude matrix (num_frames, num_subcarriers) into a compact,
denoised feature vector suitable for statistical comparison AND later usable
as direct input to a CNN/LSTM.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import signal as scipy_signal


def hampel_filter(amp_matrix: np.ndarray, window: int = 7, n_sigmas: float = 3.0) -> np.ndarray:
    if amp_matrix.shape[0] < window:
        return amp_matrix.copy()
    filtered = amp_matrix.copy()
    k = window // 2
    L = 1.4826
    for col in range(amp_matrix.shape[1]):
        series = amp_matrix[:, col]
        for i in range(k, len(series) - k):
            segment = series[i - k: i + k + 1]
            med = np.median(segment)
            mad = L * np.median(np.abs(segment - med))
            if mad == 0:
                continue
            if np.abs(series[i] - med) > n_sigmas * mad:
                filtered[i, col] = med
    return filtered


def moving_average_smooth(amp_matrix: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1 or amp_matrix.shape[0] < window:
        return amp_matrix.copy()
    kernel = np.ones(window) / window
    smoothed = np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="same"), axis=0, arr=amp_matrix
    )
    return smoothed


def bandpass_motion_band(amp_matrix, sample_rate_hz, low_hz=0.5, high_hz=10.0):
    nyquist = sample_rate_hz / 2.0
    if high_hz >= nyquist or amp_matrix.shape[0] < 15:
        return amp_matrix.copy()
    low = max(low_hz / nyquist, 1e-4)
    high = min(high_hz / nyquist, 0.999)
    sos = scipy_signal.butter(4, [low, high], btype="bandpass", output="sos")
    filtered = scipy_signal.sosfiltfilt(sos, amp_matrix, axis=0)
    return filtered


@dataclass
class CSIFeatures:
    mean_amplitude: np.ndarray
    std_amplitude: np.ndarray
    mad_amplitude: np.ndarray
    motion_energy: np.ndarray
    subcarrier_correlation: float
    total_motion_score: float

    def as_vector(self) -> np.ndarray:
        return np.concatenate([
            self.mean_amplitude, self.std_amplitude, self.mad_amplitude,
            self.motion_energy,
            [self.subcarrier_correlation, self.total_motion_score],
        ])


def extract_features(amp_matrix, sample_rate_hz, apply_bandpass=True) -> CSIFeatures:
    mean_amp = amp_matrix.mean(axis=0)
    std_amp = amp_matrix.std(axis=0)
    mad_amp = np.median(np.abs(amp_matrix - np.median(amp_matrix, axis=0)), axis=0)

    if apply_bandpass:
        motion_signal = bandpass_motion_band(amp_matrix, sample_rate_hz)
    else:
        motion_signal = amp_matrix - amp_matrix.mean(axis=0)

    motion_energy = np.mean(motion_signal ** 2, axis=0)

    n_sub = amp_matrix.shape[1]
    if n_sub >= 2 and amp_matrix.shape[0] >= 3:
        sample_cols = np.linspace(0, n_sub - 1, min(n_sub, 20), dtype=int)
        corr_matrix = np.corrcoef(amp_matrix[:, sample_cols].T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        iu = np.triu_indices_from(corr_matrix, k=1)
        subcarrier_correlation = 0.0 if len(iu[0]) == 0 else float(np.mean(np.abs(corr_matrix[iu])))
    else:
        subcarrier_correlation = 0.0

    motion_mean = np.nan_to_num(np.mean(motion_energy), nan=0.0)
    total_motion_score = float(motion_mean * (1.0 + subcarrier_correlation))

    return CSIFeatures(
        mean_amplitude=mean_amp, std_amplitude=std_amp, mad_amplitude=mad_amp,
        motion_energy=motion_energy, subcarrier_correlation=subcarrier_correlation,
        total_motion_score=total_motion_score,
    )


def process_window(amp_matrix, sample_rate_hz, hampel_window=7, smooth_window=3, apply_bandpass=True):
    denoised = hampel_filter(amp_matrix, window=hampel_window)
    smoothed = moving_average_smooth(denoised, window=smooth_window)
    return extract_features(smoothed, sample_rate_hz, apply_bandpass=apply_bandpass)