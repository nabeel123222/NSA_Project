"""
human_detector.py
------------------
Temporary threshold-based detector. Compares a live window's motion score
against the empty-room baseline, using an adaptive threshold plus temporal
persistence to avoid single-window false positives/negatives.

NOT the final AI system - this is Module A (see project docs).
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np

from baseline import BaselineStats
from signal_processing import CSIFeatures


@dataclass
class DetectorResult:
    is_human_present: bool
    z_score: float
    raw_motion_score: float
    confidence: float
    smoothed_decision: bool


class HumanDetector:
    def __init__(self, baseline: BaselineStats, z_threshold: float = 3.0):
        self.baseline = baseline
        self.z_threshold = z_threshold

        statistical_threshold = baseline.motion_score_mean + 3.0 * baseline.motion_score_std
        doubled_baseline = baseline.motion_score_mean * 2.0
        self.motion_threshold = max(0.40, statistical_threshold, doubled_baseline)

        self.required_present_windows = 3
        self.required_empty_windows = 5
        self._present_count = 0
        self._empty_count = 0
        self._current_state = False

    def _compute_z_score(self, features: CSIFeatures) -> float:
        std = max(self.baseline.motion_score_std, 1e-9)
        return (features.total_motion_score - self.baseline.motion_score_mean) / std

    def detect(self, features: CSIFeatures) -> DetectorResult:
        motion = float(features.total_motion_score)
        z = self._compute_z_score(features)
        strong_motion = motion >= self.motion_threshold

        if strong_motion:
            self._present_count += 1
            self._empty_count = 0
        else:
            self._empty_count += 1
            self._present_count = 0

        if not self._current_state and self._present_count >= self.required_present_windows:
            self._current_state = True
            print(f"PERSON DETECTED | motion={motion:.4f} threshold={self.motion_threshold:.4f} z={z:.2f}")
        elif self._current_state and self._empty_count >= self.required_empty_windows:
            self._current_state = False
            print("ROOM EMPTY")

        confidence = float(np.clip(motion / self.motion_threshold, 0.0, 1.0)) if self.motion_threshold > 0 else 0.0

        return DetectorResult(
            is_human_present=strong_motion,
            z_score=float(z),
            raw_motion_score=motion,
            confidence=confidence,
            smoothed_decision=self._current_state,
        )

    def reset_history(self):
        self._present_count = 0
        self._empty_count = 0
        self._current_state = False