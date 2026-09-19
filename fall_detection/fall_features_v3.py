import numpy as np


EPSILON = 1e-12


FEATURE_NAMES_V3 = [
    "friend_peak_ratio",
    "friend_event_energy_ratio",
    "friend_post_drop",
    "you_peak_ratio",
    "you_event_energy_ratio",
    "you_post_drop",
]


def validate_amplitude(amp):

    amp = np.asarray(amp, dtype=float)

    if amp.ndim != 2:
        raise ValueError("Amplitude must be 2D.")

    if amp.shape[1] != 64:
        raise ValueError(
            f"Expected 64 subcarriers, got {amp.shape[1]}."
        )

    if amp.shape[0] < 30:
        raise ValueError(
            "At least 30 frames are required."
        )

    if not np.isfinite(amp).all():
        raise ValueError(
            "Amplitude contains NaN or Inf."
        )

    return amp


def get_motion(amp):

    diff = np.diff(amp, axis=0)

    return np.mean(
        np.abs(diff),
        axis=1
    )


def receiver_features(amp):

    amp = validate_amplitude(amp)

    motion = get_motion(amp)

    n = len(motion)

    if n < 12:
        return np.zeros(3, dtype=float)

    # --------------------------------------------------
    # 1. Peak ratio
    # --------------------------------------------------

    baseline = np.median(motion)

    peak = np.max(motion)

    peak_ratio = (
        peak / (baseline + EPSILON)
    )

    # --------------------------------------------------
    # Find event region
    # --------------------------------------------------

    peak_index = int(
        np.argmax(motion)
    )

    # Window around the strongest event
    event_radius = max(
        3,
        n // 20
    )

    event_start = max(
        0,
        peak_index - event_radius
    )

    event_end = min(
        n,
        peak_index + event_radius + 1
    )

    event = motion[
        event_start:event_end
    ]

    # --------------------------------------------------
    # 2. Event energy ratio
    # --------------------------------------------------

    overall_energy = np.mean(
        motion ** 2
    )

    event_energy = np.mean(
        event ** 2
    )

    event_energy_ratio = (
        event_energy
        /
        (overall_energy + EPSILON)
    )

    # --------------------------------------------------
    # 3. Post-event drop
    # --------------------------------------------------

    pre_end = max(
        3,
        peak_index
    )

    pre_motion = motion[
        :pre_end
    ]

    post_motion = motion[
        peak_index + 1:
    ]

    if len(post_motion) < 3:
        post_drop = 0.0

    else:

        pre_level = np.median(
            pre_motion
        )

        post_level = np.median(
            post_motion
        )

        post_drop = (
            pre_level
            /
            (post_level + EPSILON)
        )

        # Limit extreme numerical values
        post_drop = min(
            post_drop,
            100.0
        )

    return np.asarray(
        [
            peak_ratio,
            event_energy_ratio,
            post_drop
        ],
        dtype=float
    )


def extract_two_receiver_features(
    friend_amp,
    you_amp
):

    friend_features = receiver_features(
        friend_amp
    )

    you_features = receiver_features(
        you_amp
    )

    features = np.concatenate(
        [
            friend_features,
            you_features
        ]
    )

    if not np.isfinite(features).all():

        raise ValueError(
            "V3 features contain NaN or Inf."
        )

    return features
