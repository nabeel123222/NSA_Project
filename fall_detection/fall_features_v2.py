import numpy as np


EPSILON = 1e-12


FEATURE_NAMES_V2 = [
    "friend_relative_peak",
    "friend_relative_jerk",
    "friend_post_stillness",
    "you_relative_peak",
    "you_relative_jerk",
    "you_post_stillness",
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


def motion_signal(amp):

    # Frame-to-frame CSI change
    diff = np.diff(amp, axis=0)

    # Mean absolute change across subcarriers
    motion = np.mean(
        np.abs(diff),
        axis=1
    )

    return motion


def relative_peak(amp):

    motion = motion_signal(amp)

    peak = np.max(motion)

    # Typical motion level of this capture
    baseline = np.median(motion)

    # Relative change instead of absolute CSI scale
    return float(
        peak / (baseline + EPSILON)
    )


def relative_jerk(amp):

    motion = motion_signal(amp)

    n = len(motion)

    if n < 6:
        return 0.0

    # Divide the capture into 3 temporal sections
    sections = np.array_split(
        motion,
        3
    )

    energies = []

    for section in sections:

        if len(section) == 0:
            energies.append(0.0)
        else:
            energies.append(
                float(
                    np.mean(section ** 2)
                )
            )

    e1, e2, e3 = energies

    # Largest change between neighboring sections
    jerk = max(
        abs(e2 - e1),
        abs(e3 - e2)
    )

    # Normalize by overall motion scale
    overall = np.mean(
        motion ** 2
    )

    return float(
        jerk / (overall + EPSILON)
    )


def post_event_stillness(amp):

    motion = motion_signal(amp)

    if len(motion) == 0:
        return 0.0

    peak_index = int(
        np.argmax(motion)
    )

    post = motion[
        peak_index + 1:
    ]

    if len(post) < 3:
        return 0.0

    # Compare the post-event movement
    # with the movement before the event.
    pre = motion[
        :max(3, peak_index)
    ]

    if len(pre) == 0:
        return 0.0

    post_energy = np.mean(
        post ** 2
    )

    pre_energy = np.median(
        pre ** 2
    )

    ratio = (
        post_energy
        /
        (pre_energy + EPSILON)
    )

    return float(
        1.0 / (1.0 + ratio)
    )


def extract_receiver_features(amp):

    amp = validate_amplitude(amp)

    return np.asarray(
        [
            relative_peak(amp),
            relative_jerk(amp),
            post_event_stillness(amp),
        ],
        dtype=float
    )


def extract_two_receiver_features(
    friend_amp,
    you_amp
):

    friend_features = extract_receiver_features(
        friend_amp
    )

    you_features = extract_receiver_features(
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
            "V2 features contain NaN or Inf."
        )

    return features
