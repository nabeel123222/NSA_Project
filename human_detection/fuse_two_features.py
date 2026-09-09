import numpy as np

from signal_processing import process_window
from serial_reader import CSISerialReader


PORTS = {
    "FRIEND": "/dev/ttyACM0",
    "YOU": "/dev/ttyACM1",
}

BAUDRATE = 921600
WINDOW_SIZE = 50
EXPECTED_SUBCARRIERS = 64


def capture_window(name, port):
    reader = CSISerialReader(
        port=port,
        baudrate=BAUDRATE,
        timeout=2.0,
    )

    reader.open()

    frames = []

    while len(frames) < WINDOW_SIZE:
        frame = reader.read_frame()

        if frame is None:
            continue

        if frame.complex_csi.size != EXPECTED_SUBCARRIERS:
            continue

        frames.append(frame)

    reader.close()

    matrix = np.array(
        [
            np.abs(frame.complex_csi)
            for frame in frames
        ],
        dtype=np.float32,
    )

    return matrix


print("=" * 60)
print("TWO ESP32 FEATURE FUSION TEST")
print("=" * 60)

print("\nCapturing Friend ESP32...")
friend_matrix = capture_window(
    "FRIEND",
    PORTS["FRIEND"],
)

print("Capturing Your ESP32...")
you_matrix = capture_window(
    "YOU",
    PORTS["YOU"],
)

print()
print("Raw matrices:")
print("Friend:", friend_matrix.shape)
print("You:   ", you_matrix.shape)


# Extract features using the SAME pipeline
friend_features = process_window(
    friend_matrix,
    sample_rate_hz=101.0,
).as_vector()

you_features = process_window(
    you_matrix,
    sample_rate_hz=101.0,
).as_vector()


print()
print("Feature sizes:")
print("Friend features:", friend_features.shape)
print("Your features:  ", you_features.shape)


# Combine both ESP32 feature vectors
combined_features = np.concatenate(
    [friend_features, you_features]
)


print()
print("=" * 60)
print("FUSION RESULT")
print("=" * 60)

print(
    "Combined feature vector shape:",
    combined_features.shape
)

print(
    "Expected:",
    "(516,)"
)

if combined_features.shape == (516,):
    print()
    print("SUCCESS!")
    print("Two ESP32 feature fusion is working.")
else:
    print()
    print("FAILED!")
    print("Unexpected feature size.")

print("=" * 60)
