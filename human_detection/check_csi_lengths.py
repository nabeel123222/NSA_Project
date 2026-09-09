from collections import Counter

from serial_reader import CSISerialReader


PORT = "/dev/ttyACM0"
BAUD_RATE = 921600

counts = Counter()

print("=" * 60)
print("CSI SUBCARRIER LENGTH CHECK")
print("=" * 60)
print("Collecting 200 CSI frames...")
print("Press Ctrl+C to stop early.")
print()

reader = CSISerialReader(
    port=PORT,
    baudrate=BAUD_RATE,
)

try:
    with reader:

        for i in range(200):

            frame = reader.read_frame()

            length = frame.complex_csi.size
            counts[length] += 1

            print(
                f"Frame {i + 1:3d}: "
                f"{length:3d} subcarriers"
            )

except KeyboardInterrupt:
    print("\nStopped early.")

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

for length, count in sorted(counts.items()):
    percentage = count / sum(counts.values()) * 100
    print(
        f"{length:3d} subcarriers : "
        f"{count:3d} frames ({percentage:.1f}%)"
    )

print("=" * 60)
