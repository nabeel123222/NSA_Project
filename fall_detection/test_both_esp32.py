import sys
import time

sys.path.insert(0, "../human_detection")

from serial_reader import CSISerialReader


FRIEND_PORT = "/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_94:A9:90:D2:EF:BC-if00"
YOUR_PORT = "/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_E0:72:A1:D6:F5:D0-if00"

BAUDRATE = 921600


def test_reader(name, port):

    print(f"\nTesting {name}...")

    reader = CSISerialReader(
        port=port,
        baudrate=BAUDRATE
    )

    reader.open()

    count = 0
    start = time.monotonic()

    try:

        while time.monotonic() - start < 5:

            frame = reader.read_frame()

            if frame is None:
                continue

            count += 1

            if count == 1:
                print(
                    f"{name}: first CSI frame received"
                )

            if count % 50 == 0:
                print(
                    f"{name}: {count} frames"
                )

    finally:

        reader.close()

    elapsed = time.monotonic() - start

    print(
        f"{name}: TOTAL = {count} frames "
        f"in {elapsed:.2f} sec"
    )


def main():

    print("=" * 60)
    print("ESP32-S3 CSI CONNECTION TEST")
    print("=" * 60)

    test_reader(
        "FRIEND",
        FRIEND_PORT
    )

    test_reader(
        "YOU",
        YOUR_PORT
    )

    print("\nTEST COMPLETE")


if __name__ == "__main__":
    main()
