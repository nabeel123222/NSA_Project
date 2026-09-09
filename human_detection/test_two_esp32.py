import threading
import time
from serial_reader import CSISerialReader


PORTS = {
    "ESP32_FRIEND": "/dev/ttyACM0",
    "ESP32_YOU": "/dev/ttyACM1",
}

BAUDRATE = 921600

results = {
    "ESP32_FRIEND": {"frames": 0, "last_subcarriers": 0},
    "ESP32_YOU": {"frames": 0, "last_subcarriers": 0},
}

stop_event = threading.Event()


def read_esp32(name, port):
    print(f"[{name}] Connecting to {port}...")

    try:
        reader = CSISerialReader(
            port=port,
            baudrate=BAUDRATE,
            timeout=2.0,
        )
        reader.open()
        print(f"[{name}] CONNECTED to {port}")

        while not stop_event.is_set():
            frame = reader.read_frame()

            if frame is None:
                continue

            subcarriers = frame.complex_csi.size

            results[name]["frames"] += 1
            results[name]["last_subcarriers"] = subcarriers

        reader.close()

    except Exception as e:
        print(f"[{name}] ERROR: {e}")


def display_status():
    while not stop_event.is_set():
        time.sleep(2)

        friend = results["ESP32_FRIEND"]
        you = results["ESP32_YOU"]

        print()
        print("=" * 60)
        print("TWO ESP32 CSI STATUS")
        print("=" * 60)

        print(
            f"Friend ESP32 (/dev/ttyACM0): "
            f"{friend['frames']} frames | "
            f"{friend['last_subcarriers']} subcarriers"
        )

        print(
            f"Your ESP32 (/dev/ttyACM1):    "
            f"{you['frames']} frames | "
            f"{you['last_subcarriers']} subcarriers"
        )

        print("=" * 60)


threads = []

for name, port in PORTS.items():
    thread = threading.Thread(
        target=read_esp32,
        args=(name, port),
        daemon=True,
    )

    threads.append(thread)
    thread.start()


status_thread = threading.Thread(
    target=display_status,
    daemon=True,
)

status_thread.start()


print()
print("Two-ESP32 CSI test started.")
print("Press Ctrl+C to stop.")
print()

try:
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopping...")
    stop_event.set()

    time.sleep(1)

    print()
    print("FINAL RESULT")
    print("=" * 60)

    for name, data in results.items():
        print(
            f"{name}: "
            f"{data['frames']} frames | "
            f"last = {data['last_subcarriers']} subcarriers"
        )

    print("=" * 60)
