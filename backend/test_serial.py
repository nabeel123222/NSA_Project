from serial_reader import CSISerialReader

reader = CSISerialReader()

with reader:
    print("Waiting for CSI data...\n")

    while True:
        frame = reader.read_frame()
        print(
            f"RSSI={frame.rssi} | "
            f"Subcarriers={len(frame.complex_csi)}"
        )