"""
config.py
---------
Single source of truth for hardware/serial settings.
"""

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 921600          # confirmed from working serial_reader.py

BASELINE_FRAMES = 1000      # frames collected during Room Setup
WINDOW_SIZE = 50            # frames per live detection window
WINDOW_STRIDE = 10          # how far the live window slides each step