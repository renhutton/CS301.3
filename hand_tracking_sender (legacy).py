"""Hand tracking → serial mouse control (Kalman filtered)"""
import time
import serial
import serial.tools.list_ports
import sys
import cv2
import numpy as np
import mediapipe as mp
from picamera2 import Picamera2
from libcamera import controls

# --- Config ---
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
BAUD          = 115200
SCREEN_WIDTH  = 1920
SCREEN_HEIGHT = 1080

# Kalman tuning — increase PROCESS_NOISE for more responsive (jittery),
# decrease for smoother (laggy)
PROCESS_NOISE      = 0.03   # Q — how much we trust hand movement
MEASUREMENT_NOISE  = 3.0    # R — how much we trust the raw landmark
ESTIMATION_ERROR   = 1.0    # P — initial uncertainty

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils


class KalmanFilter1D:
    """Independent Kalman filter for a single axis (x or y)."""

    def __init__(self, process_noise, measurement_noise, estimation_error):
        self.q = process_noise
        self.r = measurement_noise
        self.p = estimation_error
        self.x = None          # estimated state (unknown until first measurement)

    def update(self, measurement):
        # Initialise on first measurement
        if self.x is None:
            self.x = measurement
            return measurement

        # Predict
        self.p = self.p + self.q

        # Update (Kalman gain)
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * self.p

        return self.x


# --- Serial ---
def find_serial_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "ttyAMA0" in port.device:
            print(f"[AUTO] Found hardware UART: {port.device}")
            return port.device
    for port in ports:
        if "ttyUSB" in port.device:
            print(f"[AUTO] Found USB serial: {port.device}")
            return port.device
    if ports:
        print("\n[WARN] Could not auto-detect. Available ports:")
        for i, p in enumerate(ports):
            print(f"  [{i}] {p.device} — {p.description}")
        return ports[int(input("Enter port number: "))].device
    print("[ERROR] No serial ports found.")
    sys.exit(1)

ser = serial.Serial(find_serial_port(), BAUD, timeout=1)
time.sleep(2)
print("[OK] Serial connected.")

def send(cmd):
    ser.write((cmd + "\n").encode("utf-8"))


# --- Camera ---
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
    )
)
picam2.start()
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
time.sleep(3)

# One filter per axis
kf_x = KalmanFilter1D(PROCESS_NOISE, MEASUREMENT_NOISE, ESTIMATION_ERROR)
kf_y = KalmanFilter1D(PROCESS_NOISE, MEASUREMENT_NOISE, ESTIMATION_ERROR)

cursor_x, cursor_y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
prev_sx, prev_sy   = -1, -1

with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    max_num_hands=1,
) as hands:
    while True:
        frame = picam2.capture_array()
        if frame is None or frame.size == 0:
            continue

        frame   = cv2.flip(frame, -0)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        display = frame.copy()

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            tip  = hand.landmark[8]

            # Raw landmark → pixel coords
            raw_x = tip.x * FRAME_WIDTH
            raw_y = tip.y * FRAME_HEIGHT

            # Kalman filter each axis independently
            cursor_x = int(kf_x.update(raw_x))
            cursor_y = int(kf_y.update(raw_y))

            # Map to screen
            screen_x = int((cursor_x / FRAME_WIDTH)  * SCREEN_WIDTH)
            screen_y = int((cursor_y / FRAME_HEIGHT) * SCREEN_HEIGHT)

            if screen_x != prev_sx or screen_y != prev_sy:
                send(f"MOVE {screen_x} {screen_y}")
                prev_sx, prev_sy = screen_x, screen_y

            mp_draw.draw_landmarks(
                display, hand, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(150, 150, 150), thickness=1, circle_radius=2),
                mp_draw.DrawingSpec(color=(150, 150, 150), thickness=1),
            )

        cv2.circle(display, (cursor_x, cursor_y), 14, (0, 0, 0), -1)
        cv2.circle(display, (cursor_x, cursor_y), 12, (0, 255, 255), -1)
        cv2.imshow("HandsOn - Cursor", display)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

picam2.stop()
cv2.destroyAllWindows()
ser.close()
