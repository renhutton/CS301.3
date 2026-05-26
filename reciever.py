#sender.py — Run this on the Raspberry Pi
#Install: pip install mediapipe opencv-python picamera2 pyserial

import time
import cv2
import serial
import serial.tools.list_ports
import mediapipe as mp
from picamera2 import Picamera2
from libcamera import controls
import sys

#Camera config
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

#1080p screen resolution of reciever pc
SCREEN_WIDTH  = 1920
SCREEN_HEIGHT = 1080

#cursor smoothing (0 = no smoothing, 1 = never moves)
SMOOTHING = 0.35

#Serial config
BAUD = 115200
#set to None to auto-detect, or hardcode e.g. "/dev/ttyUSB0" / "/dev/ttyACM0"
SERIAL_PORT = None

#display
SHOW_PREVIEW   = True   #set False to run headless (faster on Pi)
CURSOR_RADIUS  = 12

#Send rate limit (seconds between MOVE commands) ---
#Keeps the serial buffer from flooding — 0.03 ≈ 30 moves/sec
SEND_INTERVAL  = 0.03


def find_serial_port() -> str:
    """Auto-detect the USB serial port connected to the Windows machine."""
    if SERIAL_PORT:
        return SERIAL_PORT

    ports = serial.tools.list_ports.comports()
    keywords = ["usb", "uart", "ch340", "cp210", "ftdi", "cdc", "serial"]
    for p in ports:
        desc = (p.description or "").lower()
        if any(kw in desc for kw in keywords):
            print(f"[AUTO] Using port: {p.device} — {p.description}")
            return p.device

    if ports:
        print("\n[WARN] Could not auto-detect port. Available:")
        for i, p in enumerate(ports):
            print(f"  [{i}] {p.device} — {p.description}")
        choice = input("Enter port number: ").strip()
        return ports[int(choice)].device

    print("[ERROR] No serial ports found.")
    sys.exit(1)


def scale(cx, cy) -> tuple[int, int]:
    """Map camera-space coordinates to Windows screen coordinates."""
    sx = int(cx * SCREEN_WIDTH  / FRAME_WIDTH)
    sy = int(cy * SCREEN_HEIGHT / FRAME_HEIGHT)
    #Clamp to screen bounds
    sx = max(0, min(SCREEN_WIDTH  - 1, sx))
    sy = max(0, min(SCREEN_HEIGHT - 1, sy))
    return sx, sy


def send(ser: serial.Serial, cmd: str):
    """Write a newline-terminated command over serial."""
    ser.write((cmd + "\n").encode("utf-8"))


def main():
    #Serial
    port = find_serial_port()
    print(f"Connecting to {port} at {BAUD} baud...")
    try:
        ser = serial.Serial(port, BAUD, timeout=1)
    except serial.SerialException as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    time.sleep(2)  #let the serial connection settle

    #Camera
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
        )
    )
    picam2.start()
    picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    print("Camera started — warming up (3 s)...")
    time.sleep(3)

    #MediaPipe config
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils

    cursor_x, cursor_y   = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    last_send_time: float = 0.0

    print("Tracking started. Press 'q' in preview window (or Ctrl-C) to stop.\n")

    with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_num_hands=1,
    ) as hands:
        try:
            while True:
                frame = picam2.capture_array()
                if frame is None or frame.size == 0:
                    continue

                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                display = frame.copy() if SHOW_PREVIEW else None

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]

                    # Landmark 8 = index finger tip
                    tip      = hand.landmark[8]
                    target_x = int(tip.x * FRAME_WIDTH)
                    target_y = int(tip.y * FRAME_HEIGHT)

                    # Exponential smoothing
                    alpha    = 1 - SMOOTHING
                    cursor_x = int(cursor_x + alpha * (target_x - cursor_x))
                    cursor_y = int(cursor_y + alpha * (target_y - cursor_y))

                    # Rate-limited serial send
                    now = time.monotonic()
                    if now - last_send_time >= SEND_INTERVAL:
                        sx, sy = scale(cursor_x, cursor_y)
                        send(ser, f"MOVE {sx} {sy}")
                        last_send_time = now

                    if SHOW_PREVIEW:
                        mp_draw.draw_landmarks(
                            display, hand, mp_hands.HAND_CONNECTIONS,
                            mp_draw.DrawingSpec(color=(150, 150, 150), thickness=1, circle_radius=2),
                            mp_draw.DrawingSpec(color=(150, 150, 150), thickness=1),
                        )

                if SHOW_PREVIEW:
                    cv2.circle(display, (cursor_x, cursor_y), CURSOR_RADIUS + 2, (0, 0, 0), -1)
                    cv2.circle(display, (cursor_x, cursor_y), CURSOR_RADIUS, (0, 255, 255), -1)
                    cv2.imshow("HandsOn - Cursor", display)
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            picam2.stop()
            ser.close()
            if SHOW_PREVIEW:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
