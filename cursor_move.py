"""Smoothing virtual cursor"""
import time
import cv2
import mediapipe as mp
from picamera2 import Picamera2
from libcamera import controls

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SMOOTHING = 0.35
CURSOR_RADIUS = 12

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
    )
)
picam2.start()

# Camera Module 3: continuous autofocus
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
time.sleep(3)  # warm-up + AF settle

cursor_x, cursor_y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2

with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    max_num_hands=1,
) as hands:
    while True:
        frame_bgr = picam2.capture_array()
        if frame_bgr is None or frame_bgr.size == 0:
            continue

        frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        display = frame_bgr.copy()

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            tip = hand.landmark[8]
            target_x = int(tip.x * FRAME_WIDTH)
            target_y = int(tip.y * FRAME_HEIGHT)

            alpha = 1 - SMOOTHING
            cursor_x = int(cursor_x + alpha * (target_x - cursor_x))
            cursor_y = int(cursor_y + alpha * (target_y - cursor_y))

            mp_draw.draw_landmarks(
                display, hand, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(150, 150, 150), thickness=1, circle_radius=2),
                mp_draw.DrawingSpec(color=(150, 150, 150), thickness=1),
            )

        cv2.circle(display, (cursor_x, cursor_y), CURSOR_RADIUS + 2, (0, 0, 0), -1)
        cv2.circle(display, (cursor_x, cursor_y), CURSOR_RADIUS, (0, 255, 255), -1)

        cv2.imshow("HandsOn - Cursor", display)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

picam2.stop()
cv2.destroyAllWindows()
