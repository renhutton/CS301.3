import cv2
import mediapipe as mp
from picamera2 import Picamera2

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
)
picam2.start()

with mp_hands.Hands(min_detection_confidence=0.7) as hands:
    while True:
        frame = picam2.capture_array()
        results = hands.process(frame)

        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    display, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        cv2.imshow("HandsOn", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

picam2.stop()
cv2.destroyAllWindows()
