import cv2
import mediapipe as mp
from picamera2 import Picamera2
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
))
picam2.start()

print("Camera started, tracking hands... (Ctrl+C to stop)")

prev_time = time.time()

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:
    while True:
        frame = picam2.capture_array()
        results = hands.process(frame)

        if results.multi_hand_landmarks:
            for i, hand_lms in enumerate(results.multi_hand_landmarks):
                wrist = hand_lms.landmark[mp_hands.HandLandmark.WRIST]
                index_tip = hand_lms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                print(f"Hand {i+1} | Wrist: ({wrist.x:.2f}, {wrist.y:.2f}) | Index tip: ({index_tip.x:.2f}, {index_tip.y:.2f})")
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw FPS on frame
        cv2.putText(bgr_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Tracking", bgr_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

picam2.stop()
cv2.destroyAllWindows()
