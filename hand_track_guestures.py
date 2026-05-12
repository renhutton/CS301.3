import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from picamera2 import Picamera2
import time
from mediapipe.framework.formats import landmark_pb2

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
))
picam2.start()

base_options = python.BaseOptions(
    model_asset_path='/home/handson2/Documents/gesture_recognizer.task'
)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=2
)
recognizer = vision.GestureRecognizer.create_from_options(options)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

print("Camera started, detecting gestures... (Ctrl+C to stop)")

prev_time = time.time()

while True:
    frame = picam2.capture_array()

    # Convert to MediaPipe image format
    frame = cv2.flip(frame, 0)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = recognizer.recognize(mp_image)

    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    

    # Draw landmarks and gestures
    if result.hand_landmarks:
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            
            landmark_list = landmark_pb2.NormalizedLandmarkList()
            landmark_list.landmark.extend([
                landmark_pb2.NormalizedLandmark(
                    x=lm.x, y=lm.y, z=lm.z
                ) for lm in hand_landmarks
            ])
            mp_draw.draw_landmarks(bgr_frame, landmark_list, mp_hands.HAND_CONNECTIONS)

            # Get gesture for this hand
            if result.gestures and i < len(result.gestures):
                gesture = result.gestures[i][0]
                label = gesture.category_name
                score = gesture.score
                print(f"Hand {i+1}: {label} ({score:.2f})")

                # Draw gesture label on frame
                h, w, _ = bgr_frame.shape
                wrist = hand_landmarks[0]
                x = int(wrist.x * w)
                y = int(wrist.y * h) - 20
                cv2.putText(bgr_frame, f"{label} ({score:.2f})",
                            (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(bgr_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", bgr_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
