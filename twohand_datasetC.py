import cv2
import mediapipe as mp
import csv
import time
from picamera2 import Picamera2
from libcamera import controls

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,        # changed from 1
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

GESTURE_LABEL = "right_hand_peace"
OUTPUT_CSV = "gesture_data.csv"
SAMPLES_PER_GESTURE = 200

# --- Config ---
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480

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

def normalise_hand(hand_landmarks):
    coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    wrist = coords[0]
    normalised = [(x - wrist[0], y - wrist[1], z - wrist[2]) for x, y, z in coords]
    scale = ((normalised[9][0])**2 + (normalised[9][1])**2) ** 0.5
    if scale == 0:
        return None
    return [v / scale for xyz in normalised for v in xyz]

def get_two_hand_features(multi_hand_landmarks, multi_handedness):
    left = None
    right = None

    for hand_landmarks, handedness in zip(multi_hand_landmarks, multi_handedness):
        # MediaPipe labels are mirrored on front camera, so Left/Right are swapped
        label = handedness.classification[0].label
        features = normalise_hand(hand_landmarks)
        if features is None:
            return None
        if label == "Left":
            right = features   # mirrored — "Left" from MediaPipe = user's right
        else:
            left = features

    # Only return if both hands are present
    if left is None or right is None:
        return None

    # Concatenate: left hand (63) + right hand (63) = 126 features
    return left + right

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(2)
for _ in range(10):
    cap.read()

count = 0
collecting = False

with open(OUTPUT_CSV, "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        frame = picam2.capture_array()
        if frame is None:
            continue

        frame = cv2.flip(frame, -0)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        both_hands_visible = (
            result.multi_hand_landmarks is not None and
            len(result.multi_hand_landmarks) == 2
        )

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if collecting and both_hands_visible:
                features = get_two_hand_features(
                    result.multi_hand_landmarks,
                    result.multi_handedness
                )
                if features is not None:
                    writer.writerow([GESTURE_LABEL] + features)
                    count += 1

        # Status display
        hand_count = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
        if not both_hands_visible:
            cv2.putText(frame, f"Show both hands ({hand_count}/2)", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        status = f"Collecting: {count}/{SAMPLES_PER_GESTURE}" if collecting else "Press SPACE to start"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Collect Gestures", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            collecting = True
        if count >= SAMPLES_PER_GESTURE:
            print(f"Done collecting '{GESTURE_LABEL}' — {count} samples saved")
            break
        if key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
