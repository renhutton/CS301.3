import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import deque
from picamera2 import Picamera2
from libcamera import controls
import time

# ── Load model ────────────────────────────────────────────────────────────────

with open("gesture_model.pkl", "rb") as f:
    data = pickle.load(f)

model   = data["model"]
scaler  = data["scaler"]
le      = data["encoder"]
# --- Config ---
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480

GESTURE_CLASSES = list(le.classes_)# --- Config ---
NUM_CLASSES     = len(GESTURE_CLASSES)

# ── MediaPipe ─────────────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

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

# ── Config ────────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.85     # minimum confidence to count a prediction
SMOOTHING_WINDOW     = 7        # frames to majority-vote over
CURRENT_GESTURE_IDX  = 0        # index into GESTURE_CLASSES — change with arrow keys

# ── State ─────────────────────────────────────────────────────────────────────

prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)

session_stats = {cls: {"correct": 0, "total": 0} for cls in GESTURE_CLASSES}

# ── Helpers ───────────────────────────────────────────────────────────────────

def normalise_hand(hand_landmarks):
    coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    wrist = coords[0]
    normalised = [(x - wrist[0], y - wrist[1], z - wrist[2]) for x, y, z in coords]
    scale = ((normalised[9][0])**2 + (normalised[9][1])**2) ** 0.5
    if scale == 0:
        return None
    return [v / scale for xyz in normalised for v in xyz]

def get_two_hand_features(multi_hand_landmarks, multi_handedness):
    left = right = None
    for hand_landmarks, handedness in zip(multi_hand_landmarks, multi_handedness):
        label    = handedness.classification[0].label
        features = normalise_hand(hand_landmarks)
        if features is None:
            return None
        if label == "Left":
            right = features
        else:
            left = features
    if left is None or right is None:
        return None
    return left + right

def get_smoothed_prediction(raw_label):
    prediction_buffer.append(raw_label)
    return max(set(prediction_buffer), key=prediction_buffer.count)

def draw_stats_panel(frame, current_gesture, smoothed_label, confidence, stats):
    h, w = frame.shape[:2]
    panel_w = 280
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 30
    cv2.putText(frame, "Accuracy by Gesture", (w - panel_w + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    y += 10

    for cls in GESTURE_CLASSES:
        y += 28
        s = stats[cls]
        pct = (s["correct"] / s["total"] * 100) if s["total"] > 0 else 0
        is_current = (cls == current_gesture)

        colour = (0, 255, 100) if is_current else (150, 150, 150)
        prefix = "> " if is_current else "  "
        cv2.putText(frame, f"{prefix}{cls}", (w - panel_w + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        y += 18
        bar_max = panel_w - 30
        bar_len = int(bar_max * pct / 100)
        cv2.rectangle(frame, (w - panel_w + 10, y), (w - 20, y + 8), (60, 60, 60), -1)
        bar_colour = (0, 200, 80) if pct >= 90 else (0, 180, 255) if pct >= 70 else (0, 80, 255)
        if bar_len > 0:
            cv2.rectangle(frame, (w - panel_w + 10, y),
                          (w - panel_w + 10 + bar_len, y + 8), bar_colour, -1)
        cv2.putText(frame, f"{pct:.0f}% ({s['correct']}/{s['total']})",
                    (w - panel_w + 10, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        y += 24

    # Overall accuracy
    total_correct = sum(s["correct"] for s in stats.values())
    total_total   = sum(s["total"]   for s in stats.values())
    overall = (total_correct / total_total * 100) if total_total > 0 else 0
    y += 10
    cv2.line(frame, (w - panel_w + 10, y), (w - 10, y), (80, 80, 80), 1)
    y += 18
    cv2.putText(frame, f"Overall: {overall:.1f}%  ({total_correct}/{total_total})",
                (w - panel_w + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 100), 1)

def draw_hud(frame, current_gesture, smoothed_label, confidence, recording):
    # Prediction bar
    conf_pct = int(confidence * 100)
    colour   = (0, 255, 80) if smoothed_label == current_gesture else (0, 80, 255)
    cv2.putText(frame, f"Predicted: {smoothed_label or '---'}  ({conf_pct}%)",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

    # Current target gesture
    cv2.putText(frame, f"Perform:  {current_gesture}",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Recording indicator
    if recording:
        cv2.circle(frame, (15, 95), 7, (0, 0, 255), -1)
        cv2.putText(frame, "RECORDING", (28, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Controls hint
    cv2.putText(frame, "SPACE: record  UP/DOWN: switch gesture  Q: quit",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

# ── Camera ────────────────────────────────────────────────────────────────────


cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(2)
for _ in range(10):
    cap.read()

recording        = False
smoothed_label   = None
confidence       = 0.0

print("Controls:")
print("  SPACE     — toggle recording on/off")
print("  UP/DOWN   — switch target gesture")
print("  Q         — quit and print final results")

# ── Main loop ─────────────────────────────────────────────────────────────────

while True:
    frame = picam2.capture_array()
    if frame is None:
        continue

    frame = cv2.flip(frame, -0)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_gesture = GESTURE_CLASSES[CURRENT_GESTURE_IDX]
    raw_label       = None
    confidence      = 0.0

    both_hands = (
        result.multi_hand_landmarks is not None and
        len(result.multi_hand_landmarks) == 2
    )

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if both_hands:
            features = get_two_hand_features(
                result.multi_hand_landmarks,
                result.multi_handedness
            )
            if features is not None:
                scaled      = scaler.transform([features])
                proba       = model.predict_proba(scaled)[0]
                confidence  = proba.max()
                raw_label   = le.inverse_transform([proba.argmax()])[0]

                smoothed_label = get_smoothed_prediction(raw_label)

                # Only record if confidence is high enough
                if recording and confidence >= CONFIDENCE_THRESHOLD:
                    is_correct = (smoothed_label == current_gesture)
                    session_stats[current_gesture]["total"]   += 1
                    session_stats[current_gesture]["correct"] += int(is_correct)
        else:
            hand_count = len(result.multi_hand_landmarks)
            cv2.putText(frame, f"Need both hands ({hand_count}/2)", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

    draw_hud(frame, current_gesture, smoothed_label, confidence, recording)
    draw_stats_panel(frame, current_gesture, smoothed_label, confidence, session_stats)

    cv2.imshow("Gesture Accuracy Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        recording = not recording
        prediction_buffer.clear()
        print(f"Recording {'ON' if recording else 'OFF'} — gesture: {current_gesture}")
    elif key == 82:  # UP arrow
        CURRENT_GESTURE_IDX = (CURRENT_GESTURE_IDX - 1) % NUM_CLASSES
        recording = False
        prediction_buffer.clear()
    elif key == 84:  # DOWN arrow
        CURRENT_GESTURE_IDX = (CURRENT_GESTURE_IDX + 1) % NUM_CLASSES
        recording = False
        prediction_buffer.clear()
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# ── Final report ──────────────────────────────────────────────────────────────

print("\n── Final Accuracy Report ──────────────────────────────")
total_correct = total_total = 0
for cls, s in session_stats.items():
    pct = (s["correct"] / s["total"] * 100) if s["total"] > 0 else 0
    print(f"  {cls:<22} {pct:5.1f}%  ({s['correct']}/{s['total']})")
    total_correct += s["correct"]
    total_total   += s["total"]

overall = (total_correct / total_total * 100) if total_total > 0 else 0
print(f"\n  Overall: {overall:.1f}%  ({total_correct}/{total_total})")
