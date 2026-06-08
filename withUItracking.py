"""Hand tracking -> serial mouse control with gesture recognition + touch UI"""
import time
import serial
import serial.tools.list_ports
import sys
import cv2
import numpy as np
import mediapipe as mp
import pickle
import threading
import tkinter as tk
from collections import deque
from picamera2 import Picamera2
from libcamera import controls
from PIL import Image, ImageTk
import queue

frame_queue = queue.Queue(maxsize=2)

config = {
    "PROCESS_NOISE":         0.40,
    "MEASUREMENT_NOISE":     4.0,
    "ESTIMATION_ERROR":      1.0,
    "SCREEN_WIDTH":          1920,
    "SCREEN_HEIGHT":         1080,
    "PINCH_CLICK_THRESHOLD": 0.20,
    "ACTIVE_ZONE": 0.6,
}

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
BAUD         = 115200

CONFIDENCE_THRESHOLD   = 0.85
SMOOTHING_WINDOW       = 7
ACTION_COOLDOWN_FRAMES = 20
SCROLL_COOLDOWN_FRAMES = 8
PINCH_HOLD_FRAMES      = 4
PINCH_COOLDOWN_FRAMES  = 8

RESOLUTIONS = [
    ("1280x720",  1280,  720),
    ("1920x1080", 1920, 1080),
    ("2560x1440", 2560, 1440),
    ("3840x2160", 3840, 2160),
]

state = {
    "running":       True,
    "left_tracked":  False,
    "right_tracked": False,
    "gesture":       "---",
    "pinch_dist":    0.0,
    "connected":     False,
}
state_lock = threading.Lock()

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils


class KalmanFilter1D:
    def __init__(self):
        self.p = None
        self.x = None

    def update(self, measurement):
        q = config["PROCESS_NOISE"]
        r = config["MEASUREMENT_NOISE"]
        if self.x is None:
            self.x = measurement
            self.p = config["ESTIMATION_ERROR"]
            return measurement
        self.p = self.p + q
        k      = self.p / (self.p + r)
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * self.p
        return self.x


with open("gesture_model.pkl", "rb") as f:
    data = pickle.load(f)
gesture_model   = data["model"]
gesture_scaler  = data["scaler"]
gesture_encoder = data["encoder"]

GESTURE_IDLE        = "both_hands_flat"
GESTURE_CLICK       = "left_hand_fist"
GESTURE_SCROLL_DOWN = "left_hand_peace"
GESTURE_SCROLL_UP   = "right_hand_peace"


def normalise_hand(hand_landmarks):
    coords     = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    wrist      = coords[0]
    normalised = [(x - wrist[0], y - wrist[1], z - wrist[2]) for x, y, z in coords]
    scale      = ((normalised[9][0])**2 + (normalised[9][1])**2) ** 0.5
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

def predict_gesture(features):
    scaled = gesture_scaler.transform([features])
    proba  = gesture_model.predict_proba(scaled)[0]
    conf   = proba.max()
    label  = gesture_encoder.inverse_transform([proba.argmax()])[0]
    return label, conf

def get_pinch_distance(hand_landmarks):
    lms   = hand_landmarks.landmark
    tip   = lms[8]
    thumb = lms[4]
    wrist = lms[0]
    mid   = lms[9]
    dx    = tip.x - thumb.x
    dy    = tip.y - thumb.y
    dist  = (dx**2 + dy**2) ** 0.5
    sx    = mid.x - wrist.x
    sy    = mid.y - wrist.y
    scale = (sx**2 + sy**2) ** 0.5
    if scale == 0:
        return None
    return dist / scale


def find_serial_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "ttyAMA0" in port.device:
            return port.device
    for port in ports:
        if "ttyUSB" in port.device:
            return port.device
    if ports:
        return ports[0].device
    print("[ERROR] No serial ports found.")
    sys.exit(1)

ser = serial.Serial(find_serial_port(), BAUD, timeout=1)
time.sleep(2)
with state_lock:
    state["connected"] = True

def send(cmd):
    ser.write((cmd + "\n").encode("utf-8"))


# =======================================================================
# CAMERA LOOP — background thread
# =======================================================================
def camera_loop():
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
        )
    )
    picam2.start()
    picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    time.sleep(3)

    kf_x = KalmanFilter1D()
    kf_y = KalmanFilter1D()

    cursor_x, cursor_y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    prev_sx, prev_sy   = -1, -1

    prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
    smoothed_gesture  = None
    action_cooldown   = 0
    click_held        = False

    pinch_cooldown     = 0
    pinch_was_held     = False
    pinch_hold_counter = 0
    pinch_is_holding   = False
    pinch_clicked      = False
    pinch_tap_count    = 0      # counts quick taps in the double-click window
    pinch_tap_timer    = 0      # frames since first tap
    PINCH_DOUBLE_TAP_WINDOW = 10  # frames within which second tap must arrive (~0.6s at 30fps)

    with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_num_hands=2,
    ) as hands:
        while state["running"]:
            frame = picam2.capture_array()
            if frame is None or frame.size == 0:
                continue

            frame = cv2.flip(frame, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame)
            display = frame.copy()

            left_tracked    = False
            right_tracked   = False
            current_gesture = None
            pinch_dist_val  = 0.0

            both_hands = (
                results.multi_hand_landmarks is not None and
                len(results.multi_hand_landmarks) == 2
            )

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness or []
                ):
                    mp_draw.draw_landmarks(
                        display, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(150, 150, 150), thickness=1, circle_radius=2),
                        mp_draw.DrawingSpec(color=(150, 150, 150), thickness=1),
                    )

                    label = handedness.classification[0].label

                    if label == "Right":
                        right_tracked = True
                        tip   = hand_landmarks.landmark[8]
                        raw_x = tip.x * FRAME_WIDTH
                        raw_y = tip.y * FRAME_HEIGHT
                        cursor_x = int(kf_x.update(raw_x))
                        cursor_y = int(kf_y.update(raw_y))
                        zone    = config["ACTIVE_ZONE"]
                        margin_x = FRAME_WIDTH  * (1 - zone) / 2
                        margin_y = FRAME_HEIGHT * (1 - zone) / 2

                        # Remap cursor position from active zone to full screen
                        norm_x = (cursor_x - margin_x) / (FRAME_WIDTH  * zone)
                        norm_y = (cursor_y - margin_y) / (FRAME_HEIGHT * zone)

                        # Clamp so edges of screen are still reachable
                        norm_x = max(0.0, min(1.0, norm_x))
                        norm_y = max(0.0, min(1.0, norm_y))

                        screen_x = int(norm_x * config["SCREEN_WIDTH"])
                        screen_y = int(norm_y * config["SCREEN_HEIGHT"])
                        if screen_x != prev_sx or screen_y != prev_sy:
                            send(f"MOVE {screen_x} {screen_y}")
                            prev_sx, prev_sy = screen_x, screen_y

                    elif label == "Left":
                        left_tracked = True
                        pinch_dist = get_pinch_distance(hand_landmarks)

                        if pinch_dist is not None:
                            pinch_dist_val = pinch_dist
                            is_pinching = pinch_dist < config["PINCH_CLICK_THRESHOLD"]

                            tip_px   = (int(hand_landmarks.landmark[8].x * FRAME_WIDTH),
                                        int(hand_landmarks.landmark[8].y * FRAME_HEIGHT))
                            thumb_px = (int(hand_landmarks.landmark[4].x * FRAME_WIDTH),
                                        int(hand_landmarks.landmark[4].y * FRAME_HEIGHT))

                            if pinch_is_holding:
                                colour = (255, 0, 0)
                            elif is_pinching:
                                colour = (255, 165, 0)
                            else:
                                colour = (120, 255, 0)

                            cv2.line(display, tip_px, thumb_px, colour, 2)
                            cv2.circle(display, tip_px,   6, colour, -1)
                            cv2.circle(display, thumb_px, 6, colour, -1)

                            mid_px = ((tip_px[0] + thumb_px[0]) // 2,
                                      (tip_px[1] + thumb_px[1]) // 2)
                            cv2.putText(display, f"{pinch_dist:.2f}", mid_px,
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

                            if is_pinching and not pinch_is_holding:
                                progress = pinch_hold_counter / PINCH_HOLD_FRAMES
                                bar_w = 60
                                bar_x = mid_px[0] - bar_w // 2
                                bar_y = mid_px[1] + 14
                                cv2.rectangle(display, (bar_x, bar_y),
                                              (bar_x + bar_w, bar_y + 6), (60, 60, 60), -1)
                                cv2.rectangle(display, (bar_x, bar_y),
                                              (bar_x + int(bar_w * progress), bar_y + 6),
                                              (255, 165, 0), -1)

                            if pinch_cooldown > 0:
                                pinch_cooldown -= 1
                            elif is_pinching:
                                pinch_hold_counter += 1
                                if pinch_hold_counter >= PINCH_HOLD_FRAMES and not pinch_is_holding:
                                    send("CLICK LEFT")
                                    pinch_is_holding = True
                                    pinch_clicked    = False
                            else:
                                if pinch_was_held:
                                    if pinch_is_holding:
                                        # Was a long hold — release
                                        send("RELEASE LEFT")
                                        pinch_is_holding = False
                                        pinch_tap_count  = 0
                                        pinch_tap_timer  = 0
                                    else:
                                        # Was a quick tap
                                        pinch_tap_count += 1

                                        if pinch_tap_count == 1:
                                            # First tap — start the double-click window, don't fire yet
                                            pinch_tap_timer = PINCH_DOUBLE_TAP_WINDOW

                                        elif pinch_tap_count >= 2:
                                            # Second tap within window — double click
                                            send("DCLICK LEFT")
                                            print("[ACTION] Double click")
                                            pinch_tap_count = 0
                                            pinch_tap_timer = 0
                                            pinch_cooldown  = PINCH_COOLDOWN_FRAMES

                                pinch_hold_counter = 0
                                pinch_was_held     = False
                                pinch_clicked      = False

                            # Tick down the double-click window and fire single click if it expires
                            if pinch_tap_timer > 0:
                                pinch_tap_timer -= 1
                                if pinch_tap_timer == 0 and pinch_tap_count == 1:
                                    # Window expired with only one tap — fire single click
                                    send("SINGLE_CLICK LEFT")
                                    print("[ACTION] Single click")
                                    pinch_tap_count = 0
                                    pinch_cooldown  = PINCH_COOLDOWN_FRAMES

                            if is_pinching:
                                pinch_was_held = True

                if both_hands:
                    features = get_two_hand_features(
                        results.multi_hand_landmarks,
                        results.multi_handedness
                    )
                    if features is not None:
                        raw_label, confidence = predict_gesture(features)
                        if confidence >= CONFIDENCE_THRESHOLD:
                            prediction_buffer.append(raw_label)
                            current_gesture = max(
                                set(prediction_buffer), key=prediction_buffer.count
                            )
                            smoothed_gesture = current_gesture
                else:
                    prediction_buffer.clear()

            with state_lock:
                state["left_tracked"]  = left_tracked
                state["right_tracked"] = right_tracked
                state["gesture"]       = smoothed_gesture or "---"
                state["pinch_dist"]    = pinch_dist_val

            if action_cooldown > 0:
                action_cooldown -= 1
            elif smoothed_gesture and smoothed_gesture != GESTURE_IDLE:
                if smoothed_gesture == GESTURE_CLICK:
                    if not click_held:
                        send("CLICK LEFT")
                        click_held      = True
                        action_cooldown = ACTION_COOLDOWN_FRAMES
                elif smoothed_gesture == GESTURE_SCROLL_DOWN:
                    send("SCROLL DOWN 3")
                    action_cooldown = SCROLL_COOLDOWN_FRAMES
                elif smoothed_gesture == GESTURE_SCROLL_UP:
                    send("SCROLL UP 3")
                    action_cooldown = SCROLL_COOLDOWN_FRAMES

            if click_held and smoothed_gesture != GESTURE_CLICK:
                send("RELEASE LEFT")
                click_held = False

            cv2.circle(display, (cursor_x, cursor_y), 14, (0, 0, 0), -1)
            cv2.circle(display, (cursor_x, cursor_y), 12, (0, 255, 255), -1)

            try:
                frame_queue.put_nowait(display.copy())
            except queue.Full:
                pass

    if click_held:
        send("RELEASE LEFT")
    picam2.stop()
    ser.close()


cam_thread = threading.Thread(target=camera_loop, daemon=True)
cam_thread.start()


# =======================================================================
# UI — main thread
# Layout: full-screen canvas preview with HUD overlaid
#         slide-in settings panel from the left
# =======================================================================
BG      = "#111111"
SURFACE = "#1a1a1a"
TEXT    = "#eeeeee"
MUTED   = "#888888"
GREEN   = "#22c55e"
RED     = "#dc2626"
BLUE    = "#378ADD"
ORANGE  = "#f97316"
FONT    = ("monospace", 11)
FONT_SM = ("monospace", 9)
FONT_LG = ("monospace", 13)

root = tk.Tk()
root.title("HandsOn")
root.configure(bg=BG)
root.overrideredirect(True)
root.update_idletasks()
SW = root.winfo_screenwidth()
SH = root.winfo_screenheight()
root.geometry(f"{SW}x{SH}+0+0")
root.update_idletasks()
root.lift()
root.focus_force()
root.bind("<Escape>", lambda e: on_close())

# ── Root layout: preview on right fills space, panel on left slides in ──
# We use a horizontal PanedWindow-style layout with place() so the
# settings panel can overlay without pushing the preview.

# Full-screen preview canvas — sits behind everything
preview_canvas = tk.Canvas(root, bg="#000000", highlightthickness=0)
preview_canvas.place(x=0, y=0, width=SW, height=SH)
preview_image_ref = [None]

def update_preview(rgb_frame):
    try:
        h, w    = rgb_frame.shape[:2]
        scale   = min(SW / w, SH / h)
        new_w   = int(w * scale)
        new_h   = int(h * scale)
        resized = cv2.resize(rgb_frame, (new_w, new_h))
        img     = Image.fromarray(resized)
        imgtk   = ImageTk.PhotoImage(image=img)
        # Centre the frame on the canvas
        x_off = (SW - new_w) // 2
        y_off = (SH - new_h) // 2
        preview_canvas.config(width=SW, height=SH)
        preview_canvas.create_image(x_off, y_off, anchor="nw", image=imgtk)
        preview_image_ref[0] = imgtk
    except Exception as e:
        print(f"[PREVIEW] {e}")

# ── HUD overlay — top bar pinned to top of screen ──
hud = tk.Frame(root, bg=BG)
hud.place(x=0, y=0, width=SW, height=44)

def on_close():
    with state_lock:
        state["running"] = False
    root.after(600, root.destroy)

# Settings toggle button (top-left)
PANEL_WIDTH = 260
panel_open  = tk.BooleanVar(value=False)

def toggle_panel(event=None):
    if panel_open.get():
        settings_panel.place_forget()
        panel_open.set(False)
        toggle_btn.config(text="> settings")
    else:
        settings_panel.place(x=0, y=0, width=PANEL_WIDTH, height=SH)
        panel_open.set(True)
        toggle_btn.config(text="< settings")

toggle_btn = tk.Button(hud, text="> settings", bg=SURFACE, fg=MUTED,
                       font=FONT, relief="flat", padx=10, pady=6,
                       command=toggle_panel)
toggle_btn.pack(side="left", padx=(8, 0), pady=4)

# Status indicators (centre)
left_dot  = tk.Label(hud, text="L", bg=BG, fg=MUTED, font=FONT_SM)
left_dot.pack(side="left", padx=(16, 2), pady=4)
right_dot = tk.Label(hud, text="R", bg=BG, fg=MUTED, font=FONT_SM)
right_dot.pack(side="left", padx=(0, 16), pady=4)

gesture_label = tk.Label(hud, text="---", bg=BG, fg=TEXT, font=FONT)
gesture_label.pack(side="left", padx=8, pady=4)

pinch_label = tk.Label(hud, text="pinch: ---", bg=BG, fg=MUTED, font=FONT_SM)
pinch_label.pack(side="left", padx=8, pady=4)

# Close button (top-right)
tk.Button(hud, text="X  close", bg=RED, fg="white",
          font=FONT, relief="flat", padx=12, pady=6,
          command=on_close).pack(side="right", padx=8, pady=4)

# ── Settings panel — slides in from left, overlays preview ──
settings_panel = tk.Frame(root, bg=SURFACE)
# Not placed yet — toggle_panel() places it

# Header inside panel
panel_header = tk.Frame(settings_panel, bg=SURFACE)
panel_header.pack(fill="x", padx=12, pady=(12, 0))
tk.Label(panel_header, text="HANDSON", bg=SURFACE, fg=MUTED,
         font=("monospace", 10, "bold")).pack(side="left")
tk.Button(panel_header, text="X", bg=SURFACE, fg=MUTED,
          font=FONT_SM, relief="flat", padx=6, pady=4,
          command=toggle_panel).pack(side="right")

tk.Frame(settings_panel, bg="#333333", height=1).pack(fill="x", padx=12, pady=8)

# Kalman sliders
tk.Label(settings_panel, text="KALMAN FILTER", bg=SURFACE, fg=MUTED,
         font=FONT_SM).pack(anchor="w", padx=12)

slider_frame = tk.Frame(settings_panel, bg=SURFACE)
slider_frame.pack(fill="x", padx=12, pady=(4, 0))
slider_frame.columnconfigure(1, weight=1)

def make_slider(parent, row, label, key, from_, to, resolution, fmt):
    tk.Label(parent, text=label, bg=SURFACE, fg=MUTED,
             font=FONT_SM, width=9, anchor="w").grid(
             row=row, column=0, sticky="w", pady=5)
    val_lbl = tk.Label(parent, text=fmt.format(config[key]),
                        bg=SURFACE, fg=TEXT, font=FONT_SM, width=5)
    val_lbl.grid(row=row, column=2, sticky="e")

    def on_change(v, k=key, fl=val_lbl, f=fmt):
        config[k] = float(v)
        fl.config(text=f.format(float(v)))

    s = tk.Scale(parent, from_=from_, to=to, resolution=resolution,
                 orient="horizontal", bg=SURFACE, fg=TEXT,
                 troughcolor="#333333", activebackground=BLUE,
                 highlightthickness=0, takefocus=1,
                 showvalue=False, command=on_change)
    s.set(config[key])
    s.grid(row=row, column=1, sticky="ew", padx=6)

make_slider(slider_frame, 0, "Q process",    "PROCESS_NOISE",         0.001, 0.5,  0.001, "{:.3f}")
make_slider(slider_frame, 1, "R measure",    "MEASUREMENT_NOISE",     0.1,   20.0, 0.1,   "{:.1f}")
make_slider(slider_frame, 2, "pinch",        "PINCH_CLICK_THRESHOLD", 0.01,  0.2,  0.01,  "{:.2f}")
make_slider(slider_frame, 3, "travel", "ACTIVE_ZONE", 0.2, 1.0, 0.05, "{:.2f}")

tk.Frame(settings_panel, bg="#333333", height=1).pack(fill="x", padx=12, pady=8)

# Resolution buttons
tk.Label(settings_panel, text="TARGET RESOLUTION", bg=SURFACE, fg=MUTED,
         font=FONT_SM).pack(anchor="w", padx=12)

res_outer = tk.Frame(settings_panel, bg=SURFACE)
res_outer.pack(fill="x", padx=12, pady=(6, 0))

res_buttons = {}

def set_resolution(w, h, label):
    config["SCREEN_WIDTH"]  = w
    config["SCREEN_HEIGHT"] = h
    for lbl, btn in res_buttons.items():
        btn.config(bg="#2a2a2a", fg=MUTED)
    res_buttons[label].config(bg=BLUE, fg="white")

for i, (label, w, h) in enumerate(RESOLUTIONS):
    is_current = (w == config["SCREEN_WIDTH"] and h == config["SCREEN_HEIGHT"])
    btn = tk.Button(res_outer, text=label,
                    bg=BLUE if is_current else "#2a2a2a",
                    fg="white" if is_current else MUTED,
                    font=FONT_SM, relief="flat", bd=0,
                    pady=10, cursor="hand2",
                    command=lambda w=w, h=h, l=label: set_resolution(w, h, l))
    btn.pack(fill="x", pady=2)
    res_buttons[label] = btn

tk.Frame(settings_panel, bg="#333333", height=1).pack(fill="x", padx=12, pady=8)

# Hand tracking status inside panel
tk.Label(settings_panel, text="TRACKING", bg=SURFACE, fg=MUTED,
         font=FONT_SM).pack(anchor="w", padx=12)

track_frame = tk.Frame(settings_panel, bg=SURFACE)
track_frame.pack(fill="x", padx=12, pady=(6, 0))

panel_left_lbl  = tk.Label(track_frame, text="left   waiting...",
                             bg=SURFACE, fg=MUTED, font=FONT_SM, anchor="w")
panel_left_lbl.pack(fill="x", pady=2)
panel_right_lbl = tk.Label(track_frame, text="right  waiting...",
                             bg=SURFACE, fg=MUTED, font=FONT_SM, anchor="w")
panel_right_lbl.pack(fill="x", pady=2)


# ── Poll loop ──
def poll():
    latest = None
    while True:
        try:
            latest = frame_queue.get_nowait()
        except queue.Empty:
            break
    if latest is not None:
        update_preview(latest)

    with state_lock:
        lt = state["left_tracked"]
        rt = state["right_tracked"]
        g  = state["gesture"]
        pd = state["pinch_dist"]

    # HUD updates
    left_dot.config(fg=GREEN if lt else MUTED)
    right_dot.config(fg=GREEN if rt else MUTED)
    gesture_label.config(text=g if g else "---")
    pinch_label.config(
        text=f"pinch: {pd:.2f}",
        fg=ORANGE if pd < config["PINCH_CLICK_THRESHOLD"] else MUTED
    )

    # Panel tracking labels
    panel_left_lbl.config(
        text=f"left   {'tracking' if lt else 'waiting...'}",
        fg=GREEN if lt else MUTED
    )
    panel_right_lbl.config(
        text=f"right  {'tracking' if rt else 'waiting...'}",
        fg=GREEN if rt else MUTED
    )

    if state["running"]:
        root.after(33, poll)

poll()
root.mainloop()

with state_lock:
    state["running"] = False

