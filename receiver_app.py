# receiver_app.py — Windows background app with status overlay
#
# Install:
#   pip install pyserial pyautogui mouse pystray pillow uiautomation
#
# Run:
#   python receiver_app.py
#
# What it does:
#   - Auto-detects the Pi's serial port and streams MOVE/CLICK/SCROLL/KEY/TYPE
#     commands into mouse/keyboard actions (same protocol as receiver.py).
#   - Runs the serial listener on a background thread.
#   - Shows a small always-on-top overlay in the corner of the main monitor
#     with live connection status (color dot + text).
#   - Lives in the system tray; closing the window minimizes to tray instead
#     of quitting, so it keeps running in the background.

import sys
import threading
import time
import queue

import serial
import serial.tools.list_ports
import pyautogui
import mouse
import uiautomation as auto

import tkinter as tk
from tkinter import ttk

import pystray
from PIL import Image, ImageDraw

# --- Config ---
BAUD = 115200
PI_VENDOR_IDS = [0x2E8A, 0x0403, 0x10C4, 0x1A86, 0x067B]
RECONNECT_DELAY = 2.0  # seconds between reconnect attempts

pyautogui.FAILSAFE = True

# --- UI snapping config ---
SNAP_ENABLED_DEFAULT = True
SNAP_THRESHOLD = 45         # px — how close the cursor must be to snap
SNAP_INTERVAL = 0.08        # seconds between UI-tree queries

TITLEBAR_ZONE_HEIGHT = 60       # px from top of screen considered "title bar zone"
TITLEBAR_SNAP_THRESHOLD = 80    # wider snap radius inside that zone

# --- Smoothing config ---
# Instead of teleporting the cursor straight onto a control (which looks like
# jitter when the detected target flickers between neighbouring controls), we
# track a floating-point cursor position and ease it toward the desired target
# each update. The "pull" blends the raw position and the snap target so the
# cursor glides toward controls magnetically rather than jumping onto them.
MOVE_SMOOTHING_DEFAULT = 0.35   # 0 = no movement, 1 = instant jump (per update)
SNAP_PULL_STRENGTH = 0.55       # how strongly a detected control pulls the cursor toward it
CURSOR_RESYNC_THRESHOLD = 80    # px — if real cursor drifts this far from our tracked
                                # position (e.g. user grabbed the mouse), resync to it

SNAP_TYPES = {
    auto.ControlType.ButtonControl,
    auto.ControlType.EditControl,
    auto.ControlType.ComboBoxControl,
    auto.ControlType.CheckBoxControl,
    auto.ControlType.RadioButtonControl,
    auto.ControlType.MenuItemControl,
    auto.ControlType.ListItemControl,
    auto.ControlType.TabItemControl,
    auto.ControlType.HyperlinkControl,
    auto.ControlType.SliderControl,
    auto.ControlType.SpinnerControl,
}


def _rect_center(rect):
    return (rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2


def _distance(ax, ay, bx, by):
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _nearest_snap_in_children(parent, x, y, threshold):
    """Walk immediate children of parent to find the closest snappable control."""
    best_dist = threshold
    best_pos = None
    try:
        child = parent.GetFirstChildControl()
        while child:
            if child.ControlType in SNAP_TYPES:
                rect = child.BoundingRectangle
                if rect.right > rect.left and rect.bottom > rect.top:
                    cx, cy = _rect_center(rect)
                    d = _distance(x, y, cx, cy)
                    if d < best_dist:
                        best_dist = d
                        best_pos = (cx, cy)
            child = child.GetNextSiblingControl()
    except Exception:
        pass
    return best_pos


_last_snap_time = 0.0
_snap_cache = None  # last resolved snap target


def get_snap_target(x, y):
    """
    Return the centre of the nearest interactive UI control if the cursor is
    within the snap threshold of it, otherwise None.

    Uses a wider threshold near the top of the screen (title bar zone) and
    falls back to searching a non-interactive parent's children so minimize/
    maximize/close buttons inside browser chrome panes are still reachable.

    Queries are throttled to SNAP_INTERVAL seconds to avoid stalling the move loop.
    """
    global _last_snap_time, _snap_cache

    now = time.monotonic()
    if now - _last_snap_time < SNAP_INTERVAL:
        return _snap_cache

    _last_snap_time = now
    threshold = TITLEBAR_SNAP_THRESHOLD if y <= TITLEBAR_ZONE_HEIGHT else SNAP_THRESHOLD

    try:
        control = auto.ControlFromPoint(x, y)
        if control:
            if control.ControlType in SNAP_TYPES:
                rect = control.BoundingRectangle
                if rect.right > rect.left and rect.bottom > rect.top:
                    cx, cy = _rect_center(rect)
                    if _distance(x, y, cx, cy) <= threshold:
                        _snap_cache = (cx, cy)
                        return _snap_cache
            else:
                child_snap = _nearest_snap_in_children(control, x, y, threshold)
                if child_snap:
                    _snap_cache = child_snap
                    return _snap_cache
    except Exception:
        pass

    _snap_cache = None
    return None


# ---------------------------------------------------------------------------
# Serial worker — runs on a background thread, posts status + log updates
# through a thread-safe queue that the UI drains on its event loop.
# ---------------------------------------------------------------------------
class SerialWorker(threading.Thread):
    def __init__(self, event_queue: queue.Queue):
        super().__init__(daemon=True)
        self.event_queue = event_queue
        self._stop = threading.Event()
        self.held_buttons = set()
        self.snap_enabled = SNAP_ENABLED_DEFAULT
        self.move_smoothing = MOVE_SMOOTHING_DEFAULT
        self._cursor_pos = None  # tracked float (x, y) for smooth easing

    def stop(self):
        self._stop.set()

    def set_snap_enabled(self, enabled: bool):
        self.snap_enabled = enabled
        self._post("log", f"[SNAP] {'Enabled' if enabled else 'Disabled'}")
        self._post("snap", enabled)

    def set_move_smoothing(self, value: float):
        self.move_smoothing = max(0.05, min(1.0, value))

    def _post(self, kind, payload):
        self.event_queue.put((kind, payload))

    def find_pi_port(self):
        ports = serial.tools.list_ports.comports()

        for port in ports:
            if port.vid in PI_VENDOR_IDS:
                self._post("log", f"Found device by VID {hex(port.vid)}: {port.device} — {port.description}")
                return port.device

        keywords = ["raspberry", "pi", "usb serial", "uart", "ch340", "cp210", "ftdi", "cdc"]
        for port in ports:
            desc = (port.description or "").lower()
            if any(kw in desc for kw in keywords):
                self._post("log", f"Found device by description: {port.device} — {port.description}")
                return port.device

        return None

    def handle_command(self, line: str):
        parts = line.strip().split()
        if not parts:
            return

        cmd = parts[0].upper()

        if cmd == "MOVE" and len(parts) == 3:
            x, y = int(parts[1]), int(parts[2])
            raw_x = (x * 1.2) - 400
            raw_y = (y * 1.2) - 200

            target_x, target_y = raw_x, raw_y
            if self.snap_enabled:
                snap = get_snap_target(int(raw_x), int(raw_y))
                if snap:
                    sx, sy = snap
                    self._post("log", f"[SNAP] {raw_x:.0f},{raw_y:.0f} -> {sx},{sy}")
                    # Magnetic pull: blend toward the control rather than
                    # teleporting onto it — smooths out flicker between
                    # neighbouring controls and feels like a gentle drag.
                    target_x = raw_x + (sx - raw_x) * SNAP_PULL_STRENGTH
                    target_y = raw_y + (sy - raw_y) * SNAP_PULL_STRENGTH

            self._move_smoothly(target_x, target_y)

        elif cmd == "SINGLE_CLICK" and len(parts) == 2:
            mouse.click(button=parts[1].lower())

        elif cmd == "CLICK" and len(parts) == 2:
            btn = parts[1].lower()
            mouse.press(button=btn)
            self.held_buttons.add(btn)

        elif cmd == "RELEASE" and len(parts) == 2:
            btn = parts[1].lower()
            if btn in self.held_buttons:
                mouse.release(button=btn)
                self.held_buttons.discard(btn)

        elif cmd == "DCLICK" and len(parts) == 2:
            pyautogui.doubleClick(button=parts[1].lower())

        elif cmd == "SCROLL" and len(parts) == 3:
            direction = parts[1].upper()
            amount = int(parts[2])
            if direction == "UP":
                pyautogui.scroll(amount)
            elif direction == "DOWN":
                pyautogui.scroll(-amount)

        elif cmd == "KEY" and len(parts) >= 2:
            pyautogui.hotkey(*parts[1].lower().split("+"))

        elif cmd == "TYPE" and len(parts) >= 2:
            pyautogui.typewrite(" ".join(parts[1:]), interval=0.03)

        elif cmd == "PRESS" and len(parts) == 2:
            pyautogui.press(parts[1].lower())

        elif cmd == "SNAP" and len(parts) == 2:
            self.set_snap_enabled(parts[1].upper() == "ON")

        else:
            self._post("log", f"[WARN] Unknown command: {line.strip()}")

    def _move_smoothly(self, target_x, target_y):
        """Ease the cursor toward (target_x, target_y) instead of jumping.

        Keeps a floating-point estimate of the cursor position and steps it a
        fraction of the remaining distance toward the target on every MOVE
        event. This is what turns raw/snap coordinate jumps into a smooth
        glide. If the real cursor has drifted far from our tracked position
        (e.g. the user grabbed the mouse), we resync instead of fighting it.
        """
        try:
            actual_x, actual_y = mouse.get_position()
        except Exception:
            actual_x, actual_y = target_x, target_y

        if self._cursor_pos is None:
            self._cursor_pos = (float(actual_x), float(actual_y))
        elif _distance(actual_x, actual_y, *self._cursor_pos) > CURSOR_RESYNC_THRESHOLD:
            self._cursor_pos = (float(actual_x), float(actual_y))

        cx, cy = self._cursor_pos
        cx += (target_x - cx) * self.move_smoothing
        cy += (target_y - cy) * self.move_smoothing
        self._cursor_pos = (cx, cy)

        mouse.move(round(cx), round(cy))

    def release_all(self):
        for btn in list(self.held_buttons):
            try:
                mouse.release(button=btn)
                self._post("log", f"[CLEANUP] Released {btn}")
            except Exception:
                pass
        self.held_buttons.clear()

    def run(self):
        while not self._stop.is_set():
            self._post("status", ("searching", None))
            port = self.find_pi_port()

            if not port:
                self._post("status", ("disconnected", None))
                if self._stop.wait(RECONNECT_DELAY):
                    return
                continue

            try:
                ser = serial.Serial(port, BAUD, timeout=1)
            except serial.SerialException as e:
                self._post("log", f"[ERROR] Could not open {port}: {e}")
                self._post("status", ("disconnected", None))
                if self._stop.wait(RECONNECT_DELAY):
                    return
                continue

            self._post("status", ("connected", port))
            self._post("log", f"Connected to {port} at {BAUD} baud")

            try:
                while not self._stop.is_set():
                    raw = ser.readline()
                    if raw:
                        line = raw.decode("utf-8", errors="ignore").strip()
                        if line:
                            self._post("log", f"[CMD] {line}")
                            try:
                                self.handle_command(line)
                            except Exception as e:
                                self._post("log", f"[ERROR] {e}")
            except serial.SerialException as e:
                self._post("log", f"[ERROR] Lost connection: {e}")
            finally:
                self.release_all()
                ser.close()
                self._post("status", ("disconnected", None))

            if self._stop.wait(RECONNECT_DELAY):
                return


# ---------------------------------------------------------------------------
# Overlay — small always-on-top, borderless status badge in a screen corner.
# ---------------------------------------------------------------------------
class Overlay(tk.Toplevel):
    COLORS = {
        "connected": "#2ecc71",
        "searching": "#f1c40f",
        "disconnected": "#e74c3c",
    }
    LABELS = {
        "connected": "Connected",
        "searching": "Searching…",
        "disconnected": "Disconnected",
    }

    def __init__(self, master):
        super().__init__(master)
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.85)
        self.configure(bg="#1e1e1e")

        pad = 10
        self.dot = tk.Canvas(self, width=14, height=14, bg="#1e1e1e", highlightthickness=0)
        self.dot_id = self.dot.create_oval(2, 2, 12, 12, fill=self.COLORS["disconnected"], outline="")
        self.dot.grid(row=0, column=0, padx=(pad, 6), pady=pad)

        self.label = tk.Label(self, text="Disconnected", fg="white", bg="#1e1e1e",
                              font=("Segoe UI", 10))
        self.label.grid(row=0, column=1, padx=(0, pad), pady=pad)

        self._place_top_right()
        self.bind("<Button-1>", self._start_drag)
        self.bind("<B1-Motion>", self._drag)

    def _place_top_right(self):
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        w = self.winfo_reqwidth()
        self.geometry(f"+{sw - w - 20}+20")

    def _start_drag(self, event):
        self._drag_x, self._drag_y = event.x, event.y

    def _drag(self, event):
        x = self.winfo_x() + (event.x - self._drag_x)
        y = self.winfo_y() + (event.y - self._drag_y)
        self.geometry(f"+{x}+{y}")

    def set_status(self, state, detail=None):
        color = self.COLORS.get(state, "#888888")
        text = self.LABELS.get(state, state)
        if state == "connected" and detail:
            text = f"Connected — {detail}"
        self.dot.itemconfig(self.dot_id, fill=color)
        self.label.config(text=text)


# ---------------------------------------------------------------------------
# Main control window — status, log, and tray integration.
# ---------------------------------------------------------------------------
class ReceiverApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Pi Remote Receiver")
        self.root.geometry("520x360")
        self.root.minsize(420, 280)
        self.root.protocol("WM_DELETE_WINDOW", self.hide_to_tray)

        self.event_queue = queue.Queue()
        self.worker = SerialWorker(self.event_queue)
        self.tray_icon = None
        self.current_state = "disconnected"

        self._build_ui()
        self.overlay = Overlay(self.root)

        self.worker.start()
        self.root.after(100, self._drain_events)

        self._start_tray()

    # --- UI -----------------------------------------------------------
    def _build_ui(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass

        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")

        self.status_canvas = tk.Canvas(top, width=16, height=16, highlightthickness=0)
        self.status_dot = self.status_canvas.create_oval(2, 2, 14, 14, fill="#e74c3c", outline="")
        self.status_canvas.pack(side="left", padx=(0, 8))

        self.status_label = ttk.Label(top, text="Disconnected", font=("Segoe UI", 11, "bold"))
        self.status_label.pack(side="left")

        self.overlay_var = tk.BooleanVar(value=True)
        overlay_check = ttk.Checkbutton(top, text="Show overlay", variable=self.overlay_var,
                                        command=self._toggle_overlay)
        overlay_check.pack(side="right")

        self.snap_var = tk.BooleanVar(value=SNAP_ENABLED_DEFAULT)
        snap_check = ttk.Checkbutton(top, text="UI snapping", variable=self.snap_var,
                                     command=self._toggle_snap)
        snap_check.pack(side="right", padx=(0, 12))

        smooth_row = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        smooth_row.pack(fill="x")
        ttk.Label(smooth_row, text="Snap glide").pack(side="left")
        self.smoothing_var = tk.DoubleVar(value=MOVE_SMOOTHING_DEFAULT)
        smoothing_scale = ttk.Scale(smooth_row, from_=0.05, to=1.0, orient="horizontal",
                                    variable=self.smoothing_var, command=self._on_smoothing_change)
        smoothing_scale.pack(side="left", fill="x", expand=True, padx=8)
        self.smoothing_label = ttk.Label(smooth_row, text=f"{MOVE_SMOOTHING_DEFAULT:.2f}", width=5)
        self.smoothing_label.pack(side="left")
        ttk.Label(smooth_row, text="(lower = smoother / slower glide)",
                  foreground="#888888").pack(side="left", padx=(8, 0))

        ttk.Separator(self.root).pack(fill="x")

        log_frame = ttk.Frame(self.root, padding=12)
        log_frame.pack(fill="both", expand=True)
        ttk.Label(log_frame, text="Activity log").pack(anchor="w")

        text_frame = ttk.Frame(log_frame)
        text_frame.pack(fill="both", expand=True, pady=(4, 0))

        self.log_text = tk.Text(text_frame, height=10, state="disabled", wrap="word",
                                font=("Consolas", 9), bg="#fafafa")
        scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        bottom = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        bottom.pack(fill="x")
        ttk.Button(bottom, text="Minimize to tray", command=self.hide_to_tray).pack(side="right")
        ttk.Button(bottom, text="Quit", command=self.quit).pack(side="right", padx=(0, 8))

    def _toggle_overlay(self):
        if self.overlay_var.get():
            self.overlay.deiconify()
        else:
            self.overlay.withdraw()

    def _toggle_snap(self):
        self.worker.set_snap_enabled(self.snap_var.get())

    def _on_smoothing_change(self, value):
        v = float(value)
        self.smoothing_label.config(text=f"{v:.2f}")
        self.worker.set_move_smoothing(v)

    def _append_log(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # --- Event pump ----------------------------------------------------
    def _drain_events(self):
        try:
            while True:
                kind, payload = self.event_queue.get_nowait()
                if kind == "status":
                    state, detail = payload
                    self._apply_status(state, detail)
                elif kind == "log":
                    self._append_log(payload)
                elif kind == "snap":
                    self.snap_var.set(payload)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)

    def _apply_status(self, state, detail):
        self.current_state = state
        colors = {"connected": "#2ecc71", "searching": "#f1c40f", "disconnected": "#e74c3c"}
        labels = {"connected": "Connected" + (f" — {detail}" if detail else ""),
                  "searching": "Searching for device…",
                  "disconnected": "Disconnected"}
        color = colors.get(state, "#888888")
        self.status_canvas.itemconfig(self.status_dot, fill=color)
        self.status_label.config(text=labels.get(state, state))
        self.overlay.set_status(state, detail)
        if self.tray_icon:
            self.tray_icon.icon = self._make_tray_image(color)
            self.tray_icon.title = f"Pi Remote Receiver — {labels.get(state, state)}"

    # --- Tray -----------------------------------------------------------
    def _make_tray_image(self, color_hex):
        size = 64
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse((4, 4, size - 4, size - 4), fill=color_hex)
        return img

    def _start_tray(self):
        menu = pystray.Menu(
            pystray.MenuItem("Show window", self._tray_show, default=True),
            pystray.MenuItem("Toggle overlay", self._tray_toggle_overlay),
            pystray.MenuItem("Quit", self._tray_quit),
        )
        self.tray_icon = pystray.Icon("pi_receiver", self._make_tray_image("#e74c3c"),
                                      "Pi Remote Receiver", menu)
        threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def _tray_show(self, icon=None, item=None):
        self.root.after(0, self._show_window)

    def _tray_toggle_overlay(self, icon=None, item=None):
        def toggle():
            self.overlay_var.set(not self.overlay_var.get())
            self._toggle_overlay()
        self.root.after(0, toggle)

    def _tray_quit(self, icon=None, item=None):
        self.root.after(0, self.quit)

    def _show_window(self):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def hide_to_tray(self):
        self.root.withdraw()

    def quit(self):
        self.worker.stop()
        if self.tray_icon:
            self.tray_icon.stop()
        self.overlay.destroy()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = ReceiverApp()
    app.run()
