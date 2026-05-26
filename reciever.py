# receiver.py — Run this on Windows
# Install: pip install pyserial pyautogui mouse uiautomation

import serial
import serial.tools.list_ports
import pyautogui
import sys
import mouse
import time
import uiautomation as auto

# --- Config ---
BAUD = 115200
PI_VENDOR_IDS = [0x2E8A, 0x0403, 0x10C4, 0x1A86, 0x067B]

pyautogui.FAILSAFE = True

# --- UI Snap Config ---
SNAP_ENABLED    = True
SNAP_THRESHOLD  = 45        # px — how close cursor must be to a control's edge to snap
SNAP_INTERVAL   = 0.08      # seconds between UI-tree queries (keeps move latency low)

# Control types that are worth snapping to
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

_last_snap_time: float = 0.0
_snap_cache: tuple[int, int] | None = None   # last resolved snap target

# Title bar buttons are tiny — use a wider radius when near the top of the screen
TITLEBAR_ZONE_HEIGHT = 60   # px from top of screen considered "title bar zone"
TITLEBAR_SNAP_THRESHOLD = 80  # wider snap radius inside that zone


def _rect_center(rect) -> tuple[int, int]:
    return (rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2


def _distance(ax, ay, bx, by) -> float:
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _nearest_snap_in_children(parent, x: int, y: int, threshold: float):
    """Walk immediate children of parent to find the closest snappable control."""
    best_dist = threshold
    best_pos  = None
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
                        best_pos  = (cx, cy)
            child = child.GetNextSiblingControl()
    except Exception:
        pass
    return best_pos


def get_snap_target(x: int, y: int) -> tuple[int, int] | None:
    """
    Return the centre of the nearest interactive UI control if the cursor is
    within SNAP_THRESHOLD pixels of it, otherwise None.

    For title bar zones (top of screen) a wider threshold is used, and if
    ControlFromPoint returns a non-interactive parent (e.g. the browser chrome
    pane), its children are searched so that minimize/maximize/close buttons
    are still reachable.

    Queries are throttled to SNAP_INTERVAL seconds to avoid stalling the move loop.
    """
    global _last_snap_time, _snap_cache

    now = time.monotonic()
    if now - _last_snap_time < SNAP_INTERVAL:
        return _snap_cache

    _last_snap_time = now

    # Use a wider threshold when in the title bar zone
    threshold = TITLEBAR_SNAP_THRESHOLD if y <= TITLEBAR_ZONE_HEIGHT else SNAP_THRESHOLD

    try:
        control = auto.ControlFromPoint(x, y)
        if control:
            if control.ControlType in SNAP_TYPES:
                # Direct hit — check distance to its centre
                rect = control.BoundingRectangle
                if rect.right > rect.left and rect.bottom > rect.top:
                    cx, cy = _rect_center(rect)
                    if _distance(x, y, cx, cy) <= threshold:
                        _snap_cache = (cx, cy)
                        return _snap_cache
            else:
                # Not directly snappable — search children (catches title bar buttons
                # hidden inside a browser chrome pane or TitleBarControl)
                child_snap = _nearest_snap_in_children(control, x, y, threshold)
                if child_snap:
                    _snap_cache = child_snap
                    return _snap_cache
    except Exception:
        pass

    _snap_cache = None
    return None


def find_pi_port():
    ports = serial.tools.list_ports.comports()

    for port in ports:
        if port.vid in PI_VENDOR_IDS:
            print(f"[AUTO] Found device by VID {hex(port.vid)}: {port.device} — {port.description}")
            return port.device

    keywords = ["raspberry", "pi", "usb serial", "uart", "ch340", "cp210", "ftdi", "cdc"]
    for port in ports:
        desc = (port.description or "").lower()
        if any(kw in desc for kw in keywords):
            print(f"[AUTO] Found device by description: {port.device} — {port.description}")
            return port.device

    if ports:
        print("\n[WARN] Could not auto-detect Pi. Available ports:")
        for i, p in enumerate(ports):
            print(f"  [{i}] {p.device} — {p.description} (VID: {hex(p.vid) if p.vid else 'N/A'})")
        choice = input("Enter port number to use: ").strip()
        return ports[int(choice)].device

    print("[ERROR] No serial ports found. Is the Pi connected via USB?")
    sys.exit(1)


# Track which buttons are currently held so we can release cleanly
held_buttons = set()

def handle_command(line: str):
    global SNAP_ENABLED
    parts = line.strip().split()
    if not parts:
        return

    cmd = parts[0].upper()

    if cmd == "MOVE" and len(parts) == 3:
        x, y = int(parts[1]), int(parts[2])
        if SNAP_ENABLED:
            snap = get_snap_target(x, y)
            if snap:
                sx, sy = snap
                print(f"[SNAP] {x},{y} → {sx},{sy}")
                x, y = sx, sy
        mouse.move(x, y)

    elif cmd == "SINGLE_CLICK" and len(parts) == 2:
        btn = parts[1].lower()
        mouse.click(button=btn)

    elif cmd == "CLICK" and len(parts) == 2:
        btn = parts[1].lower()
        mouse.press(button=btn)
        held_buttons.add(btn)

    elif cmd == "RELEASE" and len(parts) == 2:
        btn = parts[1].lower()
        if btn in held_buttons:
            mouse.release(button=btn)
            held_buttons.discard(btn)

    elif cmd == "DCLICK" and len(parts) == 2:
        btn = parts[1].lower()
        pyautogui.doubleClick(button=btn)

    elif cmd == "SCROLL" and len(parts) == 3:
        direction = parts[1].upper()
        amount    = int(parts[2])
        if direction == "UP":
            pyautogui.scroll(amount)
        elif direction == "DOWN":
            pyautogui.scroll(-amount)

    elif cmd == "KEY" and len(parts) >= 2:
        keys = parts[1].lower()
        pyautogui.hotkey(*keys.split("+"))

    elif cmd == "TYPE" and len(parts) >= 2:
        text = " ".join(parts[1:])
        pyautogui.typewrite(text, interval=0.03)

    elif cmd == "PRESS" and len(parts) == 2:
        pyautogui.press(parts[1].lower())

    # Toggle snapping on the fly with "SNAP ON" / "SNAP OFF"
    elif cmd == "SNAP" and len(parts) == 2:
        SNAP_ENABLED = parts[1].upper() == "ON"
        print(f"[SNAP] {'Enabled' if SNAP_ENABLED else 'Disabled'}")

    else:
        print(f"[WARN] Unknown command: {line.strip()}")


def release_all():
    """Safety net — release any held buttons on exit."""
    for btn in list(held_buttons):
        try:
            mouse.release(button=btn)
            print(f"[CLEANUP] Released {btn}")
        except Exception:
            pass
    held_buttons.clear()


def main():
    port = find_pi_port()
    print(f"Connecting to {port} at {BAUD} baud...")

    try:
        ser = serial.Serial(port, BAUD, timeout=1)
    except serial.SerialException as e:
        print(f"[ERROR] Could not open port: {e}")
        sys.exit(1)

    snap_status = "enabled" if SNAP_ENABLED else "disabled"
    print(f"Listening for commands. UI snapping {snap_status} (threshold: {SNAP_THRESHOLD}px).")
    print("Move mouse to top-left corner to abort.\n")
    try:
        while True:
            raw = ser.readline()
            if raw:
                line = raw.decode("utf-8", errors="ignore").strip()
                if line:
                    print(f"[CMD] {line}")
                    try:
                        handle_command(line)
                    except Exception as e:
                        print(f"[ERROR] {e}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        release_all()
        ser.close()


if __name__ == "__main__":
    main()
