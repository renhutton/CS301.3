# receiver.py — Run this on Windows
# Install: pip install pyserial pyautogui

import serial
import serial.tools.list_ports
import pyautogui
import sys
import time
import mouse

# --- Config ---
BAUD = 115200
# Known Raspberry Pi USB vendor IDs
PI_VENDOR_IDS = [0x2E8A, 0x0403, 0x10C4, 0x1A86, 0x067B]

pyautogui.FAILSAFE = True

def find_pi_port():
    """Auto-detect Raspberry Pi serial port by vendor ID or description."""
    ports = serial.tools.list_ports.comports()
    
    # First pass: match by known Pi/USB-serial vendor IDs
    for port in ports:
        if port.vid in PI_VENDOR_IDS:
            print(f"[AUTO] Found device by VID {hex(port.vid)}: {port.device} — {port.description}")
            return port.device

    # Second pass: match by common description keywords
    keywords = ["raspberry", "pi", "usb serial", "uart", "ch340", "cp210", "ftdi", "cdc"]
    for port in ports:
        desc = (port.description or "").lower()
        if any(kw in desc for kw in keywords):
            print(f"[AUTO] Found device by description: {port.device} — {port.description}")
            return port.device

    # Fallback: list all ports and let user pick
    if ports:
        print("\n[WARN] Could not auto-detect Pi. Available ports:")
        for i, p in enumerate(ports):
            print(f"  [{i}] {p.device} — {p.description} (VID: {hex(p.vid) if p.vid else 'N/A'})")
        choice = input("Enter port number to use: ").strip()
        return ports[int(choice)].device

    print("[ERROR] No serial ports found. Is the Pi connected via USB?")
    sys.exit(1)

def handle_command(line: str):
    parts = line.strip().split()
    if not parts:
        return

    cmd = parts[0].upper()

    if cmd == "MOVE" and len(parts) == 3:
        x, y = int(parts[1]), int(parts[2])
        mouse.move(x, y)

    elif cmd == "CLICK" and len(parts) == 2:
        btn = parts[1].lower()
        pyautogui.click(button=btn)

    elif cmd == "DCLICK" and len(parts) == 2:
        btn = parts[1].lower()
        pyautogui.doubleClick(button=btn)

    elif cmd == "SCROLL" and len(parts) == 2:
        amount = int(parts[1])
        pyautogui.scroll(amount)

    elif cmd == "KEY" and len(parts) >= 2:
        keys = parts[1].lower()
        pyautogui.hotkey(*keys.split("+"))

    elif cmd == "TYPE" and len(parts) >= 2:
        text = " ".join(parts[1:])
        pyautogui.typewrite(text, interval=0.03)

    elif cmd == "PRESS" and len(parts) == 2:
        pyautogui.press(parts[1].lower())

    else:
        print(f"[WARN] Unknown command: {line.strip()}")

def main():
    port = find_pi_port()
    print(f"Connecting to {port} at {BAUD} baud...")

    try:
        ser = serial.Serial(port, BAUD, timeout=1)
    except serial.SerialException as e:
        print(f"[ERROR] Could not open port: {e}")
        sys.exit(1)

    print("Listening for commands. Move mouse to top-left corner to abort.\n")
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
        ser.close()

if __name__ == "__main__":
    main()