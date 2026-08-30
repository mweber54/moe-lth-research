import time
import pyautogui

INTERVAL = 240  # seconds; 4 minutes

print("Keeping PC awake. Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(INTERVAL)
        pyautogui.press("shift")
        print("Pressed Shift")
except KeyboardInterrupt:
    print("Stopped.")