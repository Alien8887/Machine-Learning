"""
Real-time face detection that works both as a normal script (cv2.imshow)
and inside Jupyter (inline fallback).

Requirements:
    pip install opencv-python numpy
(If you're in a server environment, avoid opencv-python-headless unless you only
want the Jupyter fallback.)

Press 'q' in the regular window to quit. In Jupyter, stop the cell (KeyboardInterrupt).
"""
import time
import sys

import cv2
import numpy as np

# Try a quick check whether cv2.imshow works (some wheels are "headless")
def has_gui_support():
    try:
        win_name = "__cv_test__"
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, img)
        cv2.waitKey(1)
        cv2.destroyWindow(win_name)
        return True
    except Exception:
        return False

HEADLESS = not has_gui_support()

if HEADLESS:
    # Jupyter / headless fallback imports (import only if needed)
    try:
        from IPython.display import display, Image, clear_output
    except Exception:
        # If IPython isn't available, we'll still run but only log frames to disk
        display = None

# Load Haar cascade (bundled with OpenCV)
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise RuntimeError("Failed to load Haar cascade. Check OpenCV installation.")

# Open camera (change index if you have multiple cameras)
cam_index = 0
cap = cv2.VideoCapture(cam_index)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera index {cam_index}. Check drivers / index.")

# Optionally set resolution for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("HEADLESS mode:" , HEADLESS)
print("Press 'q' in the window to quit (or stop the kernel / Ctrl+C in Jupyter).")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if not HEADLESS:
            # Normal GUI display
            cv2.imshow("Face Detection", frame)
            # Wait 1 ms for keypress; quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # Jupyter / inline fallback: encode and display inline
            # If IPython.display is missing, save a single frame to disk instead.
            if display is None:
                # fallback: write frames to disk (every second) to inspect
                t = int(time.time())
                fname = f"frame_{t}.jpg"
                cv2.imwrite(fname, frame)
                print(f"Saved {fname}")
                time.sleep(1.0)
            else:
                _, img_bytes = cv2.imencode(".jpg", frame)
                clear_output(wait=True)
                display(Image(data=img_bytes.tobytes()))
                # small sleep to avoid maxing CPU
                time.sleep(0.03)

except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    cap.release()
    if not HEADLESS:
        cv2.destroyAllWindows()
    print("Camera released, exiting.")
