"""
face_age_expression_ui.py

Requirements:
    pip install opencv-python numpy

Model files (optional but recommended):
  - Age (Caffe):
      models/age_deploy.prototxt
      models/age_net.caffemodel
    (predicts one of: ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'])
  - Emotion (ONNX, e.g., FER+):
      models/emotion-ferplus-8.onnx
    (predicts emotions--mapping below can be adapted to the exact model you download)

If models are missing, the script falls back to smile/eye-based heuristics for expression and to "N/A" for age.
"""

import os
import time
import cv2
import numpy as np

# ------------------------
# Config / model filenames
# ------------------------
MODEL_DIR = "models"
AGE_PROTO = os.path.join(MODEL_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODEL_DIR, "age_net.caffemodel")
EMOTION_ONNX = os.path.join(MODEL_DIR, "emotion-ferplus-8.onnx")

# Haar cascades (bundled with OpenCV)
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
SMILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Age buckets typically used with the Caffe age model
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Emotion labels for a typical FER+ model (adjust if your model uses a different ordering)
EMOTION_LABELS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

# ------------------------
# Utility: nice rounded rectangle
# ------------------------
def draw_label(img, text, x, y, bg_color=(0, 128, 255), text_color=(255,255,255), padding=6, alpha=0.6):
    """Draw semi-transparent rounded rectangle + text for a nice UI label."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x1, y1 = x, max(0, y - h - 2*padding)
    x2, y2 = x + w + 2*padding, y
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x1 + padding, y2 - padding - 2), font, scale, text_color, thickness, cv2.LINE_AA)

# ------------------------
# Load optional models if available
# ------------------------
age_net = None
emotion_net = None
have_age_model = os.path.exists(AGE_PROTO) and os.path.exists(AGE_MODEL)
have_emotion_model = os.path.exists(EMOTION_ONNX)

if have_age_model:
    try:
        age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
        print("[INFO] Age model loaded.")
    except Exception as e:
        print("[WARN] Failed to load age model:", e)
        age_net = None
        have_age_model = False

if have_emotion_model:
    try:
        emotion_net = cv2.dnn.readNetFromONNX(EMOTION_ONNX)
        print("[INFO] Emotion ONNX model loaded.")
    except Exception as e:
        print("[WARN] Failed to load emotion model:", e)
        emotion_net = None
        have_emotion_model = False

# ------------------------
# Predict helpers
# ------------------------
def predict_age(face_img):
    """Predict age bucket using the Caffe model. Expects BGR face (as from OpenCV)."""
    if age_net is None:
        return "N/A", 0.0
    # Age model expects 227x227 and mean subtraction (as in many example repos)
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                 (78.42633776, 87.76891437, 114.89584775), swapRB=False, crop=False)
    age_net.setInput(blob)
    preds = age_net.forward()
    i = preds[0].argmax()
    conf = float(preds[0][i])
    return AGE_BUCKETS[i], conf

def predict_emotion(face_img):
    """Predict emotion using an ONNX FER+ model (if available). Expects BGR face."""
    if emotion_net is None:
        return None
    # Many FER+ ONNX models expect grayscale 64x64, normalized to [0,1]
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    inp = resized.astype("float32") / 255.0
    inp = inp.reshape(1, 1, 64, 64)  # NCHW
    emotion_net.setInput(inp)
    preds = emotion_net.forward()
    # Some models return logits; apply softmax
    probs = np.exp(preds - np.max(preds)) / np.sum(np.exp(preds - np.max(preds)), axis=1, keepdims=True)
    probs = probs.flatten()
    idx = int(np.argmax(probs))
    return EMOTION_LABELS[idx], float(probs[idx])

def heuristic_expression(face_gray, face_color):
    """
    Light heuristic using Haar cascades:
      - Detect smile -> 'happy'
      - No smile but eyes closed -> 'sleepy/neutral'
      - else 'neutral'
    """
    smiles = SMILE_CASCADE.detectMultiScale(face_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25,25))
    eyes = EYE_CASCADE.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=6, minSize=(10,10))
    if len(smiles) > 0:
        return "happy", 0.9
    if len(eyes) == 0:
        return "eyes-closed", 0.6
    return "neutral", 0.6

# ------------------------
# Main loop & UI
# ------------------------
def main(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}. Try another index or check drivers.")

    # prefer moderate resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # detection tunables (exposed via trackbars)
    win_name = "Face + Age + Expression (press 'q' to quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Create trackbars
    def nothing(x): pass
    cv2.createTrackbar("scale x100", win_name, 110, 200, nothing)       # scaleFactor *100 (1.10 default)
    cv2.createTrackbar("minNeighbors", win_name, 5, 50, nothing)       # minNeighbors
    cv2.createTrackbar("minSize", win_name, 30, 300, nothing)          # min face size px

    prev_time = time.time()
    fps_smooth = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # read trackbars
            scale = cv2.getTrackbarPos("scale x100", win_name) / 100.0
            minN = cv2.getTrackbarPos("minNeighbors", win_name)
            minS = max(10, cv2.getTrackbarPos("minSize", win_name))

            # detect faces
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=scale, minNeighbors=max(1,minN), minSize=(minS, minS))

            # draw translucent info panel on top-left
            overlay = frame.copy()
            panel_h = 80
            cv2.rectangle(overlay, (0,0), (320, panel_h), (20,20,20), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            # show FPS
            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps = 1.0 / dt if dt>0 else 0.0
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps
            cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)
            model_status = f"Age:{'ON' if have_age_model else 'OFF'} EMO:{'ON' if have_emotion_model else 'OFF'}"
            cv2.putText(frame, model_status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

            for (x, y, w, h) in faces:
                # expand box slightly
                pad_x = int(0.05 * w)
                pad_y = int(0.08 * h)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(frame.shape[1], x + w + pad_x)
                y2 = min(frame.shape[0], y + h + pad_y)

                face_color = frame[y1:y2, x1:x2]
                face_gray = gray[y1:y2, x1:x2]

                # Predict age (if model available)
                age_label, age_conf = predict_age(face_color) if have_age_model else ("N/A", 0.0)

                # Predict emotion (prefer DNN model if available, otherwise heuristic)
                if have_emotion_model:
                    emo_label, emo_conf = predict_emotion(face_color)
                else:
                    emo_label, emo_conf = heuristic_expression(face_gray, face_color)

                # draw rounded rectangle (simple)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

                # labels
                label_text = f"{emo_label} {emo_conf:.2f}" if emo_label is not None else "N/A"
                age_text = f"Age:{age_label} {age_conf:.2f}" if age_label != "N/A" else "Age:N/A"

                draw_label(frame, label_text, x1, y1 - 6, bg_color=(90, 140, 255))
                draw_label(frame, age_text, x1, y2 + 24, bg_color=(0,160,120))

            # show instructions footer
            cv2.putText(frame, "Press 'q' to quit. Adjust scale/minNeighbors/minSize with sliders.", (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230,230,230), 1, cv2.LINE_AA)

            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
