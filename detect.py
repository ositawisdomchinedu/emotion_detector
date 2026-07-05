"""
Emotion Detector Pro - Main Application (v2)

Run: python detect.py

Controls:
    q / ESC  -> quit
"""

import cv2
from tensorflow.keras.models import load_model

from utils import EmotionDetector, UIRenderer

MODEL_PATH = "models/final_emotion_model.keras"
PROTOTXT = "models/deploy.prototxt"
CAFFEMODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"

WINDOW_NAME = "Emotion Detector Pro"
PANEL_WIDTH = 320
PANEL_ALPHA = 0.55
MAX_FACES_IN_PANEL = 2


def main():
    print("Loading emotion model...")
    model = load_model(MODEL_PATH, compile=False)

    detector = EmotionDetector(model, PROTOTXT, CAFFEMODEL)
    renderer = UIRenderer(panel_width=PANEL_WIDTH,
                           panel_alpha=PANEL_ALPHA,
                           max_faces_shown=MAX_FACES_IN_PANEL)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # mirror view, feels more natural on webcam

        tracks, fps = detector.process_frame(frame)
        frame = renderer.render(frame, tracks, fps)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()