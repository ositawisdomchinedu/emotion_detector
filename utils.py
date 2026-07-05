"""
Emotion Detector Pro - Core Engine (v2)
NOTE: This extends the starter (utils.py / detect.py) with a
production-style pipeline.

Includes:
- OpenCV DNN face detection
- Lightweight centroid-based multi-face tracker (persistent IDs)
- Exponential moving average (EMA) smoothing -> no label/box flicker
- Emotion prediction with per-face confidence bars (all 7 classes)
- Professional HUD: semi-transparent side panel, FPS, face count,
  per-face detection confidence, date/time
- Corner-accent bounding boxes with per-emotion color coding

Usage:
    detector = EmotionDetector(model)
    renderer = UIRenderer()
    tracks, fps = detector.process_frame(frame)
    frame = renderer.render(frame, tracks, fps)
"""

import time
from datetime import datetime

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

COLORS = {
    "Angry":    (60,  60,  220),
    "Disgust":  (60,  140, 60),
    "Fear":     (200, 60,  200),
    "Happy":    (60,  200, 80),
    "Sad":      (200, 120, 60),
    "Surprise": (60,  200, 220),
    "Neutral":  (210, 210, 210),
}

UI_BG = (28, 28, 30)          # side panel background
UI_ACCENT = (0, 200, 255)     # headings / highlights
UI_TEXT = (235, 235, 235)     # primary text
UI_SUBTEXT = (160, 160, 165)  # secondary text
UI_DIVIDER = (80, 80, 85)


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #

class FaceTrack:
    """A single tracked face, persistent across frames."""

    __slots__ = ("id", "bbox", "centroid", "probs_ema", "det_conf", "misses", "alpha")

    def __init__(self, track_id, bbox, probs, det_conf, alpha=0.35):
        self.id = track_id
        self.bbox = bbox
        self.centroid = self._centroid(bbox)
        self.probs_ema = probs.copy()
        self.det_conf = det_conf
        self.misses = 0
        self.alpha = alpha  # smoothing factor: lower = smoother/slower

    @staticmethod
    def _centroid(bbox):
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

    def update(self, bbox, probs, det_conf):
        self.bbox = bbox
        self.centroid = self._centroid(bbox)
        # EMA smoothing -> prevents emotion label/box jitter frame to frame
        self.probs_ema = self.alpha * probs + (1 - self.alpha) * self.probs_ema
        self.det_conf = det_conf
        self.misses = 0

    @property
    def emotion(self):
        idx = int(np.argmax(self.probs_ema))
        return EMOTIONS[idx], float(self.probs_ema[idx]) * 100.0

    @property
    def area(self):
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


class CentroidTracker:
    """Minimal dependency-free tracker: greedy nearest-centroid matching."""

    def __init__(self, max_misses=8, max_distance=80):
        self.tracks = {}
        self.next_id = 1
        self.max_misses = max_misses
        self.max_distance = max_distance

    def update(self, detections):
        """detections: list of (bbox, probs, det_conf). Returns active FaceTracks."""
        if not detections:
            for t in self.tracks.values():
                t.misses += 1
            self._prune()
            return list(self.tracks.values())

        det_centroids = [FaceTrack._centroid(d[0]) for d in detections]

        pairs = []
        for tid, track in self.tracks.items():
            for di, c in enumerate(det_centroids):
                dist = float(np.linalg.norm(track.centroid - c))
                pairs.append((dist, tid, di))
        pairs.sort(key=lambda p: p[0])

        matched_tracks, matched_dets = set(), set()
        for dist, tid, di in pairs:
            if tid in matched_tracks or di in matched_dets or dist > self.max_distance:
                continue
            bbox, probs, det_conf = detections[di]
            self.tracks[tid].update(bbox, probs, det_conf)
            matched_tracks.add(tid)
            matched_dets.add(di)

        # unmatched detections -> spawn new tracks
        for di in range(len(detections)):
            if di in matched_dets:
                continue
            bbox, probs, det_conf = detections[di]
            self.tracks[self.next_id] = FaceTrack(self.next_id, bbox, probs, det_conf)
            self.next_id += 1

        # unmatched tracks -> age out
        for tid in self.tracks:
            if tid not in matched_tracks:
                self.tracks[tid].misses += 1

        self._prune()
        return list(self.tracks.values())

    def _prune(self):
        dead = [tid for tid, t in self.tracks.items() if t.misses > self.max_misses]
        for tid in dead:
            del self.tracks[tid]


# --------------------------------------------------------------------------- #
# FPS
# --------------------------------------------------------------------------- #

class FPSMeter:
    """Exponentially smoothed FPS so the on-screen number doesn't jump around."""

    def __init__(self, smoothing=0.9):
        self.smoothing = smoothing
        self.fps = 0.0
        self._last = time.time()

    def tick(self):
        now = time.time()
        dt = now - self._last
        self._last = now
        if dt > 0:
            inst = 1.0 / dt
            self.fps = inst if self.fps == 0 else (self.fps * self.smoothing + inst * (1 - self.smoothing))
        return self.fps


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #

class EmotionDetector:
    def __init__(self, model, prototxt="deploy.prototxt",
                 caffemodel="res10_300x300_ssd_iter_140000.caffemodel",
                 conf_threshold=0.5):
        self.model = model
        self.conf_threshold = conf_threshold
        self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        self.tracker = CentroidTracker()
        self.fps_meter = FPSMeter()

    def _detect_faces(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                      (300, 300), (104, 177, 123))
        self.net.setInput(blob)
        detections = self.net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(((x1, y1, x2, y2), conf))
        return boxes

    
    def _predict_emotion(self, frame, bbox):
        x1, y1, x2, y2 = bbox

        h, w = frame.shape[:2]

        pad = 10

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
         return None

        # Resize to training size
        face = cv2.resize(face, (96, 96))

        # Convert OpenCV BGR -> RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Float32
        face = face.astype(np.float32)

        # SAME preprocessing used during training
        face = preprocess_input(face)

        # Add batch dimension
        face = np.expand_dims(face, axis=0)

        pred = self.model.predict(face, verbose=0)[0]

        return pred.astype(np.float32)

    def process_frame(self, frame):
        """Runs detection, emotion inference, tracking and smoothing.

        Returns (tracks, fps) where tracks is a list of FaceTrack objects
        with stable IDs and smoothed emotion probabilities.
        """
        boxes = self._detect_faces(frame)

        detections = []
        for bbox, conf in boxes:
            probs = self._predict_emotion(frame, bbox)
            if probs is None:
                continue
            detections.append((bbox, probs, conf))

        tracks = self.tracker.update(detections)
        fps = self.fps_meter.tick()
        return tracks, fps


# --------------------------------------------------------------------------- #
# Rendering / UI
# --------------------------------------------------------------------------- #

class UIRenderer:
    """Draws the professional HUD: bounding boxes + semi-transparent side panel."""

    def __init__(self, panel_width=320, panel_alpha=0.55, max_faces_shown=2):
        self.panel_width = panel_width
        self.panel_alpha = panel_alpha
        self.max_faces_shown = max_faces_shown

    # -- bounding boxes -----------------------------------------------------

    def draw_boxes(self, frame, tracks):
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            emotion, score = t.emotion
            color = COLORS[emotion]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            self._draw_corners(frame, (x1, y1, x2, y2), color)

            label = f"ID {t.id} | {emotion} {score:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ly = max(th + 12, y1 - 8)
            cv2.rectangle(frame, (x1, ly - th - 8), (x1 + tw + 10, ly + 2), color, -1)
            cv2.putText(frame, label, (x1 + 5, ly - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        return frame

    @staticmethod
    def _draw_corners(frame, bbox, color, length=14, thickness=3):
        x1, y1, x2, y2 = bbox
        segments = [
            ((x1, y1), (x1 + length, y1), (x1, y1 + length)),
            ((x2, y1), (x2 - length, y1), (x2, y1 + length)),
            ((x1, y2), (x1 + length, y2), (x1, y2 - length)),
            ((x2, y2), (x2 - length, y2), (x2, y2 - length)),
        ]
        for corner, h_end, v_end in segments:
            cv2.line(frame, corner, h_end, color, thickness)
            cv2.line(frame, corner, v_end, color, thickness)

    # -- side panel -----------------------------------------------------------

    def draw_panel(self, frame, tracks, fps):
        h, w = frame.shape[:2]
        px = w - self.panel_width

        # semi-transparent panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, 0), (w, h), UI_BG, -1)
        frame[:] = cv2.addWeighted(overlay, self.panel_alpha, frame, 1 - self.panel_alpha, 0)

        pad = 18
        x = px + pad
        y = 32

        cv2.putText(frame, "EMOTION DETECTOR PRO", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, UI_ACCENT, 2, cv2.LINE_AA)
        y += 12
        cv2.line(frame, (x, y), (w - pad, y), UI_DIVIDER, 1)
        y += 24

        now = datetime.now()
        cv2.putText(frame, now.strftime("%a, %d %b %Y"), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, UI_TEXT, 1, cv2.LINE_AA)
        y += 20
        cv2.putText(frame, now.strftime("%H:%M:%S"), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, UI_SUBTEXT, 1, cv2.LINE_AA)
        y += 28

        cv2.putText(frame, f"FPS: {fps:.1f}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, UI_TEXT, 1, cv2.LINE_AA)
        y += 22
        cv2.putText(frame, f"Faces Detected: {len(tracks)}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, UI_TEXT, 1, cv2.LINE_AA)
        y += 18
        cv2.line(frame, (x, y), (w - pad, y), UI_DIVIDER, 1)
        y += 24

        if not tracks:
            cv2.putText(frame, "No face detected", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, UI_SUBTEXT, 1, cv2.LINE_AA)
            return frame

        # largest / closest faces first
        ranked = sorted(tracks, key=lambda t: t.area, reverse=True)
        bar_w = self.panel_width - 2 * pad - 70
        block_height = 20 + len(EMOTIONS) * 18 + 14

        shown = 0
        for t in ranked[:self.max_faces_shown]:
            if y + block_height > h - pad:
                break

            cv2.putText(frame, f"Face #{t.id}  (det {t.det_conf * 100:.0f}%)", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, UI_ACCENT, 1, cv2.LINE_AA)
            y += 20

            for e_idx, emotion in enumerate(EMOTIONS):
                p = float(t.probs_ema[e_idx])
                color = COLORS[emotion]

                cv2.putText(frame, emotion[:4], (x, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, UI_SUBTEXT, 1, cv2.LINE_AA)

                bar_x = x + 48
                cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + 12), (60, 60, 65), -1)
                fill_w = int(bar_w * p)
                if fill_w > 0:
                    cv2.rectangle(frame, (bar_x, y), (bar_x + fill_w, y + 12), color, -1)
                cv2.putText(frame, f"{p * 100:.0f}%", (bar_x + bar_w + 6, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI_SUBTEXT, 1, cv2.LINE_AA)
                y += 18

            y += 14
            shown += 1

        remaining = len(ranked) - shown
        if remaining > 0:
            cv2.putText(frame, f"+{remaining} more face(s)", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, UI_SUBTEXT, 1, cv2.LINE_AA)

        return frame

    # -- combined -------------------------------------------------------------

    def render(self, frame, tracks, fps):
        self.draw_boxes(frame, tracks)
        self.draw_panel(frame, tracks, fps)
        return frame