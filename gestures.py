"""
Hand tracking + gesture recognition using the new MediaPipe Tasks API
(mediapipe >= 0.10).

Uses HandLandmarker in VIDEO mode for per-frame landmark detection.
Gesture is determined from finger extension states and pinch distance.
"""

import os
import numpy as np
import mediapipe as mp
from collections import deque
from mediapipe.tasks.python        import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# Path to the bundled model — same directory as this file
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# Finger tip / PIP landmark indices (MediaPipe hand topology)
_TIPS = [4, 8, 12, 16, 20]
_PIPS = [3, 6, 10, 14, 18]


class HandTracker:
    def __init__(self):
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found: {_MODEL_PATH}\n"
                "Download it with:\n"
                "  python -c \"import urllib.request; "
                "urllib.request.urlretrieve("
                "'https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'"
                ", 'hand_landmarker.task')\""
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.65,
            min_tracking_confidence=0.55,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

        # Draw utilities (from the new API)
        self._draw_utils  = mp.tasks.vision.drawing_utils
        self._draw_styles = mp.tasks.vision.drawing_styles

        # Rolling window of index-finger tip positions for velocity
        self._history: deque = deque(maxlen=6)
        self._ts_ms: int = 0   # monotonic timestamp fed to VIDEO mode

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process(self, bgr_frame):
        """
        Run hand landmark detection on a BGR OpenCV frame.
        Returns a HandLandmarkerResult (or None-equivalent if no hands).
        """
        import cv2
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms += 33          # ~30 fps synthetic timestamp
        return self._landmarker.detect_for_video(mp_image, self._ts_ms)

    def landmarks_px(self, hand_landmarks, W: int, H: int):
        """
        Convert normalised landmark list to pixel (x, y) tuples.
        hand_landmarks: one element from result.hand_landmarks
        """
        return [(int(lm.x * W), int(lm.y * H)) for lm in hand_landmarks]

    # ------------------------------------------------------------------
    # Finger state helpers
    # ------------------------------------------------------------------

    def _finger_states(self, lm):
        """
        lm: list of (x_px, y_px) for 21 landmarks.
        Returns [thumb, index, middle, ring, pinky] — True = extended.
        Assumes mirrored (flipped) webcam feed.
        """
        # Thumb: in a mirrored feed the tip moves left when extended
        thumb = lm[4][0] < lm[3][0]
        # Other fingers: tip.y < pip.y → pointing upward
        rest = [lm[t][1] < lm[p][1] for t, p in zip(_TIPS[1:], _PIPS[1:])]
        return [thumb] + rest

    # ------------------------------------------------------------------
    # Main gesture API
    # ------------------------------------------------------------------

    def get_gesture(self, lm, frame_shape):
        """
        lm          : list of (x_px, y_px) from landmarks_px()
        frame_shape : (H, W, C) shape of the current frame

        Returns:
          gesture   : str
          index_tip : np.ndarray [x, y] pixels
          thumb_tip : np.ndarray [x, y] pixels
          velocity  : float  (pixels moved over last ~3 frames)
        """
        thumb, index, middle, ring, pinky = self._finger_states(lm)

        index_tip = np.array(lm[8], dtype=float)
        thumb_tip = np.array(lm[4], dtype=float)

        # Velocity from rolling history
        self._history.append(index_tip.copy())
        velocity = 0.0
        if len(self._history) >= 3:
            velocity = float(np.linalg.norm(
                self._history[-1] - self._history[-3]
            ))

        # Pinch distance
        pinch_dist = float(np.linalg.norm(index_tip - thumb_tip))
        pinching   = pinch_dist < 48

        # Gesture rules (most specific first)
        all_up     = index and middle and ring and pinky
        only_index = index and not middle and not ring and not pinky
        two_up     = index and middle and not ring and not pinky
        fist       = not index and not middle and not ring and not pinky

        if all_up:
            gesture = "SPLATTER"
        elif only_index and pinching:
            gesture = "COLOR_PICK"
        elif only_index:
            gesture = "DRAW"
        elif two_up:
            gesture = "ERASE"
        elif fist:
            gesture = "FIST"
        else:
            gesture = "IDLE"

        return gesture, index_tip, thumb_tip, velocity

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_skeleton(self, frame, hand_landmarks, W: int, H: int):
        """
        Draw a subtle hand skeleton overlay directly on the BGR frame.
        Uses OpenCV lines to avoid dependency on the old mp.solutions.drawing_utils.
        """
        import cv2
        # Connections from MediaPipe hand topology (21 keypoints)
        CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17),
        ]
        lm = self.landmarks_px(hand_landmarks, W, H)
        for a, b in CONNECTIONS:
            cv2.line(frame, lm[a], lm[b], (60, 60, 60), 1, cv2.LINE_AA)
        for pt in lm:
            cv2.circle(frame, pt, 2, (80, 80, 80), -1, cv2.LINE_AA)
