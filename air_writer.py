"""
Air Writer — Write in the air with your index finger.

Gestures:
  Index finger only (no pinch)  -> DRAW  (pen down)
  Index + pinch (thumb close)   -> SET WIDTH  (spread thumb away = thicker)
  Two fingers (index + middle)  -> ERASE (rubber)
  Fist                          -> PEN UP (move without drawing)

Keyboard:
  C      -> cycle colour
  R      -> rainbow / cycle hue automatically
  Z      -> undo last stroke
  X      -> clear canvas
  S      -> save PNG
  Q/Esc  -> quit
"""

import cv2
import numpy as np
import time
import colorsys
from gestures import HandTracker

# ---------------------------------------------------------------------------
# Palette (BGR for OpenCV)
# ---------------------------------------------------------------------------
COLORS = [
    (255, 255, 255),   # White   ← default (looks great on dark bg)
    (80,   80, 255),   # Red
    (50,  165, 255),   # Orange
    (0,   230, 255),   # Yellow
    (80,  220,  80),   # Green
    (220, 200,   0),   # Cyan
    (255, 100,  80),   # Blue
    (255,  50, 180),   # Purple
    (180,  80, 255),   # Pink
]
COLOR_NAMES = [
    "White", "Red", "Orange", "Yellow",
    "Green", "Cyan", "Blue", "Purple", "Pink"
]


def rainbow_bgr(hue: float) -> tuple:
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


# ---------------------------------------------------------------------------
# Smoothing helper
# ---------------------------------------------------------------------------
class EMA:
    """Exponential Moving Average for 2-D positions — kills finger jitter."""
    def __init__(self, alpha: float = 0.45):
        self.alpha = alpha     # lower = smoother but more lag
        self._v = None

    def update(self, pos: np.ndarray) -> np.ndarray:
        if self._v is None:
            self._v = pos.astype(float)
        else:
            self._v = self.alpha * pos + (1.0 - self.alpha) * self._v
        return self._v.copy()

    def reset(self):
        self._v = None


# ---------------------------------------------------------------------------
# Canvas that supports undo
# ---------------------------------------------------------------------------
class Canvas:
    MAX_HISTORY = 20

    def __init__(self, W: int, H: int):
        self.W = W
        self.H = H
        self._buf  = np.zeros((H, W, 3), dtype=np.uint8)
        self._hist = []

    # ── snapshot before each new stroke ──────────────────────────────────

    def begin_stroke(self):
        """Call once when pen touches down — saves undo snapshot."""
        self._hist.append(self._buf.copy())
        if len(self._hist) > self.MAX_HISTORY:
            self._hist.pop(0)

    def draw_line(self, p1, p2, color: tuple, width: int):
        cv2.line(
            self._buf,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            color, width, cv2.LINE_AA
        )

    def draw_dot(self, p, color: tuple, width: int):
        """Draw a single dot (for very slow / stationary drawing)."""
        cv2.circle(
            self._buf,
            (int(p[0]), int(p[1])),
            max(1, width // 2),
            color, -1, cv2.LINE_AA
        )

    def erase(self, p, radius: int = 30):
        cv2.circle(self._buf, (int(p[0]), int(p[1])), radius, (0, 0, 0), -1)

    def undo(self):
        if self._hist:
            self._buf = self._hist.pop()

    def clear(self):
        self._hist.append(self._buf.copy())
        self._buf[:] = 0

    @property
    def image(self) -> np.ndarray:
        return self._buf


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

def draw_hud(frame, color, color_name, stroke_w, gesture, fps, rainbow, undo_count):
    H, W = frame.shape[:2]

    # Top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (W, 64), (18, 18, 18), -1)
    cv2.addWeighted(bar, 0.75, frame, 0.25, 0, frame)

    # Colour dot
    cv2.circle(frame, (28, 32), 16, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (28, 32), 16, (255, 255, 255), 1, cv2.LINE_AA)
    if rainbow:
        cv2.putText(frame, "Rainbow", (52, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    else:
        cv2.putText(frame, color_name, (52, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Stroke width preview line
    hw = max(1, stroke_w // 2)
    cx = 52 + 80
    cv2.line(frame, (cx, 44), (cx + 60, 44), color, stroke_w, cv2.LINE_AA)
    cv2.putText(frame, f"W:{stroke_w}", (cx + 68, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1)

    # Gesture badge
    g_col = {
        "DRAW":     (80, 255, 80),
        "RESIZE":   (0, 200, 255),
        "ERASE":    (80, 80, 255),
        "FIST":     (100, 100, 100),
        "IDLE":     (60, 60, 60),
    }.get(gesture, (80, 80, 80))
    tw = cv2.getTextSize(gesture, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)[0][0]
    cv2.putText(frame, gesture, (W // 2 - tw // 2, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, g_col, 2)

    # FPS + undo count
    cv2.putText(frame, f"FPS {fps:4.0f}", (W - 110, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 120, 120), 1)
    cv2.putText(frame, f"Undo:{undo_count}", (W - 110, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

    # Bottom hint bar
    bot = frame.copy()
    cv2.rectangle(bot, (0, H - 36), (W, H), (18, 18, 18), -1)
    cv2.addWeighted(bot, 0.68, frame, 0.32, 0, frame)
    hint = ("[Index] Draw  [Pinch] Set Width  [2-Fingers] Erase  [Fist] Lift  |"
            "  C=colour  R=rainbow  Z=undo  X=clear  S=save  Q=quit")
    cv2.putText(frame, hint, (8, H - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, (130, 130, 130), 1)


def draw_cursor(frame, pos, gesture, color, stroke_w):
    x, y = int(pos[0]), int(pos[1])
    if gesture == "DRAW":
        # Pen nib: filled dot = stroke preview
        r = max(2, stroke_w // 2)
        cv2.circle(frame, (x, y), r, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), r + 2, (255, 255, 255), 1, cv2.LINE_AA)

    elif gesture == "RESIZE":
        # Show a circle representing stroke width
        r = max(2, stroke_w // 2)
        cv2.circle(frame, (x, y), r, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"W={stroke_w}", (x + r + 4, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

    elif gesture == "ERASE":
        cv2.circle(frame, (x, y), 30, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, "ERASE", (x + 32, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("""
==================================================
   Air Writer
==================================================
  Index finger only  -> Draw (pen down)
  Pinch              -> Adjust stroke width
                        (thumb far  = thick)
                        (thumb near = thin)
  Two fingers up     -> Erase
  Fist               -> Lift pen (move freely)

  C = colour   R = rainbow   Z = undo
  X = clear    S = save      Q = quit
==================================================
""")

    # Camera — auto-detect working index
    cap = None
    for idx in range(4):
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            c = cv2.VideoCapture(idx, backend)
            if c.isOpened():
                ret, test = c.read()
                if ret and test is not None:
                    cap = c
                    print(f"[INFO] Camera found: index={idx} backend={backend}")
                    break
                c.release()
            if cap:
                break
        if cap:
            break

    if cap is None:
        print("[ERROR] No working webcam found.")
        print("  Make sure no other app (air_paint, Teams, Zoom) is using it.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    ret, f = cap.read()
    if not ret:
        print("[ERROR] Camera opened but cannot read frame.")
        return
    H, W = f.shape[:2]
    print(f"[INFO] Resolution: {W}x{H}")

    tracker = HandTracker()
    canvas  = Canvas(W, H)
    smoother = EMA(alpha=0.40)   # 0.3=very smooth, 0.6=responsive

    # State
    color_idx    = 0
    stroke_w     = 8
    rainbow      = False
    rainbow_hue  = 0.0

    prev_draw_pos  = None    # last position while drawing
    was_drawing    = False   # True = pen was down last frame
    fps            = 30.0

    while True:
        t0 = time.perf_counter()

        ret, raw = cap.read()
        if not ret:
            break
        frame = cv2.flip(raw, 1)

        # Current colour
        if rainbow:
            rainbow_hue   = (rainbow_hue + 0.005) % 1.0
            current_color = rainbow_bgr(rainbow_hue)
            cname         = "Rainbow"
        else:
            current_color = COLORS[color_idx]
            cname         = COLOR_NAMES[color_idx]

        # ── Hand tracking ──────────────────────────────────────────────────
        result  = tracker.process(frame)
        gesture = "IDLE"
        tip_pos = None

        if result.hand_landmarks:
            hand_lms = result.hand_landmarks[0]
            lm       = tracker.landmarks_px(hand_lms, W, H)
            gesture, index_tip, thumb_tip, velocity = tracker.get_gesture(
                lm, frame.shape
            )

            # Smooth the fingertip position
            raw_pos   = np.array(index_tip, dtype=float)
            smooth_pos = smoother.update(raw_pos)
            tip_pos    = smooth_pos

            # Draw skeleton subtly
            tracker.draw_skeleton(frame, hand_lms, W, H)

            # ── DRAW ─────────────────────────────────────────────────────
            if gesture == "DRAW":
                if not was_drawing:
                    # Pen just touched down — save undo snapshot
                    canvas.begin_stroke()
                    prev_draw_pos = smooth_pos.copy()
                    was_drawing   = True

                if prev_draw_pos is not None:
                    dist = np.linalg.norm(smooth_pos - prev_draw_pos)
                    if dist > 1.5:   # minimum movement threshold
                        canvas.draw_line(prev_draw_pos, smooth_pos,
                                         current_color, stroke_w)
                    elif dist <= 1.5:
                        canvas.draw_dot(smooth_pos, current_color, stroke_w)
                prev_draw_pos = smooth_pos.copy()

            # ── RESIZE (pinch to set width) ───────────────────────────────
            elif gesture == "COLOR_PICK":   # re-use pinch gesture as RESIZE
                gesture = "RESIZE"
                was_drawing   = False
                prev_draw_pos = None
                smoother.reset()

                # Map pinch distance to stroke width
                thumb_pos = np.array(thumb_tip, dtype=float)
                dist      = float(np.linalg.norm(smooth_pos - thumb_pos))
                # dist range: ~20 (tight pinch) to ~150 (wide spread)
                stroke_w  = int(np.clip(
                    np.interp(dist, [20, 150], [1, 40]),
                    1, 40
                ))

            # ── ERASE ─────────────────────────────────────────────────────
            elif gesture == "ERASE":
                canvas.erase(smooth_pos, radius=28)
                was_drawing   = False
                prev_draw_pos = None
                smoother.reset()

            # ── PEN UP / IDLE / FIST ──────────────────────────────────────
            else:
                was_drawing   = False
                prev_draw_pos = None
                smoother.reset()

        else:
            was_drawing   = False
            prev_draw_pos = None
            smoother.reset()
            tracker._history.clear()

        # ── Composite: webcam (dark) + canvas strokes + UI ────────────────
        # Darken the webcam feed so strokes stand out
        darkened = (frame.astype(np.float32) * 0.35).astype(np.uint8)

        # Overlay canvas on darkened feed
        mask    = canvas.image.sum(axis=2) > 0
        display = darkened.copy()
        display[mask] = cv2.addWeighted(
            canvas.image, 0.95, darkened, 0.05, 0
        )[mask]

        # Cursor
        if tip_pos is not None:
            draw_cursor(display, tip_pos, gesture, current_color, stroke_w)

        # HUD
        fps = 0.9 * fps + 0.1 / max(time.perf_counter() - t0, 1e-6)
        draw_hud(display, current_color, cname, stroke_w,
                 gesture, fps, rainbow, len(canvas._hist))

        cv2.imshow("Air Writer", display)

        # ── Keyboard ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('c'):
            color_idx = (color_idx + 1) % len(COLORS)
            rainbow   = False
            print(f"[Color] {COLOR_NAMES[color_idx]}")
        elif key == ord('r'):
            rainbow = not rainbow
            print(f"[Rainbow] {'ON' if rainbow else 'OFF'}")
        elif key == ord('z'):
            canvas.undo()
            print("[Undo]")
        elif key == ord('x'):
            canvas.clear()
            print("[Clear]")
        elif key == ord('s'):
            # Save the clean canvas (white bg version)
            save_img = canvas.image.copy()
            fname    = f"airwrite_{int(time.time())}.png"
            cv2.imwrite(fname, save_img)
            print(f"[Save] {fname}")
        # Width fine-tune with keyboard too
        elif key == ord('['):
            stroke_w = max(1,  stroke_w - 1)
        elif key == ord(']'):
            stroke_w = min(60, stroke_w + 1)

    cap.release()
    cv2.destroyAllWindows()
    print("[Done]")


if __name__ == "__main__":
    main()
