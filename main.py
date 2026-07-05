"""
Air Paint with Physics — main application.

Controls
--------
Gestures:
  Index finger only  -> DRAW   (physics-based paint trail)
  Index + middle     -> ERASE  (rub away paint)
  All fingers up     -> SPLATTER (explode nearby particles)
  Fist               -> PAUSE  (freeze without drawing)

Keyboard:
  B  -> cycle brush  (Watercolor / Spray / Neon / Glitter / Oil / Fire)
  C  -> cycle colour
  R  -> toggle rainbow mode
  X  -> clear canvas
  S  -> save screenshot as PNG
  Q / Esc -> quit
"""

import cv2
import numpy as np
import time

from particles import ParticleSystem
from gestures  import HandTracker
from brushes   import (
    BRUSH_ORDER, COLORS, COLOR_NAMES,
    BRUSHES, make_particles, rainbow_color,
)
from renderer  import (
    draw_particles, burn_slow_particles,
    fade_paint_layer, apply_paint_layer,
    draw_cursor, draw_hud,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def open_camera(W: int = 1280, H: int = 720):
    for idx in range(4):
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, test = cap.read()
                if ret and test is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    print(f"[INFO] Camera: index={idx} backend={backend}")
                    return cap
                cap.release()
    return None


def print_banner():
    print("""
==================================================
   Air Paint with Physics
==================================================
  Gestures:
    Index finger only  -> DRAW
    Index + middle     -> ERASE
    All fingers up     -> SPLATTER
    Fist               -> PAUSE

  Keyboard:
    B = cycle brush    C = cycle colour
    R = rainbow mode   X = clear canvas
    S = save PNG       Q = quit
==================================================
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print_banner()

    cap = open_camera()
    if cap is None:
        print("[ERROR] No working webcam found. Close other apps using it.")
        return
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot open webcam — check camera index.")
        return

    H, W = frame.shape[:2]
    print(f"[INFO] Camera: {W}x{H}")

    # Core systems
    psys    = ParticleSystem(W, H)
    tracker = HandTracker()

    # State
    brush_idx   = 0
    color_idx   = 0
    rainbow     = False
    rainbow_hue = 0.0

    prev_tip         = None
    fps              = 30.0
    cpick_cooldown   = 0

    # Persistent paint layer (dried paint outlasts active particles)
    paint_layer = np.zeros((H, W, 3), dtype=np.uint8)

    # ---------------------------------------------------------------------------
    # Frame loop
    # ---------------------------------------------------------------------------
    while True:
        t0 = time.perf_counter()

        ret, raw = cap.read()
        if not ret:
            break

        frame = cv2.flip(raw, 1)   # mirror so it feels natural

        # ── Current brush / colour ─────────────────────────────────────────
        brush_name = BRUSH_ORDER[brush_idx]

        if rainbow:
            rainbow_hue   = (rainbow_hue + 0.004) % 1.0
            current_color = rainbow_color(rainbow_hue)
            current_cname = "Rainbow"
        else:
            current_color = COLORS[color_idx]
            current_cname = COLOR_NAMES[color_idx]

        # ── Hand tracking (new Tasks API) ──────────────────────────────────
        result   = tracker.process(frame)
        gesture  = "IDLE"
        tip_pos  = None
        velocity = 0.0

        # result.hand_landmarks is a list of lists (one per detected hand)
        if result.hand_landmarks:
            hand_lms = result.hand_landmarks[0]          # first hand
            lm       = tracker.landmarks_px(hand_lms, W, H)

            gesture, index_tip, thumb_tip, velocity = tracker.get_gesture(
                lm, frame.shape
            )
            tip_pos = index_tip

            # Subtle skeleton overlay
            tracker.draw_skeleton(frame, hand_lms, W, H)

            # ── DRAW ────────────────────────────────────────────────────────
            if gesture == "DRAW":
                x, y = float(index_tip[0]), float(index_tip[1])

                finger_vx, finger_vy = 0.0, 0.0
                if prev_tip is not None:
                    finger_vx = (index_tip[0] - prev_tip[0]) * 0.25
                    finger_vy = (index_tip[1] - prev_tip[1]) * 0.25

                particles = make_particles(brush_name, x, y, velocity, current_color)
                boosted   = [
                    (px, py, vx + finger_vx, vy + finger_vy, c, sz, lf)
                    for (px, py, vx, vy, c, sz, lf) in particles
                ]
                psys.emit_many(boosted)
                prev_tip = index_tip.copy()

            # ── ERASE ───────────────────────────────────────────────────────
            elif gesture == "ERASE":
                x, y = int(index_tip[0]), int(index_tip[1])
                psys.erase_region(x, y, radius=45)
                cv2.circle(paint_layer, (x, y), 45, (0, 0, 0), -1)
                prev_tip = None

            # ── SPLATTER ────────────────────────────────────────────────────
            elif gesture == "SPLATTER":
                psys.splatter(
                    float(index_tip[0]), float(index_tip[1]),
                    radius=230, force=20
                )
                prev_tip = None

            # ── COLOUR PICK (pinch) ──────────────────────────────────────────
            elif gesture == "COLOR_PICK":
                if cpick_cooldown <= 0:
                    color_idx      = (color_idx + 1) % len(COLORS)
                    rainbow        = False
                    cpick_cooldown = 18
                prev_tip = None

            else:
                prev_tip = None
        else:
            prev_tip = None
            tracker._history.clear()

        # ── Cooldown tick ──────────────────────────────────────────────────
        if cpick_cooldown > 0:
            cpick_cooldown -= 1

        # ── Physics update ─────────────────────────────────────────────────
        cfg = BRUSHES[brush_name]
        psys.update(gravity=cfg["gravity"], drag=cfg["drag"])

        # ── Render pipeline ────────────────────────────────────────────────

        # 1. Blend dried paint under webcam feed
        apply_paint_layer(frame, paint_layer)

        # 2. Burn slow / settled particles into persistent layer
        snap = psys.snapshot()
        burn_slow_particles(paint_layer, snap, threshold_spd=0.5)

        # 3. Gradually fade the paint layer (paint "dries")
        fade_paint_layer(paint_layer, rate=0.9985)

        # 4. Draw active particles on top
        draw_particles(frame, snap, brush_name)

        # 5. Fingertip cursor
        if tip_pos is not None:
            draw_cursor(frame, tip_pos, gesture, current_color)

        # 6. HUD overlay
        draw_hud(
            frame, brush_name, current_color, fps,
            psys.alive_count, gesture, rainbow, current_cname
        )

        cv2.imshow("Air Paint with Physics", frame)

        # ── FPS (exponential moving average) ──────────────────────────────
        t1  = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(t1 - t0, 1e-6)

        # ── Keyboard controls ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('b'):
            brush_idx = (brush_idx + 1) % len(BRUSH_ORDER)
            print(f"[Brush] {BRUSHES[BRUSH_ORDER[brush_idx]]['label']}")
        elif key == ord('c'):
            color_idx = (color_idx + 1) % len(COLORS)
            rainbow   = False
            print(f"[Color] {COLOR_NAMES[color_idx]}")
        elif key == ord('r'):
            rainbow = not rainbow
            print(f"[Rainbow] {'ON' if rainbow else 'OFF'}")
        elif key == ord('x'):
            psys.clear()
            paint_layer[:] = 0
            print("[Canvas] Cleared")
        elif key == ord('s'):
            fname = f"airpaint_{int(time.time())}.png"
            cv2.imwrite(fname, frame)
            print(f"[Save] {fname}")

    # ── Cleanup ────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[Done]")


if __name__ == "__main__":
    main()
