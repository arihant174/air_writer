"""
Rendering helpers — draw particles onto frames with glow, fire colouring,
and overlay blending.  All drawing is batched onto a single canvas per frame
so only one cv2.addWeighted call is needed at the end.
"""

import cv2
import numpy as np
import colorsys

from brushes import BRUSHES, BRUSH_ORDER, COLORS, COLOR_NAMES


def _fire_color(life_ratio: float) -> tuple:
    """Map life_ratio (1=just born, 0=dying) to fire colours (BGR)."""
    # 1.0 → bright yellow-white core
    # 0.5 → orange
    # 0.0 → deep red / dark
    if life_ratio > 0.7:
        hue = 0.12        # yellow
    elif life_ratio > 0.4:
        hue = 0.06        # orange
    else:
        hue = 0.01        # red
    brightness = 0.4 + 0.6 * life_ratio
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, brightness)
    return (int(b * 255), int(g * 255), int(r * 255))


# -----------------------------------------------------------------------
# Main particle draw  (called once per frame)
# -----------------------------------------------------------------------

def draw_particles(frame: np.ndarray, snap, brush_name: str) -> np.ndarray:
    """
    snap  : tuple from ParticleSystem.snapshot()
    Draws all alive particles onto `frame` in-place and returns it.
    """
    if snap is None or len(snap[0]) == 0:
        return frame

    x_a, y_a, b_a, g_a, r_a, sz_a, lf_a, mlf_a, vx_a, vy_a = snap
    H, W = frame.shape[:2]
    cfg       = BRUSHES.get(brush_name, BRUSHES["watercolor"])
    glow      = cfg.get("glow", False)
    glow_lyrs = cfg.get("glow_layers", 3)
    fire_mode = cfg.get("fire_mode", False)

    # One shared canvas — draw everything here, blend once
    canvas = np.zeros_like(frame, dtype=np.uint8)

    n = len(x_a)
    for i in range(n):
        xi = int(x_a[i])
        yi = int(y_a[i])
        if xi < 0 or xi >= W or yi < 0 or yi >= H:
            continue

        lr   = float(lf_a[i]) / float(mlf_a[i])   # life ratio [0,1]
        sz   = max(1, int(sz_a[i]))

        if fire_mode:
            color = _fire_color(lr)
        else:
            color = (int(b_a[i]), int(g_a[i]), int(r_a[i]))

        if glow:
            # Outer glow rings (drawn first, so core paints over)
            for layer in range(glow_lyrs, 0, -1):
                gsz    = sz + layer * 5
                factor = 0.25 / layer
                gc     = tuple(int(c * factor) for c in color)
                cv2.circle(canvas, (xi, yi), gsz, gc, -1, cv2.LINE_AA)

        # Core circle — full brightness
        cv2.circle(canvas, (xi, yi), sz, color, -1, cv2.LINE_AA)

    # Blend canvas onto frame
    # Use src-over style: where canvas has paint, mostly show canvas
    mask = canvas.sum(axis=2) > 0
    np.copyto(frame, cv2.addWeighted(canvas, 0.88, frame, 0.12, 0), where=mask[:, :, None])

    return frame


# -----------------------------------------------------------------------
# Persistent paint layer helpers
# -----------------------------------------------------------------------

def burn_slow_particles(paint_layer: np.ndarray, snap, threshold_spd: float = 0.6):
    """
    Particles that are nearly stationary get 'dried' onto paint_layer
    so they persist after the particle dies.
    """
    if snap is None or len(snap[0]) == 0:
        return
    x_a, y_a, b_a, g_a, r_a, sz_a, lf_a, mlf_a, vx_a, vy_a = snap
    H, W = paint_layer.shape[:2]

    speed  = np.sqrt(vx_a ** 2 + vy_a ** 2)
    lr     = lf_a / np.maximum(mlf_a, 1e-6)
    slow   = (speed < threshold_spd) & (lr < 0.55)
    idx    = np.where(slow)[0]

    for i in idx:
        xi, yi = int(x_a[i]), int(y_a[i])
        if 0 <= xi < W and 0 <= yi < H:
            sz = max(1, int(sz_a[i]))
            c  = (int(b_a[i]), int(g_a[i]), int(r_a[i]))
            cv2.circle(paint_layer, (xi, yi), sz, c, -1, cv2.LINE_AA)


def fade_paint_layer(paint_layer: np.ndarray, rate: float = 0.997):
    """Very slow fade so paint dries gradually. Operates in-place."""
    np.multiply(paint_layer, rate, out=paint_layer, casting="unsafe")


def apply_paint_layer(frame: np.ndarray, paint_layer: np.ndarray):
    """Blend paint_layer over the webcam frame."""
    mask = paint_layer.sum(axis=2) > 0
    blended = cv2.addWeighted(paint_layer, 0.75, frame, 0.25, 0)
    np.copyto(frame, blended, where=mask[:, :, None])


# -----------------------------------------------------------------------
# Cursor
# -----------------------------------------------------------------------

def draw_cursor(frame: np.ndarray, pos, gesture: str, color: tuple, sz: int = 14):
    x, y = int(pos[0]), int(pos[1])
    if gesture == "DRAW":
        cv2.circle(frame, (x, y), sz, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 3,  color, -1)
    elif gesture == "ERASE":
        cv2.rectangle(frame, (x - 22, y - 22), (x + 22, y + 22), (255, 255, 255), 2)
        cv2.putText(frame, "ERASE", (x + 25, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    elif gesture == "SPLATTER":
        for ang in range(0, 360, 45):
            rad = np.radians(ang)
            ex  = int(x + 28 * np.cos(rad))
            ey  = int(y + 28 * np.sin(rad))
            cv2.line(frame, (x, y), (ex, ey), (0, 180, 255), 2, cv2.LINE_AA)
    elif gesture == "COLOR_PICK":
        cv2.circle(frame, (x, y), sz, color, -1)
        cv2.circle(frame, (x, y), sz + 3, (255, 255, 255), 1)


# -----------------------------------------------------------------------
# HUD / UI overlay
# -----------------------------------------------------------------------

_GESTURE_COLORS = {
    "DRAW":       (80,  255, 80),
    "ERASE":      (80,  80,  255),
    "SPLATTER":   (0,   180, 255),
    "FIST":       (100, 100, 100),
    "COLOR_PICK": (200, 100, 255),
    "IDLE":       (70,  70,  70),
}


def draw_hud(frame: np.ndarray, brush_name: str, color: tuple,
             fps: float, particle_count: int, gesture: str,
             rainbow: bool, color_name: str):
    H, W = frame.shape[:2]

    # ── Top bar ──────────────────────────────────────────────────────────
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (W, 68), (15, 15, 15), -1)
    cv2.addWeighted(bar, 0.72, frame, 0.28, 0, frame)

    brush_label = BRUSHES[brush_name]["label"]
    cv2.putText(frame, brush_label, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (220, 220, 220), 2)

    # Color dot or RAINBOW tag
    if rainbow:
        cv2.putText(frame, "RAINBOW MODE", (14, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 1)
    else:
        cv2.circle(frame, (220, 50), 13, color, -1)
        cv2.circle(frame, (220, 50), 13, (255, 255, 255), 1)
        cv2.putText(frame, color_name, (240, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

    # Gesture badge (centred)
    gc   = _GESTURE_COLORS.get(gesture, (80, 80, 80))
    gtxt = gesture.replace("_", " ")
    tw   = cv2.getTextSize(gtxt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
    cv2.putText(frame, gtxt, (W // 2 - tw // 2, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, gc, 2)

    # FPS + particle count (right)
    cv2.putText(frame, f"FPS {fps:4.0f}", (W - 110, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)
    cv2.putText(frame, f"{particle_count:4d} pts", (W - 110, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 100, 100), 1)

    # ── Bottom help bar ──────────────────────────────────────────────────
    bot = frame.copy()
    cv2.rectangle(bot, (0, H - 36), (W, H), (15, 15, 15), -1)
    cv2.addWeighted(bot, 0.65, frame, 0.35, 0, frame)

    hint = ("[Index] Draw   [2-fingers] Erase   [Palm] Splatter   [Fist] Pause  |"
            "  B=brush  C=color  R=rainbow  X=clear  S=save  Q=quit")
    cv2.putText(frame, hint, (8, H - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)
