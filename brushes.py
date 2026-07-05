"""
Brush definitions and color palettes.
Each brush config controls particle count, size, lifetime, physics, and visual effects.
"""

import numpy as np
import colorsys

# -----------------------------------------------------------------------
# Color palette  (BGR for OpenCV)
# -----------------------------------------------------------------------
COLORS = [
    (80,   80,  255),   # 0  Red
    (50,  165,  255),   # 1  Orange
    (0,   230,  255),   # 2  Yellow
    (80,  220,   80),   # 3  Green
    (220, 200,    0),   # 4  Cyan
    (255, 100,   80),   # 5  Blue
    (255,  50,  180),   # 6  Purple
    (180,  80,  255),   # 7  Pink
    (255, 255,  255),   # 8  White
    (160, 255,  160),   # 9  Mint
]

COLOR_NAMES = [
    "Red", "Orange", "Yellow", "Green",
    "Cyan", "Blue", "Purple", "Pink", "White", "Mint"
]


def rainbow_color(hue: float):
    """hue in [0, 1] → BGR tuple."""
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


# -----------------------------------------------------------------------
# Brush configurations
# -----------------------------------------------------------------------
BRUSHES = {
    "watercolor": {
        "label":       "Watercolor",
        "count":       14,
        "size_range":  (8, 22),
        "life_range":  (130, 220),
        "spread":      2.5,
        "pos_jitter":  10,
        "gravity":     0.04,
        "drag":        0.992,
        "glow":        False,
        "color_var":   45,
        "random_hue":  False,
    },
    "spray": {
        "label":       "Spray Can",
        "count":       40,
        "size_range":  (1, 5),
        "life_range":  (50, 110),
        "spread":      11.0,
        "pos_jitter":  32,
        "gravity":     0.08,
        "drag":        0.960,
        "glow":        False,
        "color_var":   25,
        "random_hue":  False,
    },
    "neon": {
        "label":       "Neon",
        "count":       10,
        "size_range":  (4, 9),
        "life_range":  (110, 200),
        "spread":      0.8,
        "pos_jitter":  3,
        "gravity":     0.008,
        "drag":        0.998,
        "glow":        True,
        "glow_layers": 4,
        "color_var":   15,
        "random_hue":  False,
    },
    "glitter": {
        "label":       "Glitter",
        "count":       28,
        "size_range":  (2, 6),
        "life_range":  (40, 130),
        "spread":      8.0,
        "pos_jitter":  18,
        "gravity":     0.18,
        "drag":        0.950,
        "glow":        True,
        "glow_layers": 2,
        "color_var":   0,
        "random_hue":  True,
    },
    "oil": {
        "label":       "Oil Paint",
        "count":       7,
        "size_range":  (14, 30),
        "life_range":  (200, 380),
        "spread":      0.4,
        "pos_jitter":  6,
        "gravity":     0.55,
        "drag":        0.870,
        "glow":        False,
        "color_var":   30,
        "random_hue":  False,
    },
    "fire": {
        "label":       "Fire",
        "count":       22,
        "size_range":  (3, 11),
        "life_range":  (35, 85),
        "spread":      2.8,
        "pos_jitter":  6,
        "gravity":     -0.55,
        "drag":        0.975,
        "glow":        True,
        "glow_layers": 3,
        "color_var":   0,
        "random_hue":  False,
        "fire_mode":   True,
    },
}

BRUSH_ORDER = list(BRUSHES.keys())


# -----------------------------------------------------------------------
# Particle factory
# -----------------------------------------------------------------------

def make_particles(brush_name: str, x: float, y: float,
                   finger_vel: float, base_color: tuple) -> list:
    """
    Return a list of (x, y, vx, vy, bgr_color, size, lifetime) tuples
    ready to be fed into ParticleSystem.emit_many().

    finger_vel scales the emission count so faster drawing = more paint.
    """
    cfg = BRUSHES[brush_name]

    # Scale count by finger speed  (clamped 0.4 – 3×)
    vel_scale = float(np.clip(finger_vel / 14.0, 0.4, 3.0))
    count = int(np.clip(cfg["count"] * vel_scale, 4, 80))

    spread    = cfg["spread"]
    pos_j     = cfg["pos_jitter"]
    sz_lo, sz_hi = cfg["size_range"]
    lf_lo, lf_hi = cfg["life_range"]
    var       = cfg["color_var"]
    fire_mode = cfg.get("fire_mode", False)
    rand_hue  = cfg.get("random_hue", False)

    out = []
    for _ in range(count):
        vx = np.random.uniform(-spread, spread)
        vy = np.random.uniform(-spread, spread)
        sz = np.random.uniform(sz_lo, sz_hi)
        lf = np.random.uniform(lf_lo, lf_hi)
        px = x + np.random.uniform(-pos_j, pos_j)
        py = y + np.random.uniform(-pos_j, pos_j)

        if fire_mode:
            # Hot orange/yellow — colour will shift in renderer as particle ages
            hue = np.random.uniform(0.02, 0.12)  # red→yellow range
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            color = (int(b * 255), int(g * 255), int(r * 255))
        elif rand_hue:
            color = rainbow_color(np.random.random())
        else:
            color = (
                int(base_color[0]) + np.random.randint(-var, var + 1),
                int(base_color[1]) + np.random.randint(-var, var + 1),
                int(base_color[2]) + np.random.randint(-var, var + 1),
            )

        out.append((px, py, vx, vy, color, sz, lf))

    return out
