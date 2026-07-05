# 🎨 Air Paint with Physics

A real-time finger-painting app using your webcam. Draw in the air with your index finger — strokes have gravity, drag, bounce, and splatter physics.

---

## Features

| | |
|---|---|
| **6 Brush Modes** | Watercolor, Spray, Neon, Glitter, Oil, Fire |
| **Physics** | Gravity, air drag, floor bounce, wall bounce |
| **Persistent Paint** | Slow particles dry onto a persistent canvas |
| **Gesture Controls** | Index=Draw, Two fingers=Erase, Open palm=Splatter, Fist=Pause |
| **Color Modes** | 10 preset colors + rainbow cycling mode |
| **Screenshot** | Press S to save the current frame |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run
python main.py
```

---

## Gesture Guide

| Hand Shape | Action |
|---|---|
| ☝️ Index finger only | **Draw** — leave physics paint trail |
| ✌️ Index + middle fingers | **Erase** — remove paint in a 45px radius |
| 🤏 Pinch (index + thumb close) | **Cycle colour** |
| ✋ All 4 fingers up | **Splatter** — blast nearby particles outward |
| ✊ Fist | **Pause** — stop drawing |

---

## Keyboard Controls

| Key | Action |
|---|---|
| `B` | Cycle brush mode |
| `C` | Cycle colour |
| `R` | Toggle rainbow mode |
| `X` | Clear canvas |
| `S` | Save screenshot |
| `Q` / `Esc` | Quit |

---

## Brush Modes

| Mode | Behaviour |
|---|---|
| **Watercolor** | Soft, large, transparent drops with gentle gravity |
| **Spray** | Fine mist with wide scatter |
| **Neon** | Glowing streaks, almost zero gravity, long lifetime |
| **Glitter** | Random rainbow sparkles that rain down |
| **Oil** | Thick heavy blobs, slow drag, very long lifetime |
| **Fire** | Particles rise upward, shift from yellow → orange → red |

---

## Physics Details

Each particle has:
- `position (x, y)` — updated every frame
- `velocity (vx, vy)` — inherits some of your finger movement speed
- `gravity` — pulls particles down (negative = up for Fire mode)
- `drag` — slows velocity each frame (simulates air resistance)
- `lifetime` — fades and dies after N frames
- **Floor bounce** — loses 65% energy on hitting the bottom
- **Wall bounce** — reflects off left/right edges

Particles that become nearly stationary get "burned" into a persistent paint layer so paint stays on screen even after the particle dies.

---

## Project Structure

```
air-paint/
├── main.py          ← run this
├── particles.py     ← NumPy particle engine
├── gestures.py      ← MediaPipe hand tracker + gesture recognition
├── brushes.py       ← brush configs + colour palettes
├── renderer.py      ← drawing, glow, HUD
└── requirements.txt
```

---

## Interview Talking Points

- **Why NumPy arrays instead of a list of particle objects?** — Vectorised operations run physics on all particles simultaneously in C, orders of magnitude faster than Python loops
- **Why rule-based gesture recognition instead of a trained classifier?** — 21 MediaPipe landmarks are already structured; a simple geometric rule achieves >98% accuracy without needing labelled data
- **The two-layer paint system** — Active particles move with physics; slow particles get dried onto a persistent canvas with exponential fade, mimicking real wet→dry paint behaviour
- **Velocity inheritance** — Emitted particles inherit finger movement direction, so fast strokes produce streaks and slow strokes produce pools
- **Real-time constraint** — every component must complete in <33ms. NumPy physics + single-canvas batch render keeps the frame loop well under budget
