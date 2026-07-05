"""
Particle System — NumPy vectorized for real-time performance.
All particles stored in pre-allocated arrays; no Python lists.
"""

import numpy as np

MAX_PARTICLES = 6000


class ParticleSystem:
    def __init__(self, width: int, height: int):
        self.W = width
        self.H = height
        n = MAX_PARTICLES

        # Position
        self.x        = np.zeros(n, np.float32)
        self.y        = np.zeros(n, np.float32)
        # Velocity
        self.vx       = np.zeros(n, np.float32)
        self.vy       = np.zeros(n, np.float32)
        # Color (BGR for OpenCV)
        self.b        = np.zeros(n, np.uint8)
        self.g        = np.zeros(n, np.uint8)
        self.r        = np.zeros(n, np.uint8)
        # Properties
        self.size     = np.zeros(n, np.float32)
        self.life     = np.zeros(n, np.float32)
        self.max_life = np.zeros(n, np.float32)
        self.alive    = np.zeros(n, bool)

        self._cursor  = 0   # circular slot pointer

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def _next_slot(self) -> int:
        """Return an available slot, recycling the oldest if full."""
        start = self._cursor
        for _ in range(MAX_PARTICLES):
            idx = self._cursor
            self._cursor = (self._cursor + 1) % MAX_PARTICLES
            if not self.alive[idx]:
                return idx
        # All slots occupied — kill the one we'd have used
        return start

    def emit(self, x, y, vx, vy, color, size, lifetime):
        idx = self._next_slot()
        self.x[idx]        = x
        self.y[idx]        = y
        self.vx[idx]       = vx
        self.vy[idx]       = vy
        self.b[idx]        = np.clip(color[0], 0, 255)
        self.g[idx]        = np.clip(color[1], 0, 255)
        self.r[idx]        = np.clip(color[2], 0, 255)
        self.size[idx]     = max(1.0, size)
        self.life[idx]     = lifetime
        self.max_life[idx] = lifetime
        self.alive[idx]    = True

    def emit_many(self, particles):
        """Bulk emit. particles = list of (x,y,vx,vy,color,size,life) tuples."""
        for p in particles:
            self.emit(*p)

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def update(self, gravity: float = 0.2, drag: float = 0.97):
        a = self.alive

        # Gravity & drag
        self.vy[a] += gravity
        self.vx[a] *= drag
        self.vy[a] *= drag

        # Move
        self.x[a] += self.vx[a]
        self.y[a] += self.vy[a]

        # Floor bounce  (lose 65% energy on bounce)
        floor = a & (self.y > self.H - 8)
        self.vy[floor]  = -np.abs(self.vy[floor]) * 0.35
        self.y[floor]   = self.H - 8
        self.vx[floor] *= 0.75   # floor friction

        # Side wall bounce
        left  = a & (self.x < 4)
        right = a & (self.x > self.W - 4)
        self.vx[left]  =  np.abs(self.vx[left])  * 0.6
        self.vx[right] = -np.abs(self.vx[right]) * 0.6

        # Age
        self.life[a] -= 1.0
        self.alive[a & (self.life <= 0)] = False

    # ------------------------------------------------------------------
    # Effects
    # ------------------------------------------------------------------

    def splatter(self, x: float, y: float, radius: float = 220, force: float = 20):
        """Blast particles outward from a point."""
        a   = self.alive
        dx  = self.x - x
        dy  = self.y - y
        d   = np.sqrt(dx * dx + dy * dy)
        hit = a & (d < radius) & (d > 0.1)
        if not np.any(hit):
            return
        n       = np.sum(hit)
        norm    = d[hit]
        jitter  = np.random.uniform(0.5, 1.5, n)
        self.vx[hit] += (dx[hit] / norm) * force * jitter
        self.vy[hit] += (dy[hit] / norm) * force * jitter

    def erase_region(self, x: float, y: float, radius: float = 40):
        """Kill all particles within radius."""
        a   = self.alive
        dx  = self.x - x
        dy  = self.y - y
        d   = np.sqrt(dx * dx + dy * dy)
        self.alive[a & (d < radius)] = False

    def clear(self):
        self.alive[:] = False

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    @property
    def alive_count(self) -> int:
        return int(np.sum(self.alive))

    def snapshot(self):
        """Return arrays for alive particles only (read-only views)."""
        a = self.alive
        return (
            self.x[a],  self.y[a],
            self.b[a],  self.g[a],  self.r[a],
            self.size[a],
            self.life[a], self.max_life[a],
            self.vx[a],  self.vy[a],
        )
