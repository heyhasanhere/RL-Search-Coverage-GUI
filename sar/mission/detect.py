"""Survivor detection sources for the orchestrator (plan sections 4.7, 7).

SimDetector uses ground-truth survivor positions with a detection probability
and a false-positive rate (kinematic runs). The FloodUAV backend has its own
detector (sar/mission/flooduav.py): Unity reports the survivors it finds.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    e: float
    n: float
    score: float
    drone: int
    snapshot: bytes | None = None


class SimDetector:
    """One call per look: survivors inside the camera footprint are seen with prob pd."""

    def __init__(self, truth_en, pd: float, fp: float, footprint_m: float, seed: int = 0, noise_m: float = 2.0):
        self.truth = np.asarray(truth_en, float).reshape(-1, 2)
        self.pd, self.fp, self.half = pd, fp, footprint_m / 2
        self.rng = np.random.default_rng(seed)
        self.noise = noise_m

    async def look(self, drone_id, pos, yaw_deg):
        out = []
        if len(self.truth):
            d = np.abs(self.truth - pos[None, :2]).max(-1)
            for k in np.nonzero(d <= self.half)[0]:
                if self.rng.random() < self.pd:
                    e, n = self.truth[k] + self.rng.normal(0, self.noise, 2)
                    out.append(Detection(float(e), float(n), float(self.rng.uniform(0.6, 0.95)), drone_id))
        if self.rng.random() < self.fp:
            e, n = pos[:2] + self.rng.uniform(-self.half, self.half, 2)
            out.append(Detection(float(e), float(n), float(self.rng.uniform(0.3, 0.6)), drone_id))
        return out
