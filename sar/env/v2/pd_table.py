"""Detection probability table Pd[level, land cover class, canopy class] (plan section 7.1).

The table lives in `data/pd_table.json`. It is a hand-set placeholder until the M1
detector gate measures recall per class on Isaac renders (plan section 10.3); the
loader accepts the same JSON shape when the measured table replaces it.

The env never reads the JSON directly: it converts the table once into a
[G, G] map per level at reset, so the per-cell lookup in the step function is a
single gather.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from sar.config_v2 import (
    CANOPY_NAMES,
    HIGH,
    LANDCOVER_NAMES,
    LOW,
    N_CANOPY,
    N_LANDCOVER,
    N_LEVELS,
    PD_TABLE_PATH,
)


class PdTable:
    """Pd table plus the lookup from (land cover, canopy) cell indices to Pd."""

    def __init__(self, pd: np.ndarray, meta: dict | None = None):
        if pd.shape != (N_LEVELS, N_LANDCOVER, N_CANOPY):
            raise ValueError(f"Pd table must be [{N_LEVELS},{N_LANDCOVER},{N_CANOPY}], got {pd.shape}")
        self.pd = np.asarray(pd, np.float32)
        self.pd_j = jnp.asarray(self.pd)      # traced lookup, keeps reset vmappable
        self.meta = meta or {}

    @classmethod
    def load(cls, path: Path | str = PD_TABLE_PATH) -> "PdTable":
        raw = json.loads(Path(path).read_text())
        return cls(np.asarray(raw["pd"], np.float32), raw)

    @property
    def measured(self) -> bool:
        return bool(self.meta.get("measured", False))

    def lookup(self, landcover, canopy, level: int):
        """Per-cell Pd for one level: landcover/canopy are int index arrays [G,G].

        Uses the jax copy of the table, so this works on traced values and the
        env's reset stays vmappable over a batch of keys.
        """
        lc = jnp.clip(jnp.asarray(landcover), 0, N_LANDCOVER - 1)
        cp = jnp.clip(jnp.asarray(canopy), 0, N_CANOPY - 1)
        return self.pd_j[int(level), lc, cp]

    def level_maps(self, landcover, canopy, pd_scale: float = 1.0):
        """[2, G, G] Pd per level for the given cell classes."""
        return jnp.stack([self.lookup(landcover, canopy, lv)
                          for lv in range(N_LEVELS)]) * pd_scale

    def describe(self) -> str:
        rows = []
        for lv, name in ((LOW, "low"), (HIGH, "high")):
            for i, ln in enumerate(LANDCOVER_NAMES):
                rows.append(f"  {name:4s} {ln:9s} " + " ".join(f"{v:.3f}" for v in self.pd[lv, i]))
        head = "  landcover      " + " ".join(f"{c:>6s}" for c in CANOPY_NAMES)
        return "\n".join([f"Pd table ({'measured' if self.measured else 'PLACEHOLDER'})", head] + rows)


def pd_channel(pd_maps: np.ndarray, level: int) -> np.ndarray:
    """Observation channel: Pd of one level, clipped into [0,1]."""
    return np.clip(np.asarray(pd_maps)[int(level)], 0.0, 1.0)


def as_jax(pd_maps: np.ndarray):
    return jnp.asarray(np.asarray(pd_maps, np.float32))