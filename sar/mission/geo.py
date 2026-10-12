"""Local ENU frame, area grid, and polygon rasterization (plan section 4.8).

The frame is anchored at a fixed origin (the launch site). Positions are
(east, north) in meters. Grid rows run north to south, columns west to east,
so grid row r, column c matches the JAX env's (row, col) cell layout.
"""

import math
from dataclasses import dataclass

import numpy as np

R_EARTH = 6378137.0


@dataclass(frozen=True)
class Frame:
    lon0: float
    lat0: float

    def to_en(self, lon, lat):
        e = np.radians(np.asarray(lon) - self.lon0) * R_EARTH * math.cos(math.radians(self.lat0))
        n = np.radians(np.asarray(lat) - self.lat0) * R_EARTH
        return e, n

    def to_lonlat(self, e, n):
        lon = self.lon0 + np.degrees(np.asarray(e) / (R_EARTH * math.cos(math.radians(self.lat0))))
        lat = self.lat0 + np.degrees(np.asarray(n) / R_EARTH)
        return lon, lat


def points_in_polygon(x, y, poly):
    """Even-odd ray casting. x, y arrays; poly [K,2]. Returns bool array."""
    inside = np.zeros(np.shape(x), bool)
    px, py = poly[:, 0], poly[:, 1]
    k = len(poly)
    for i in range(k):
        x1, y1, x2, y2 = px[i], py[i], px[(i + 1) % k], py[(i + 1) % k]
        cond = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (x < xi)
    return inside


class AreaGrid:
    """A g x g cell grid covering the polygon, centered on its bounding box."""

    def __init__(self, frame: Frame, polygon_lonlat, cell_m: float, g: int):
        self.frame, self.cell, self.g = frame, cell_m, g
        poly = np.asarray(polygon_lonlat, float)
        if len(poly) > 1 and np.allclose(poly[0], poly[-1]):
            poly = poly[:-1]
        if len(poly) < 3:
            raise ValueError("polygon needs at least 3 vertices")
        e, n = frame.to_en(poly[:, 0], poly[:, 1])
        self.poly_en = np.stack([e, n], -1)
        side = g * cell_m
        if e.max() - e.min() > side or n.max() - n.min() > side:
            raise ValueError(f"area is larger than {side / 1000:.2f} km on a side")
        ce, cn = (e.max() + e.min()) / 2, (n.max() + n.min()) / 2
        self.e0 = ce - side / 2            # west edge
        self.n0 = cn + side / 2            # north edge
        rr, cc = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
        ex, ny = self.cell_center(rr, cc)
        self.mask = points_in_polygon(ex, ny, self.poly_en)
        if not self.mask.any():
            raise ValueError("polygon covers no grid cell")

    def cell_center(self, r, c):
        return self.e0 + (np.asarray(c) + 0.5) * self.cell, self.n0 - (np.asarray(r) + 0.5) * self.cell

    def cell_of(self, e, n):
        c = int(math.floor((e - self.e0) / self.cell))
        r = int(math.floor((self.n0 - n) / self.cell))
        return r, c

    def in_grid(self, r, c):
        return 0 <= r < self.g and 0 <= c < self.g

    def corners_lonlat(self):
        """Top-left, top-right, bottom-right, bottom-left as [lon, lat]."""
        side = self.g * self.cell
        pts = [(self.e0, self.n0), (self.e0 + side, self.n0),
               (self.e0 + side, self.n0 - side), (self.e0, self.n0 - side)]
        return [list(map(float, self.frame.to_lonlat(e, n))) for e, n in pts]

    @classmethod
    def from_mask(cls, frame: Frame, mask, cell_m: float, center_en=(0.0, 0.0)):
        """Grid for a cell mask (Tier 0/1 maps), centered at center_en."""
        self = cls.__new__(cls)
        g = mask.shape[0]
        self.frame, self.cell, self.g = frame, cell_m, g
        side = g * cell_m
        self.e0 = center_en[0] - side / 2
        self.n0 = center_en[1] + side / 2
        self.mask = np.asarray(mask, bool)
        rows, cols = np.nonzero(self.mask)
        e, n = self.cell_center(rows, cols)
        self.poly_en = np.stack([e, n], -1)   # cell centers stand in for the outline
        return self


def koester_prior(grid: AreaGrid, lkp_en=None, floor: float = 0.15):
    """Ring prior around the last-known point plus a uniform floor (plan 5.2).

    Ring distances are the 25/50/75/95% quantiles of the Koester lost-person
    tables, scaled to the size of the drawn area. Without a last-known point
    the prior is uniform. Returns a [g,g] map that sums to 1 inside the mask.
    """
    m = grid.mask
    uni = m / m.sum()
    if lkp_en is None:
        return uni
    rr, cc = np.meshgrid(np.arange(grid.g), np.arange(grid.g), indexing="ij")
    ex, ny = grid.cell_center(rr, cc)
    d = np.hypot(ex - lkp_en[0], ny - lkp_en[1])
    rows, cols = np.nonzero(m)
    extent = max(np.ptp(rows), np.ptp(cols), 1) * grid.cell
    q = extent / 6.0 * np.array([0.4, 1.0, 1.8, 3.2])
    edges = np.concatenate([[0.0], q])
    dens = np.array([0.25, 0.25, 0.25, 0.20]) / (np.pi * (edges[1:] ** 2 - edges[:-1] ** 2))
    band = np.searchsorted(q, d)
    ring = np.where(band < 4, dens[np.minimum(band, 3)], 0.05 / (np.pi * q[-1] ** 2))
    ring = np.where(m, ring, 0.0)
    ring = ring / ring.sum()
    return (1 - floor) * ring + floor * uni
