"""Flight layer tests (plan section 19): the floor is never broken along a segment on synthetic
terrain, and avoidance clears a pole placed on the path."""

import numpy as np

from sar.mission.flight import FlightConfig, FlightLayer, Terrain


def _hills(g=32, cell=30.0):
    x = np.arange(g) * cell
    xx, yy = np.meshgrid(x, x)
    dem = 40 * np.sin(xx / 200.0) * np.cos(yy / 170.0)
    canopy = np.where(np.sin(xx / 90.0) > 0, 22.0, 0.0)
    return Terrain(dem, canopy, cell_m=cell)


def test_floor_never_broken_along_segment():
    t = _hills()
    fl = FlightLayer(t)
    home = np.array([100.0, 100.0])
    p0, p1 = np.array([60.0, 60.0]), np.array([900.0, 850.0])
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    for s in np.linspace(0, 1, 400):
        e, n = p0 + s * (p1 - p0)
        sp = fl.to_vehicle(0, home, (e, n, 23.0), heading=u)
        agl = fl.agl(home, sp)
        surf = float(t.surface(np.array([e]), np.array([n]))[0])
        assert agl >= surf + fl.cfg.clearance - 1e-6
        assert agl <= fl.cfg.ceiling + 1e-6
        assert agl >= 23.0 - 1e-6


def test_flat_terrain_is_identity():
    fl = FlightLayer(Terrain.flat())
    sp = fl.to_vehicle(1, np.array([0.0, 0.0]), (50.0, 40.0, 23.0))
    assert np.allclose(sp, [50.0, 40.0, 23.0])
    assert FlightLayer(None).to_vehicle(0, (0, 0), (1, 2, 3)).tolist() == [1, 2, 3]


def test_terrain_follows_ground():
    dem = np.tile(np.linspace(0, 100, 16), (16, 1))   # rises 100 m to the east
    fl = FlightLayer(Terrain(dem, cell_m=30.0))
    home = np.array([15.0, 240.0])
    lo = fl.to_vehicle(0, home, (15.0, 240.0, 23.0))
    hi = fl.to_vehicle(0, home, (465.0, 240.0, 23.0))
    assert abs(lo[2] - 23.0) < 1e-3
    assert abs(hi[2] - (23.0 + 100.0)) < 1.0


def test_avoidance_clears_pole():
    """A pole 40 m ahead and 60 m tall: the drone stops and climbs until the depth view is clear."""
    fl = FlightLayer(Terrain.flat(), FlightConfig(avoid_m=15.0))
    pole_e, pole_top = 40.0, 60.0
    e, h, dt = 0.0, 23.0, 0.05
    for _ in range(4000):
        alt = h + fl.bias.get(0, 0.0)
        dist = pole_e - e
        seen = 0.0 < dist < 30.0 and alt < pole_top + 2.0
        depth = np.full((8, 8), dist if seen else 100.0)
        blocked = fl.sense(0, dt, range_m=alt, depth=depth)
        if not blocked:
            e += 5.0 * dt
        # never inside the pole's 15 m stop distance below its top
        assert not (abs(pole_e - e) < 1.0 and alt < pole_top)
        if e > pole_e + 20:
            break
    assert e > pole_e + 20
    assert fl.stats["detours"] >= 1


def test_terrain_hold_climbs_on_low_range():
    fl = FlightLayer(Terrain.flat())
    for _ in range(10):
        fl.sense(0, 0.05, range_m=4.0)
    assert fl.bias[0] > 5.0
    for _ in range(2000):
        fl.sense(0, 0.05, range_m=40.0)
    assert fl.bias[0] == 0.0
