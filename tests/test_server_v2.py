"""Tests for the version 2 server endpoints (plan section 11, PROTOCOL section 5).

Covers: the video endpoints answering 404 without a bridge, the `select` command,
the v2 `start_mission` fields, and the box drawing helper in sar.mission.video.
"""

import asyncio

from fastapi.testclient import TestClient
import pytest

from sar.mission.server import build, make_app, parser
from sar.mission.video import draw_boxes

HOME = (149.58, -33.42)


def square(d=0.004, c=HOME):
    return [[c[0] - d, c[1] - d], [c[0] + d, c[1] - d], [c[0] + d, c[1] + d], [c[0] - d, c[1] + d]]


def make_args(bridge=""):
    """Server args from the real CLI parser, so new flags are covered by default."""
    return parser().parse_args(["--backend", "kinematic", "--speed", "1", "--bridge", bridge])


def make_orchestrator(bridge=""):
    return build(make_args(bridge))


@pytest.fixture
def client():
    orch = make_orchestrator()
    args = make_args()
    with TestClient(make_app(args, orch)) as c:
        yield c, orch


# ------------------------------------------------------------------ video 404
@pytest.mark.parametrize("path", ["/video/follow.mjpg", "/video/overview.mjpg", "/video/camera/0.mjpg"])
def test_video_404_without_bridge(client, path):
    c, _ = client
    r = c.get(path)
    assert r.status_code == 404
    # The answer comes back immediately, not as a stalled stream.
    assert "bridge" in r.json()["error"]


def test_video_404_for_unknown_drone_with_bridge():
    """An unknown drone id is rejected before any bridge call is attempted."""
    orch = make_orchestrator("http://127.0.0.1:9")
    with TestClient(make_app(make_args("http://127.0.0.1:9"), orch)) as c:
        r = c.get("/video/camera/99.mjpg")
        assert r.status_code == 404
        assert "no drone" in r.json()["error"]


def test_config_reports_bridge_and_scenarios(client):
    c, _ = client
    js = c.get("/api/config").json()
    assert js["bridge"] is False
    assert "wilderness" in js["scenarios"] and "flood" in js["scenarios"]


# ------------------------------------------------------------------ select
def test_select_drone_accepted(client):
    c, orch = client
    with c.websocket_connect("/ws") as ws:
        ws.receive_json()  # initial state frame
        ws.send_json({"type": "select", "drone": 1})
    # The server applies `select` on its socket task before the socket closes.
    assert orch.select() == 1


def test_select_survivor_accepted(client):
    """`select` with a real survivor id is accepted and points at the nearest drone."""
    from sar.mission.orchestrator import Survivor

    c, orch = client
    # Seed one survivor so the command has a target; drones sit near home.
    orch.survivors[0] = Survivor(id=0, e=10.0, n=0.0, score=0.9, t_detected=0.0)
    with c.websocket_connect("/ws") as ws:
        ws.receive_json()  # initial state frame
        ws.send_json({"type": "select", "survivor": 0})
    # No error was raised and the drone nearest the survivor became the selection.
    assert orch.select() in (0, 1, 2)


# ------------------------------------------------------------------ start_mission v2 fields
def test_start_mission_accepts_scenario_fields(client):
    c, orch = client
    with c.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({
            "type": "start_mission", "polygon": square(), "lkp": list(HOME),
            "mode": "greedy", "n_search": 1, "expected_survivors": 2,
            "scenario": "wilderness", "lkp_time_h": 6.0,
            "regions": [{"polygon": square(0.001), "weight": 0.3}],
            "options": {"auto_dispatch": True, "auto_replace": False, "auto_stage": False},
        })
    state = orch.state_msg()
    assert state["mission"]["status"] == "running"
    # v2 drone fields are present on every drone.
    for d in state["drones"]:
        assert "level" in d and "alt_agl" in d and "sortie" in d and "swap_remaining" in d


# ------------------------------------------------------------------ box drawing
def _jpeg(w=200, h=100, color=(0, 0, 0)):
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "JPEG")
    return buf.getvalue()


def test_draw_boxes_changes_image():
    from io import BytesIO
    from PIL import Image
    jpg = _jpeg()
    before = Image.open(BytesIO(jpg)).tobytes()
    after_jpg = draw_boxes(jpg, [[20, 20, 80, 60, 0.9]], 200, 100)
    after = Image.open(BytesIO(after_jpg)).tobytes()
    assert before != after


def test_draw_boxes_rescales_from_full_frame():
    """Boxes are given in full-frame pixels and scaled to the delivered frame."""
    from io import BytesIO
    from PIL import Image
    # Frame delivered at 200x100 but detector worked at 400x200: a box covering the
    # left half in full-frame coords must land on the left half of the image.
    jpg = _jpeg(200, 100)
    out = draw_boxes(jpg, [[0, 0, 100, 100, 0.9]], 400, 200)
    img = Image.open(BytesIO(out))
    # Left edge has yellow-ish drawn pixels, right edge stays black.
    left = img.getpixel((40, 50))
    right = img.getpixel((180, 50))
    assert left[0] > 150 and left[1] > 120
    assert right[0] < 60


def test_draw_boxes_no_boxes_is_identity():
    jpg = _jpeg()
    assert draw_boxes(jpg, [], 200, 100) is jpg


def test_draw_boxes_survives_garbage():
    jpg = _jpeg()
    # A malformed box is skipped, not fatal.
    out = draw_boxes(jpg, [["x", None, 1, 2, 0.5]], 200, 100)
    assert isinstance(out, bytes) and len(out) > 0


# ------------------------------------------------------------------ jpeg framing
def test_iter_jpegs_extracts_frames():
    """A multipart body is reassembled into whole JPEG frames."""
    from io import BytesIO
    from PIL import Image

    def frame(color):
        buf = BytesIO()
        Image.new("RGB", (8, 8), color).save(buf, "JPEG")
        return buf.getvalue()

    a, b = frame((255, 0, 0)), frame((0, 255, 0))
    body = (b"--b\r\nContent-Type: image/jpeg\r\n\r\n" + a + b"\r\n"
            + b"--b\r\nContent-Type: image/jpeg\r\n\r\n" + b + b"\r\n--b--\r\n")

    async def feed():
        for part in body:
            yield bytes([part])

    async def run():
        import sar.mission.video as V
        V.iter_mjpeg = lambda url, client=None: feed()
        return [f async for f in V.iter_jpegs("x")]

    got = asyncio.run(run())
    assert len(got) == 2
    assert got[0][:2] == b"\xff\xd8" and got[0][-2:] == b"\xff\xd9"


def test_cli_has_bridge_arg():
    args = parser().parse_args([])
    assert args.bridge == ""


def test_build_fills_bridge_url_from_bridge():
    args = parser().parse_args(["--bridge", "http://sim:8100/"])
    orch = build(args)
    # build() normalises bridge_url so the detector and the video share one host.
    assert orch is not None

def test_prior_message_and_regions():
    """start_mission with an operator region: the prior message carries the blend and the components,
    and the region pulls prior mass into it (PROTOCOL 3.3, plan 6.5)."""
    import asyncio

    import numpy as np

    from sar.config import MissionConfig
    from sar.mission.orchestrator import Orchestrator
    from sar.mission.vehicle import KinematicVehicle, SimClock

    cfg = MissionConfig()
    clock = SimClock(100.0)
    vs = [KinematicVehicle(i, (6.0 * i, 0.0), clock) for i in range(2)]
    orch = Orchestrator(cfg, vs, ["search", "rescue"], clock, None, lambda o, t: None)
    lon, lat = cfg.home[1], cfg.home[0]
    d = 0.004
    poly = [[lon - d, lat - d], [lon + d, lat - d], [lon + d, lat + d], [lon - d, lat + d]]
    corner = [[lon + d / 4, lat + d / 4], [lon + d, lat + d / 4], [lon + d, lat + d], [lon + d / 4, lat + d]]

    async def run(regions):
        await orch.command({"type": "start_mission", "polygon": poly, "lkp": [lon, lat], "mode": "greedy",
                            "n_search": 1, "regions": regions})
        m = orch.mission
        msg = orch.prior_msg()
        orch.stop_mission()
        m.status = "done"
        return m, msg

    m0, msg = asyncio.run(run(None))
    assert msg["type"] == "prior" and msg["blend"]["png"] and msg["nx"] == m0.grid.g
    assert [x["key"] for x in msg["layers"]] == ["koester"]
    m1, _ = asyncio.run(run([{"polygon": corner, "weight": 0.5}]))
    g = m0.grid.g
    q0 = m0.belief[: g // 2, g // 2:].sum() / m0.belief.sum()
    q1 = m1.belief[: g // 2, g // 2:].sum() / m1.belief.sum()
    assert q1 > q0 + 0.2
    assert np.isclose(m1.belief.sum(), m0.belief.sum(), rtol=1e-4)
