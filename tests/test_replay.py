"""Tests for the split replay endpoints (plan section 11, step 7).

Covers `GET /api/replays` and `GET /api/replay/<id>` with a small synthetic
mission log, plus the log rows the orchestrator writes for the replay.
"""

import json

import pytest
from fastapi.testclient import TestClient

from sar.mission import replay
from sar.mission.server import build, make_app, parser

HOME = (149.58, -33.42)


def synthetic_log(path, mode="mcts", seed=7, scenario="wilderness"):
    """A small mission log in the orchestrator's JSON-lines format."""
    rows = [
        {"kind": "mission", "t": 100.0, "mode": mode, "g": 32, "cells": 484, "expected": 3.0,
         "scenario": scenario, "seed": seed,
         "polygon": [[149.57, -33.43], [149.59, -33.43], [149.59, -33.41], [149.57, -33.41]]},
        {"kind": "event", "t": 0.0, "type": "event", "level": "info", "text": "mission started"},
        {"kind": "track", "t": 100.0, "found": 0.0, "found_abs": 0.0,
         "drones": [{"id": 0, "name": "S1", "state": "SEARCH", "pos": [149.58, -33.42],
                     "alt": 23.0, "heading": 90.0}]},
        {"kind": "track", "t": 101.0, "found": 0.25, "found_abs": 0.75,
         "drones": [{"id": 0, "name": "S1", "state": "SEARCH", "pos": [149.581, -33.421],
                     "alt": 23.0, "heading": 95.0}]},
        {"kind": "track", "t": 102.0, "found": 0.5, "found_abs": 1.5,
         "drones": [{"id": 0, "name": "S1", "state": "SEARCH", "pos": [149.582, -33.422],
                     "alt": 23.0, "heading": 100.0}]},
        {"kind": "belief", "t": 101.0,
         "corners": [[149.57, -33.43], [149.59, -33.43], [149.59, -33.41], [149.57, -33.41]],
         "png": "AAA", "coverage_png": "BBB"},
        {"kind": "candidate", "t": 100.8, "sid": 0, "e": 100.0, "n": 200.0,
         "pos": [149.5805, -33.4205], "drone": 0, "score": 0.7},
        {"kind": "confirmed", "t": 101.5, "sid": 0, "e": 100.0, "n": 200.0,
         "pos": [149.5805, -33.4205], "how": "persistence check", "true": True},
        {"kind": "reached", "t": 102.5, "sid": 0, "drone": 1,
         "pos": [149.5805, -33.4205], "true": True},
        {"kind": "event", "t": 2.0, "type": "event", "level": "warn", "text": "battery rule"},
    ]
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


@pytest.fixture
def log_dir(tmp_path):
    d = tmp_path / "runs" / "mission"
    d.mkdir(parents=True)
    synthetic_log(d / "alpha.jsonl", mode="mcts", seed=7)
    synthetic_log(d / "beta.jsonl", mode="planner", seed=7)
    return tmp_path


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A server whose replay endpoints read the temporary log folder."""
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    args = parser().parse_args(["--backend", "kinematic", "--speed", "1"])
    orch = build(args)
    import sar.mission.server as server_mod
    monkeypatch.setattr(server_mod, "ROOT", tmp_path)
    with TestClient(make_app(args, orch)) as c:
        yield c


# ------------------------------------------------------------------- listing
def test_replays_lists_logs(client, log_dir):
    r = client.get("/api/replays")
    assert r.status_code == 200
    rows = r.json()["replays"]
    assert len(rows) == 2
    ids = {x["id"] for x in rows}
    assert ids == {"alpha", "beta"}
    by_id = {x["id"]: x for x in rows}
    assert by_id["alpha"]["mode"] == "mcts"
    assert by_id["beta"]["mode"] == "planner"
    assert by_id["alpha"]["seed"] == 7
    assert by_id["alpha"]["scenario"] == "wilderness"
    assert by_id["alpha"]["duration"] == 2.0
    assert by_id["alpha"]["n_frames"] == 3
    assert by_id["alpha"]["n_finds"] == 3


def test_replays_empty_when_no_logs(client, tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "ROOT", tmp_path / "empty")
    import sar.mission.server as server_mod
    monkeypatch.setattr(server_mod, "ROOT", tmp_path / "empty")
    r = client.get("/api/replays")
    assert r.status_code == 200
    assert r.json()["replays"] == []


# -------------------------------------------------------------------- detail
def test_replay_detail_has_time_series(client, log_dir):
    r = client.get("/api/replay/alpha")
    assert r.status_code == 200
    js = r.json()
    assert js["meta"]["mode"] == "mcts"
    assert js["meta"]["seed"] == 7
    assert js["meta"]["scenario"] == "wilderness"
    assert len(js["frames"]) == 3
    # The first frame is at time zero, the last at two seconds.
    assert js["frames"][0]["t"] == 0.0
    assert js["frames"][-1]["t"] == 2.0
    assert js["frames"][1]["drones"][0]["pos"] == [149.581, -33.421]
    assert js["frames"][1]["drones"][0]["name"] == "S1"
    assert js["curve"] == [[0.0, 0.0], [1.0, 0.25], [2.0, 0.5]]
    assert len(js["belief"]) == 1
    assert js["belief"][0]["png"] == "AAA"
    assert len(js["finds"]) == 3
    assert js["finds"][0]["status"] == "candidate"
    assert js["finds"][0]["pos"] == [149.5805, -33.4205]
    assert js["finds"][1]["status"] == "confirmed"
    assert js["finds"][2]["status"] == "reached"


def test_replay_detail_404_for_unknown_id(client, log_dir):
    r = client.get("/api/replay/nope")
    assert r.status_code == 404
    assert "nope" in r.json()["error"]


def test_two_methods_can_be_compared(client, log_dir):
    """The two panes of the split replay: one log each, same scenario and seed."""
    a = client.get("/api/replay/alpha").json()
    b = client.get("/api/replay/beta").json()
    assert a["meta"]["mode"] != b["meta"]["mode"]
    assert a["meta"]["seed"] == b["meta"]["seed"]
    assert a["meta"]["scenario"] == b["meta"]["scenario"]
    # Both curves start at zero so the chart lines up.
    assert a["curve"][0] == b["curve"][0]


# ---------------------------------------------------------------- the parser
def test_parse_log_skips_bad_lines(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"kind": "mission", "t": 1.0, "mode": "rl"}\nnot json\n\n{"kind": "track", "t": 2.0, "found": 0.5, "drones": []}\n')
    parsed = replay.parse_log(p)
    assert parsed["meta"]["mode"] == "rl"
    assert len(parsed["frames"]) == 1


def test_parse_log_subsamples_belief(tmp_path):
    p = tmp_path / "many.jsonl"
    rows = [{"kind": "mission", "t": 0.0, "mode": "rl"}]
    for i in range(200):
        rows.append({"kind": "belief", "t": float(i), "corners": [], "png": f"p{i}", "coverage_png": "c"})
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    parsed = replay.parse_log(p)
    assert len(parsed["belief"]) == replay.MAX_BELIEF_SNAPSHOTS


def test_tier2_record_folder_is_found(tmp_path):
    """Tier 2 record folders hold one mission log per episode."""
    d = tmp_path / "runs" / "tier2" / "rec" / "s1000_mcts"
    d.mkdir(parents=True)
    synthetic_log(d / "mission.jsonl", mode="mcts", seed=1000)
    found = replay.find_logs(tmp_path)
    assert len(found) == 1
    # The id carries the episode folder, so two episodes never collide.
    assert replay.replay_id_of(found[0]) == "s1000_mcts_mission"
    rows = replay.list_replays(tmp_path)
    assert rows[0]["mode"] == "mcts"
    assert rows[0]["seed"] == 1000


# ------------------------------------------------------- the orchestrator log
def test_orchestrator_writes_track_rows(tmp_path):
    """The orchestrator logs the tracks the replay needs (plan 11.2)."""
    import asyncio

    from sar.config import MissionConfig
    from sar.mission.orchestrator import Orchestrator
    from sar.mission.vehicle import KinematicVehicle, SimClock

    log = tmp_path / "mission.jsonl"
    cfg = MissionConfig()
    clock = SimClock(20.0)
    vs = [KinematicVehicle(i, (6.0 * i, 0.0), clock) for i in range(2)]
    orch = Orchestrator(cfg, vs, ["search", "rescue"], clock, None, lambda o, t: None,
                        log_path=log, seed=3)
    lon, lat = cfg.home
    d = 0.004

    async def run():
        await orch.start()
        poly = [[lon - d, lat - d], [lon + d, lat - d], [lon + d, lat + d], [lon - d, lat + d]]
        await orch.command({"type": "start_mission", "polygon": poly, "lkp": [lon, lat],
                             "mode": "greedy", "n_search": 1})
        await asyncio.sleep(3)
        orch.stop_mission("test")
        await asyncio.sleep(0.5)
        await orch.shutdown()

    asyncio.run(run())

    rows = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    kinds = {r["kind"] for r in rows}
    assert "track" in kinds and "mission" in kinds
    mission = next(r for r in rows if r["kind"] == "mission")
    assert mission["scenario"] == "wilderness"
    assert mission["seed"] == 3
    assert mission["polygon"][0][0] == pytest.approx(lon - d)   # [lon, lat] order
    track = next(r for r in rows if r["kind"] == "track")
    assert len(track["drones"]) == 2
    assert len(track["drones"][0]["pos"]) == 2
    assert 0.0 <= track["found"] <= 1.0

    # And the replay module reads that log back.
    parsed = replay.parse_log(log)
    assert parsed["frames"]
    assert parsed["meta"]["mode"] == "greedy"
    assert parsed["meta"]["seed"] == 3


def test_packed_replay_videos(tmp_path):
    """A folder under runs/replays with timeline.json: videos on the replay time axis, files served safely."""
    folder = tmp_path / "runs" / "replays" / "demo1"
    folder.mkdir(parents=True)
    synthetic_log(folder / "mission.jsonl")
    (folder / "follow.mp4").write_bytes(b"\x00" * 64)
    (folder / "timeline.json").write_text(json.dumps({"videos": {
        "follow": {"file": "follow.mp4", "duration": 4.0, "approx": False,
                   "points": [[100.0, 0.0], [102.0, 4.0]]},
        "overview": {"file": "missing.mp4", "points": [[100.0, 0.0]]}}}))

    data = replay.get_replay("demo1_mission", tmp_path)
    assert data["meta"]["n_videos"] == 1
    v = data["videos"]["follow"]
    assert v["url"] == "api/replay_media/demo1/follow.mp4"
    assert v["points"] == [[0.0, 0.0], [2.0, 4.0]]      # shifted to the first track row

    assert replay.media_path("demo1", "follow.mp4", tmp_path) == folder / "follow.mp4"
    assert replay.media_path("demo1", "mission.jsonl", tmp_path) is None
    assert replay.media_path("..", "follow.mp4", tmp_path) is None
    assert replay.media_path("demo1", "nope.mp4", tmp_path) is None


def test_plain_log_has_no_videos(tmp_path):
    log = tmp_path / "runs" / "mission" / "m.jsonl"
    log.parent.mkdir(parents=True)
    synthetic_log(log)
    (log.parent / "timeline.json").write_text(json.dumps({"videos": {"follow": {"file": "m.jsonl"}}}))
    assert replay.get_replay("m", tmp_path)["videos"] == {}


def test_survivor_history_from_finds_and_events():
    """Rejections and dispatches only appear in the event text; the history picks them up."""
    finds = [{"t": 5.0, "sid": 0, "pos": [1.0, 2.0], "status": "candidate", "drone": 1, "score": 0.4, "true": None},
             {"t": 9.0, "sid": 1, "pos": [3.0, 4.0], "status": "candidate", "drone": 0, "score": 0.9, "true": None},
             {"t": 9.5, "sid": 1, "pos": [3.0, 4.0], "status": "confirmed", "drone": None, "score": None, "true": True},
             {"t": 30.0, "sid": 1, "pos": [3.0, 4.0], "status": "reached", "drone": 3, "score": None, "true": True}]
    events = [{"t": 5.2, "level": "info", "text": "survivor #0 rejected by operator"},
              {"t": 9.7, "level": "info", "text": "R1 dispatched to #1 (operator)"}]
    s0, s1 = replay.survivor_history(finds, events)
    assert (s0["t_detected"], s0["t_rejected"], s0["score"]) == (5.0, 5.2, 0.4)
    assert (s1["t_confirmed"], s1["t_assigned"], s1["t_reached"], s1["assigned"]) == (9.5, 9.7, 30.0, 3)
    assert s1["true"] is True
