"""Mission replay: turn orchestrator mission logs into a scrubbable time series.

Plan section 11 (step 7) and the M7 row: the operator compares two methods
(for example `mcts` against `greedy`) on the same scenario and seed, in two
map panes side by side, with one shared time slider.

The server exposes two endpoints (docs/PROTOCOL.md section 6):

  GET /api/replays      the list of available mission logs
  GET /api/replay/<id>  one mission log as a compact time series

The logs are the JSON lines the orchestrator writes through `Orchestrator.log`
(one JSON object per line, `kind` says what it is). The `track` kind carries the
per-tick drone positions and the found-expected value; `belief` carries a
subsampled belief snapshot; `mission`, `event`, `candidate`, `confirmed` and
`reached` carry the rest.

A log file is appended to, so one file can hold several missions. The parser
splits it: every `mission` row starts a new mission, and each mission gets its
own replay id (`<log id>` or `<log id>#2` for the second one).

Everything here is read-only: no mission behaviour depends on this module.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# How much of the log the replay keeps.
MAX_BELIEF_SNAPSHOTS = 60      # belief PNGs per mission
MAX_FRAMES = 900               # track frames per mission
TRACK_DECIMALS = 2

# The mission frame origin, used to place the finds of old logs that stored
# them in local metres. The server sets this from the mission config.
HOME_ORIGIN: tuple[float, float] | None = None


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _round(v, nd=TRACK_DECIMALS):
    return round(float(v), nd) if isinstance(v, (int, float)) else v


def en_to_lonlat(e: float, n: float, origin: tuple[float, float] | None) -> list[float] | None:
    """Local mission metres to [lon, lat], using the mission frame origin.

    Old logs store the finds in the mission frame instead of lon/lat; the
    origin is the configured home point.
    """
    if origin is None:
        return None
    from sar.mission.geo import Frame
    lon, lat = Frame(*origin).to_lonlat(e, n)
    return [round(float(lon), 7), round(float(lat), 7)]


# --------------------------------------------------------------------- finding
def find_logs(root: Path | None = None) -> list[Path]:
    """Every mission log under the project, newest last.

    Two places hold logs: `runs/mission/*.jsonl` (the live server and the
    Tier 1 runs) and the Tier 2 record folders (`--record runs/tier2/rec`),
    where each episode writes `s<seed>_<mode>/mission.jsonl`.
    """
    root = root or ROOT
    out: list[Path] = []
    mission_dir = root / "runs" / "mission"
    if mission_dir.is_dir():
        out += sorted(p for p in mission_dir.glob("*.jsonl") if p.is_file())
    # Tier 2 and other record folders: any directory that holds a mission log.
    runs = root / "runs"
    if runs.is_dir():
        for p in runs.rglob("*.jsonl"):
            if p in out or not p.is_file():
                continue
            if p.name.endswith(".mission.jsonl") or "mission" in p.name:
                out.append(p)
    return sorted(set(out))


def _mode_from_name(path: Path) -> str | None:
    """Tier 2 folders are named `s<seed>_<mode>`; the log inside keeps that name."""
    m = re.match(r"^s(\d+)_(.+)$", path.parent.name)
    return m.group(2) if m else None


def _seed_from_name(path: Path) -> int | None:
    m = re.match(r"^s(\d+)_(.+)$", path.parent.name)
    return int(m.group(1)) if m else None


def replay_id_of(path: Path) -> str:
    """A stable id for one log file: the file stem, plus the folder when it matters.

    Two Tier 2 episodes can both hold a `mission.jsonl`, so the folder name
    (the episode `s<seed>_<mode>`) goes into the id.
    """
    stem = path.stem
    if stem.endswith(".mission"):
        stem = stem[: -len(".mission")]
    parent = path.parent.name
    if parent and parent not in ("mission", "runs") and not parent.startswith("."):
        return f"{parent}_{stem}"
    return stem


# --------------------------------------------------------------------- parsing
def read_rows(path: Path) -> list[dict]:
    """The JSON lines of one log, skipping the lines that do not parse."""
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def split_missions(rows: list[dict]) -> list[list[dict]]:
    """Split the rows of one log into missions.

    A `mission` row starts a new mission. Rows before the first `mission` row
    (an old log, or the tail of a crashed run) form a mission of their own, so
    no data is thrown away.
    """
    out: list[list[dict]] = []
    cur: list[dict] = []
    for row in rows:
        if row.get("kind") == "mission" and cur:
            out.append(cur)
            cur = []
        cur.append(row)
    if cur:
        out.append(cur)
    return out


def parse_mission(rows: list[dict], path: Path, index: int = 0) -> dict:
    """Turn the rows of one mission into the replay time series.

    The result has:
      meta      scenario, mode, seed, start time, duration, grid, expected
      frames    one entry per sampled time step: drone positions, found expected
      belief    subsampled belief snapshots (PNG + corners)
      finds     survivor detections with their times and status
      events    the event log with mission times
      curve     the found-expected curve [[t, value], ...]
    """
    base_id = replay_id_of(path)
    meta = {"id": base_id if index == 0 else f"{base_id}#{index + 1}",
            "path": _rel(path), "scenario": None, "mode": None, "seed": None,
            "started": None, "duration": None, "grid": None, "expected": None, "n_frames": 0,
            "n_finds": 0, "polygon": None, "lkp": None}
    frames: list[dict] = []
    belief: list[dict] = []
    finds: list[dict] = []
    events: list[dict] = []
    curve: list[list[float]] = []
    states: list[dict] = []
    prior = None

    t0 = None            # time of the first frame, used to normalise
    mission_start = None

    for row in rows:
        kind = row.get("kind")
        if kind == "mission":
            meta["mode"] = row.get("mode", meta["mode"])
            meta["grid"] = row.get("g", meta["grid"])
            meta["expected"] = row.get("expected", meta["expected"])
            if row.get("scenario"):
                meta["scenario"] = row["scenario"]
            if row.get("seed") is not None:
                meta["seed"] = row["seed"]
            if row.get("polygon"):
                meta["polygon"] = row["polygon"]
            if mission_start is None:
                mission_start = float(row.get("t", 0.0))
            meta["started"] = mission_start
        elif kind == "track":
            t = float(row.get("t", 0.0))
            if t0 is None:
                t0 = t
            found = row.get("found")
            frames.append({
                "t": _round(t - t0),
                "found": _round(found, 4) if found is not None else None,
                "found_abs": _round(row.get("found_abs"), 4) if row.get("found_abs") is not None else None,
                # Newer logs also carry covered/remaining and the full drone fields
                # of the state message (battery, task, segment, ...); keep them all.
                "covered": row.get("covered"),
                "remaining": row.get("remaining"),
                "drones": [dict(d) for d in row.get("drones", [])],
            })
            if row.get("lkp") and not meta.get("lkp"):
                meta["lkp"] = row["lkp"]
            if found is not None:
                curve.append([_round(t - t0), _round(found, 4)])
        elif kind == "belief":
            belief.append({"t": _round(float(row.get("t", 0.0)) - (t0 or 0.0)),
                           "corners": row.get("corners"),
                           "png": row.get("png"),
                           "coverage_png": row.get("coverage_png")})
        elif kind in ("candidate", "confirmed", "reached"):
            t = float(row.get("t", 0.0))
            finds.append({
                "t": _round(t),
                "sid": row.get("sid"),
                "pos": row.get("pos"),
                "e": row.get("e"),
                "n": row.get("n"),
                "status": kind,
                "drone": row.get("drone"),
                "true": row.get("true"),
                "score": row.get("score"),
            })
        elif kind == "prior":
            prior = {k: v for k, v in row.items() if k not in ("kind", "t")}
        elif kind == "state":
            states.append({"t": float(row.get("t", 0.0)), "drone": row.get("drone"),
                           "state": row.get("state"), "text": row.get("text", "")})
        elif kind == "event":
            events.append({"t": float(row.get("t", 0.0)), "level": row.get("level", "info"),
                           "text": row.get("text", "")})

    # The finds carry the wall-clock time; the replay timeline starts at the
    # first track. Shift them so they sit on the same clock.
    if t0 is not None:
        for f in finds:
            f["t"] = _round(f["t"] - t0)
        for b in belief:
            b["t"] = _round(b["t"] - t0)
        for st in states:
            st["t"] = _round(st["t"] - t0)
    else:
        # No track rows: the mission row (or the first find) starts the clock.
        base = mission_start if mission_start is not None else min((f["t"] for f in finds), default=0.0)
        for f in finds:
            f["t"] = _round(f["t"] - base)

    # Old logs store the finds in the mission frame; turn them into lon/lat.
    for f in finds:
        if f.get("pos") is None and f.get("e") is not None and f.get("n") is not None:
            f["pos"] = en_to_lonlat(f["e"], f["n"], HOME_ORIGIN)

    # A log written before the scenario and seed were logged: the folder name
    # may still say which method and seed the run used.
    if meta["mode"] is None:
        meta["mode"] = _mode_from_name(path)
    if meta["seed"] is None:
        meta["seed"] = _seed_from_name(path)

    # Older logs do not name the scenario; the file or folder name may.
    if meta["scenario"] is None:
        name = path.stem.lower()
        for scenario in ("wilderness", "flood"):
            if scenario in name or scenario in str(path.parent).lower():
                meta["scenario"] = scenario
                break

    # The duration is the last track time. A log without track rows (an old
    # run, before the replay rows were added) falls back to the last find or
    # event time, so the slider still spans the mission.
    end = frames[-1]["t"] if frames else 0.0
    if not frames:
        ends = [f["t"] for f in finds] + [e["t"] for e in events]
        if ends:
            end = max(ends)
    meta["duration"] = _round(max(end, 0.0))
    meta["n_frames"] = len(frames)
    meta["n_finds"] = len(finds)
    videos = load_videos(path, t0 or 0.0)
    meta["n_videos"] = len(videos)

    return {"meta": meta, "frames": subsample_frames(frames), "belief": subsample_belief(belief),
            "finds": finds, "events": events, "curve": curve, "videos": videos,
            "states": states, "survivors": survivor_history(finds, events), "prior": prior}


# The operator decisions that have no log row of their own are in the event text.
REJECT_RE = re.compile(r"survivor #(\d+) rejected")
DISPATCH_RE = re.compile(r"^(\S+) dispatched to #(\d+)")


def survivor_history(finds: list[dict], events: list[dict]) -> list[dict]:
    """One entry per survivor with the replay time of each status change.

    The UI rebuilds the survivor list at any slider time from these times.
    Event times are mission times, which is the replay clock (it starts at the
    mission row, the same time as the first track row).
    """
    out: dict[int, dict] = {}
    for f in sorted(finds, key=lambda f: f["t"]):
        sid = f.get("sid")
        if sid is None:
            continue
        s = out.setdefault(sid, {"id": sid, "pos": f.get("pos"), "score": 0.0, "drone": None, "true": None,
                                 "t_detected": None, "t_confirmed": None, "t_rejected": None,
                                 "t_assigned": None, "assigned": None, "t_reached": None})
        s["pos"] = s["pos"] or f.get("pos")
        if f.get("true") is not None:
            s["true"] = f["true"]
        if f["status"] == "candidate":
            s["t_detected"] = f["t"]
            s["score"] = _round(f.get("score") or 0.0, 3)
            s["drone"] = f.get("drone")
        elif f["status"] == "confirmed":
            s["t_confirmed"] = f["t"]
        elif f["status"] == "reached":
            s["t_reached"] = f["t"]
            s["assigned"] = f.get("drone")
    for e in events:
        m = REJECT_RE.search(e["text"])
        if m and int(m.group(1)) in out:
            out[int(m.group(1))]["t_rejected"] = _round(e["t"])
        m = DISPATCH_RE.search(e["text"])
        if m and int(m.group(2)) in out:
            s = out[int(m.group(2))]
            s["t_assigned"] = _round(e["t"])
            s["assigned_name"] = m.group(1)
    for s in out.values():
        if s["t_detected"] is None:
            s["t_detected"] = min(t for t in (s["t_confirmed"], s["t_reached"]) if t is not None)
    return sorted(out.values(), key=lambda s: s["id"])


# ---------------------------------------------------------------------- videos
# A packed replay (scripts/replay_pack.py) is a folder under runs/replays with the
# mission log, one video per recorded view and timeline.json, which maps the
# mission clock to the time in each video.
REPLAY_DIR = "replays"
MEDIA_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def load_videos(path: Path, t0: float) -> dict:
    """The recorded videos of a packed replay, with points [replay time, video time].

    The timeline stores the log clock; the replay time starts at the first track
    row (`t0`), the same shift `parse_mission` applies to everything else.
    """
    tl = path.parent / "timeline.json"
    if path.parent.parent.name != REPLAY_DIR or not tl.is_file():
        return {}
    try:
        spec = json.loads(tl.read_text()).get("videos", {})
    except (OSError, json.JSONDecodeError):
        return {}
    out = {}
    for view, v in spec.items():
        if not MEDIA_RE.match(v.get("file", "")) or not (path.parent / v["file"]).is_file():
            continue
        out[view] = {"url": f"api/replay_media/{path.parent.name}/{v['file']}",
                     "approx": bool(v.get("approx")), "duration": v.get("duration"),
                     "points": [[_round(s - t0), _round(vt)] for s, vt in v.get("points", [])]}
    return out


def media_path(folder: str, name: str, root: Path | None = None) -> Path | None:
    """The file behind `GET /api/replay_media/<folder>/<name>`, or None if it is not a replay video."""
    if not (MEDIA_RE.match(folder) and MEDIA_RE.match(name)) or not name.endswith(".mp4"):
        return None
    p = (root or ROOT) / "runs" / REPLAY_DIR / folder / name
    return p if p.is_file() else None


def parse_log(path: Path, index: int = 0) -> dict:
    """Parse one mission out of a log file (the first one by default)."""
    missions = split_missions(read_rows(path))
    if not missions:
        return parse_mission([], path, index)
    return parse_mission(missions[min(index, len(missions) - 1)], path, index)


def parse_all(path: Path) -> list[dict]:
    """Every mission in one log file."""
    return [parse_mission(rows, path, i) for i, rows in enumerate(split_missions(read_rows(path)))]


def subsample_belief(belief: list[dict], limit: int = MAX_BELIEF_SNAPSHOTS) -> list[dict]:
    """Keep at most `limit` belief snapshots, evenly spread over the mission."""
    if len(belief) <= limit:
        return belief
    step = len(belief) / float(limit)
    keep = [belief[int(i * step)] for i in range(limit)]
    if belief[-1] is not keep[-1]:
        keep[-1] = belief[-1]
    return keep


def subsample_frames(frames: list[dict], limit: int = MAX_FRAMES) -> list[dict]:
    """Keep at most `limit` track frames, evenly spread over the mission.

    The frames where the found-expected value changes are always kept, so the
    curve the UI draws still turns at the right time.
    """
    if len(frames) <= limit:
        return frames
    step = len(frames) / float(limit)
    keep = {int(i * step) for i in range(limit)}
    keep.add(len(frames) - 1)
    for i, f in enumerate(frames):
        if i and f.get("found") != frames[i - 1].get("found"):
            keep.add(i)
    return [f for i, f in enumerate(frames) if i in keep]


# ------------------------------------------------------------------- endpoints
def is_replayable(parsed: dict) -> bool:
    """True when a mission has something to show: tracks, finds or belief."""
    return bool(parsed["frames"] or parsed["finds"] or parsed["belief"])


def list_replays(root: Path | None = None) -> list[dict]:
    """The rows of `GET /api/replays`: one per mission in every mission log.

    A log file is appended to, so it can hold many missions. Only the missions
    with something to replay are listed; a file whose missions are all empty
    still gets one row, so the operator can see that the log exists.
    """
    out = []
    for path in find_logs(root):
        try:
            missions = parse_all(path)
        except OSError:
            continue
        listed = False
        for parsed in missions:
            if not is_replayable(parsed):
                continue
            m = parsed["meta"]
            out.append(_row(m))
            listed = True
        if not listed and missions:
            out.append(_row(missions[-1]["meta"]))
    return out


def _row(m: dict) -> dict:
    return {
        "id": m["id"], "path": m["path"], "scenario": m["scenario"], "mode": m["mode"],
        "seed": m["seed"], "started": m["started"], "duration": m["duration"],
        "grid": m["grid"], "expected": m["expected"], "n_frames": m["n_frames"],
        "n_finds": m["n_finds"], "n_videos": m.get("n_videos", 0),
    }


def get_replay(replay_id: str, root: Path | None = None) -> dict | None:
    """The body of `GET /api/replay/<id>`, or None when there is no such mission."""
    base, _, suffix = replay_id.partition("#")
    index = int(suffix) - 1 if suffix.isdigit() else 0
    for path in find_logs(root):
        if base not in (replay_id_of(path), _rel(path), path.name, path.stem):
            continue
        missions = parse_all(path)
        if not missions:
            return None
        return missions[min(max(index, 0), len(missions) - 1)]
    return None
