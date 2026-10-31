"""Pack a recorded mission into a replay folder the web UI can play.

A replay folder is runs/replays/<name>/ with:
  mission.jsonl   the rows of one mission from the server log
  <view>.mp4      one video per recorded view (follow, overview, cam0, ...)
  timeline.json   for each video, points [mission clock, video time] (seconds)

The mission clock is the `t` of the log rows (the sim clock), and the sim can
run slower than the wall clock, so the videos need this map to play in step
with the map and the time slider.

With a recording folder from scripts/replay_record.py (--rec) every frame has
its mission time, so the map is exact. A plain video (--video, older
recordings) is stretched evenly over the mission, from the first to the last
track row, and the timeline is marked approximate.

  python scripts/replay_pack.py --log runs/demo/demo_rl_1622.jsonl \\
      --video follow=runs/demo/demo_rl_1622.follow.mp4 --name flood_rl_1622
  python scripts/replay_pack.py --log runs/demo/demo_rl_1700.jsonl --rec runs/demo/rec_1700 --name flood_rl_1700
Needs ffmpeg and ffprobe.
"""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_POINTS = 2000         # timeline points per video


def read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def pick_mission(rows: list[dict], index: int) -> list[dict]:
    """The rows of one mission: from its `mission` row to the next one (-1 = last)."""
    starts = [i for i, r in enumerate(rows) if r.get("kind") == "mission"]
    if not starts:
        return rows
    i = starts[index]
    nxt = [s for s in starts if s > i]
    return rows[i: nxt[0] if nxt else len(rows)]


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def encode(src: Path, dst: Path, width: int) -> None:
    """H.264 with a keyframe every 2 s and the index at the front, so the browser can seek."""
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src),
                    "-vf", f"scale='min({width},iw)':-2", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "26", "-g", "20", "-keyint_min", "20", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-an", str(dst)], check=True)


def thin(points: list[list[float]], limit: int = MAX_POINTS) -> list[list[float]]:
    if len(points) <= limit:
        return points
    step = len(points) / float(limit)
    keep = [points[int(i * step)] for i in range(limit)]
    keep[-1] = points[-1]
    return keep


def exact_points(frames: list[list[float]], video_len: float) -> list[list[float]]:
    """[sim, video] per recorded frame, kept increasing in sim time and inside the video."""
    pts = []
    for sim, v in frames:
        if v <= video_len and (not pts or sim > pts[-1][0]):
            pts.append([sim, v])
    return thin(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="mission log (JSON lines)")
    ap.add_argument("--video", action="append", default=[], help="view=path, repeatable")
    ap.add_argument("--rec", default="", help="recording folder from replay_record.py (exact timeline)")
    ap.add_argument("--name", required=True, help="replay folder name under runs/replays")
    ap.add_argument("--mission", type=int, default=-1, help="which mission in the log (default the last)")
    ap.add_argument("--width", type=int, default=1280, help="maximum video width")
    ap.add_argument("--root", default=str(ROOT), help="project folder that holds runs/replays")
    args = ap.parse_args()

    rows = pick_mission(read_rows(Path(args.log)), args.mission)
    tracks = [r["t"] for r in rows if r.get("kind") == "track"]
    if not tracks:
        raise SystemExit("no track rows in that mission")
    root = Path(args.root)
    out = root / "runs" / "replays" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "mission.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    specs = [tuple(v.partition("=")[::2]) for v in args.video]
    frames = {}
    if args.rec:
        raw = json.loads((Path(args.rec) / "timeline_raw.json").read_text())["views"]
        for view, pts in raw.items():
            if len(pts) >= 2 and (Path(args.rec) / f"{view}.mp4").is_file():
                frames[view] = pts
                specs.append((view, str(Path(args.rec) / f"{view}.mp4")))

    videos = {}
    for view, src in specs:
        dst = out / f"{view}.mp4"
        encode(Path(src), dst, args.width)
        length = duration(dst)
        if view in frames:
            points, approx = exact_points(frames[view], length), False
        else:
            points, approx = [[tracks[0], 0.0], [tracks[-1], round(length, 3)]], True
        videos[view] = {"file": dst.name, "duration": round(length, 3), "approx": approx, "points": points}
        print(f"{view}: {length:.1f} s video over {points[-1][0] - points[0][0]:.1f} s mission"
              f"{' (approximate)' if approx else ''}")

    (out / "timeline.json").write_text(json.dumps({"videos": videos}, indent=1))
    print(f"wrote {out.relative_to(root)}")


if __name__ == "__main__":
    main()
