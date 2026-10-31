"""Record the 3D views of a live mission for the replay, with the sim time of every frame.

Reads the MJPEG streams the mission server proxies from the sim bridge
(follow, overview and the drone cameras with detector boxes) and pipes each
JPEG into its own ffmpeg at a fixed output rate. Every frame is stamped with
the mission log clock (`GET /api/clock`), so frame i of a view sits at video
time i / fps and mission time `sim`; scripts/replay_pack.py turns these points
into the replay timeline. Runs until SIGINT or SIGTERM, then closes the videos.

  .venv/bin/python scripts/replay_record.py --out runs/demo/rec_1700 \\
      --views follow,follow1,overview,cam0,cam1
Writes <out>/<view>.mp4 and <out>/timeline_raw.json. Needs httpx and ffmpeg.
"""

import argparse
import json
import signal
import subprocess
import threading
import time
from pathlib import Path

import httpx

SOI, EOI = b"\xff\xd8", b"\xff\xd9"


class Clock:
    """The server's mission log clock, polled in the background and extrapolated between polls."""

    def __init__(self, server: str, poll: float = 0.2):
        self.url, self.poll = f"{server}/api/clock", poll
        self.sim, self.wall, self.rate = self._fetch(), time.monotonic(), 0.4
        threading.Thread(target=self._run, daemon=True).start()

    def _fetch(self) -> float:
        return float(httpx.get(self.url, timeout=3.0).json()["t"])

    def _run(self):
        while True:
            time.sleep(self.poll)
            try:
                s, w = self._fetch(), time.monotonic()
            except (httpx.HTTPError, ValueError, KeyError):
                continue
            if w > self.wall and s > self.sim:
                self.rate = 0.7 * self.rate + 0.3 * (s - self.sim) / (w - self.wall)
            self.sim, self.wall = max(s, self.sim), w

    def now(self) -> float:
        return self.sim + min(time.monotonic() - self.wall, 1.0) * self.rate


def stream_url(server: str, view: str) -> str:
    if view.startswith("cam"):
        return f"{server}/video/camera/{int(view[3:])}.mjpg?full=1"
    if view.startswith("follow") and view[6:]:
        return f"{server}/video/follow/{int(view[6:])}.mjpg?full=1"
    return f"{server}/video/{view}.mjpg?full=1"


def jpegs(url: str, stop: threading.Event):
    """The JPEG frames of an MJPEG stream (split on the JPEG markers); reconnects on errors."""
    while not stop.is_set():
        try:
            with httpx.stream("GET", url, timeout=httpx.Timeout(10.0, read=30.0)) as r:
                r.raise_for_status()
                buf = b""
                for chunk in r.iter_bytes():
                    if stop.is_set():
                        return
                    buf += chunk
                    while True:
                        a = buf.find(SOI)
                        b = buf.find(EOI, a + 2) if a >= 0 else -1
                        if b < 0:
                            break
                        yield buf[a: b + 2]
                        buf = buf[b + 2:]
                    if len(buf) > 8_000_000:       # no frame end in sight: drop the junk
                        buf = b""
        except httpx.HTTPError as exc:
            print(f"{url}: {exc!r}, reconnecting", flush=True)
            stop.wait(2.0)


def record(view: str, url: str, out: Path, fps: int, clock: Clock, stop: threading.Event, points: dict):
    # Fragmented MP4, so a killed recorder still leaves a playable file.
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "image2pipe", "-framerate", str(fps),
                           "-c:v", "mjpeg", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                           "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                           "-movflags", "+frag_keyframe+empty_moov", str(out / f"{view}.mp4")],
                          stdin=subprocess.PIPE)
    pts = points.setdefault(view, [])
    for i, jpg in enumerate(jpegs(url, stop)):
        pts.append([round(clock.now(), 3), round(i / fps, 3)])
        try:
            ff.stdin.write(jpg)
        except BrokenPipeError:
            break
    ff.stdin.close()
    ff.wait()
    print(f"{view}: {len(pts)} frames", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8001")
    ap.add_argument("--clock-server", default="", help="server for /api/clock (default --server)")
    ap.add_argument("--views", default="follow,follow1,overview,cam0,cam1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=10, help="output frame rate (video time = frame / fps)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clock, stop, points = Clock(args.clock_server or args.server), threading.Event(), {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    threads = [threading.Thread(target=record, args=(v, stream_url(args.server, v), out, args.fps, clock, stop, points))
               for v in args.views.split(",")]
    for th in threads:
        th.start()
    while not stop.is_set():
        stop.wait(1.0)
    # A stream that sends nothing keeps its reader blocked; the timeout ends it.
    for th in threads:
        th.join(timeout=40)
    (out / "timeline_raw.json").write_text(json.dumps({"fps": args.fps, "views": points}))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
