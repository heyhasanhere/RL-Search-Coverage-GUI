"""Record a running mission from the UI and pack it into a replay folder.

`POST /api/record/start` runs scripts/replay_record.py against this server, so it
records every 3D view (both follow views, the overview and the drone cameras).
`POST /api/record/stop` stops it and runs scripts/replay_pack.py on the latest
mission of the server log, which writes runs/replays/<name>/. Without a sim
bridge there are no views, and the replay holds the mission log only.
"""

import asyncio
import re
import shutil
import signal
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")


class Recorder:
    def __init__(self, root: Path, server: str, log: Path | None, views: list[str]):
        self.root, self.server, self.log, self.views = root, server, log, views
        self.videos = bool(views)
        self.proc: asyncio.subprocess.Process | None = None
        self.state = "idle"            # idle, recording, packing
        self.name = ""
        self.started = 0.0
        self.last = ""                 # the name of the last packed replay
        self.error = ""

    def status(self) -> dict:
        return {"state": self.state, "name": self.name, "videos": self.videos,
                "elapsed": round(time.time() - self.started, 1) if self.state == "recording" else 0.0,
                "last": self.last, "error": self.error}

    def _name(self, name: str) -> str:
        name = NAME_RE.sub("_", name.strip()).strip("_")[:60]
        name = name or time.strftime("rec_%Y%m%d_%H%M%S")
        base, i = name, 2
        while (self.root / "runs" / "replays" / name).exists():
            name, i = f"{base}_{i}", i + 1
        return name

    def _rec_dir(self) -> Path:
        return self.root / "runs" / "recordings" / self.name

    async def start(self, name: str = "") -> str:
        """Start recording; returns an error message, or an empty string."""
        if self.state != "idle":
            return f"already {self.state}"
        if self.log is None:
            return "the server has no mission log (--log)"
        if self.videos and shutil.which("ffmpeg") is None:
            return "recording the views needs ffmpeg"
        self.name, self.error = self._name(name), ""
        if self.videos:
            out = self._rec_dir()
            out.mkdir(parents=True, exist_ok=True)
            self.proc = await asyncio.create_subprocess_exec(
                sys.executable, str(SCRIPTS / "replay_record.py"),
                "--server", self.server, "--out", str(out), "--views", ",".join(self.views),
                stdout=(out / "record.log").open("w"), stderr=asyncio.subprocess.STDOUT)
        self.state, self.started = "recording", time.time()
        return ""

    async def stop(self) -> str:
        """Stop recording and pack the replay in the background."""
        if self.state != "recording":
            return "not recording"
        self.state = "packing"
        asyncio.create_task(self._finish())
        return ""

    async def _finish(self):
        try:
            if self.proc is not None:
                if self.proc.returncode is None:
                    self.proc.send_signal(signal.SIGINT)
                try:
                    # the recorder gives each view up to 40 s to close its video
                    await asyncio.wait_for(self.proc.wait(), 60.0 + 40.0 * len(self.views))
                except asyncio.TimeoutError:
                    self.proc.kill()
                    await self.proc.wait()
                self.proc = None
            cmd = [sys.executable, str(SCRIPTS / "replay_pack.py"),
                   "--log", str(self.log), "--name", self.name, "--root", str(self.root)]
            rec = self._rec_dir()
            if (rec / "timeline_raw.json").is_file():
                cmd += ["--rec", str(rec)]
            pack = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.STDOUT)
            out, _ = await pack.communicate()
            if pack.returncode != 0:
                lines = out.decode(errors="replace").strip().splitlines()
                self.error = f"packing failed: {lines[-1] if lines else pack.returncode}"
            else:
                self.last = self.name
        except Exception as exc:       # the UI shows it; the server keeps running
            self.error = f"recording failed: {exc!r}"
        finally:
            self.state = "idle"

    async def shutdown(self):
        if self.proc is not None and self.proc.returncode is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                await asyncio.wait_for(self.proc.wait(), 10.0)
            except asyncio.TimeoutError:
                self.proc.kill()
