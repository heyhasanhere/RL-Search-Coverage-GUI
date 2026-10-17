"""FastAPI server: WebSocket for the operator UI plus the static UI build.

Run, for example:
  uv run python -m sar.mission.server --backend kinematic --speed 5
  uv run python -m sar.mission.server --backend flooduav --bridge http://127.0.0.1:8101   (sim/flooduav/README.md)
Protocol: docs/PROTOCOL.md.
"""

import argparse
import asyncio
import contextlib
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from sar.config import MissionConfig
from sar.mission.detect import SimDetector
from sar.mission.orchestrator import Orchestrator
from sar.mission import replay
from sar.mission.scenario import sample_survivors
from sar.mission.search_policy import MODES
from sar.mission.vehicle import BridgeClock, KinematicVehicle, SimClock
from sar.mission.video import frames, frames_with_boxes, set_view, stream_headers

ROOT = Path(__file__).resolve().parents[2]


def home_pads(cfg: MissionConfig, n: int):
    return [(cfg.home_spacing * (i - (n - 1) / 2), 0.0) for i in range(n)]


def build(args) -> Orchestrator:
    args.bridge_url = args.bridge.rstrip("/")
    cfg = MissionConfig(pd=args.pd)
    if getattr(args, "home", ""):
        # the launch site must be the world origin, so the map and the drones share one frame
        lon, lat = (float(v) for v in args.home.split(","))
        cfg = replace(cfg, home=(lon, lat))
    if args.backend == "flooduav":
        return build_flooduav(args, cfg)
    clock = SimClock(args.speed)
    roles = args.roles.split(",")
    pads = home_pads(cfg, len(roles))
    vehicles = [KinematicVehicle(i, pads[i], clock, max_speed=cfg.transit_speed) for i in range(len(roles))]
    rng = np.random.default_rng(args.seed)

    def detector_factory(orch, truth_en):
        m = orch.mission
        if truth_en is None and args.truth_from_prior:
            prior = m.belief / m.belief.sum()
            n = max(1, int(rng.poisson(m.expected))) if args.poisson else max(1, int(round(m.expected)))
            truth_en = sample_survivors(m.grid, prior, n, rng)
            orch.truth_en = truth_en
        return SimDetector(truth_en if truth_en is not None else np.zeros((0, 2)), pd=args.pd, fp=args.fp,
                           footprint_m=cfg.cell_m, seed=int(rng.integers(1 << 30)))

    ckpt = Path(args.ckpt) if args.ckpt else None
    log = Path(args.log) if args.log else None
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
    orch = Orchestrator(cfg, vehicles, roles, clock, ckpt, detector_factory, log_path=log, seed=args.seed)
    if args.flood_layers and Path(args.flood_layers).exists():
        from sar.prior import flood
        orch.prior_builder = flood.make_prior_builder(args.flood_layers)
    return orch


def build_flooduav(args, cfg: MissionConfig) -> Orchestrator:
    """The two Unity drones; Unity confirms the survivors; the scene boundary is the search area."""
    from sar.mission.flooduav import FloodDetector, FloodVehicle, fetch_info
    if not args.bridge:
        raise SystemExit("--backend flooduav needs --bridge <FloodUAV bridge url>")
    info = fetch_info(args.bridge_url)
    clock = BridgeClock(args.bridge_url)
    vehicles = [FloodVehicle(i, clock, args.bridge_url, alt_offset=info["min_alt"] - cfg.search_alt,
                             camera_fov_deg=info.get("camera_fov", 60.0))
                for i in range(len(info.get("homes", [0])))]
    truth = np.array([[s["e"], s["n"]] for s in info["survivors"]], float)

    def detector_factory(orch, truth_en):
        orch.truth_en = truth
        return FloodDetector(vehicles, truth)

    async def restart():
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            (await client.post(f"{args.bridge_url}/reset")).raise_for_status()
        for v in vehicles:
            v.landed = True
        await clock.sleep(3.0)       # Unity resets on its next step; the drones settle on their pads

    log = Path(args.log) if args.log else None
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt) if args.ckpt else None
    roles = ["search", "rescue", "rescue"][: len(vehicles)]
    orch = Orchestrator(cfg, vehicles, roles, clock, ckpt, detector_factory, log_path=log, seed=args.seed)
    orch.scenarios = ("flood",)
    orch.flood_info = info
    orch.on_restart = restart
    for v in vehicles:
        v.on_reset = lambda reason: orch.event(f"Unity mission ended ({reason}); drones reset to their pads", "warn")
    return orch


def flood_extras(orch) -> dict:
    """Search area and map background for the UI, in lon/lat."""
    info = getattr(orch, "flood_info", None)
    if info is None:
        return {}
    ll = lambda p: [round(float(v), 7) for v in orch.lonlat(p[0], p[1])]  # noqa: E731
    return {"area": [ll(p) for p in info["area_en"]],
            "map_image": {"url": "api/flood_overview.jpg", "live_url": "api/flood_overview_live.jpg",
                          "coordinates": [ll(p) for p in info["overview_en"]]}}


def make_app(args, orch: Orchestrator) -> FastAPI:
    cfg = orch.cfg
    bridge = (getattr(args, "bridge", "") or "").rstrip("/")

    @asynccontextmanager
    async def lifespan(app):
        await orch.start()
        if orch.sensors is not None:
            orch.sensors.start()
        yield
        await orch.shutdown()

    app = FastAPI(lifespan=lifespan)

    @app.get("/api/config")
    def config():
        side = cfg.grid * cfg.cell_m
        return {
            "home": list(cfg.home), "cell_m": cfg.cell_m, "max_grid": cfg.grid,
            "max_area_km2": round(side * side / 1e6, 2), "modes": list(MODES),
            "drones": [{"id": d.id, "name": d.name, "role": d.role} for d in orch.drones],
            "backend": args.backend, "detector": args.detector,
            "bridge": bool(bridge),
            "scenarios": list(getattr(orch, "scenarios", ("wilderness", "flood"))),
            **flood_extras(orch),
        }

    overview: dict = {}

    @app.get("/api/flood_overview.jpg")
    async def flood_overview():
        # the first frame with the drone still at its start is good enough as a map background
        if "jpg" not in overview:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{bridge}/overview.jpg")
            if r.status_code != 200:
                return JSONResponse({"error": "no overview frame"}, status_code=503)
            overview["jpg"] = r.content
        return Response(overview["jpg"], media_type="image/jpeg")

    @app.get("/api/flood_overview_live.jpg")
    async def flood_overview_live():
        """The current top-down frame: the map's Overview mode refreshes it about once a second."""
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{bridge}/overview.jpg")
        if r.status_code != 200:
            return JSONResponse({"error": "no overview frame"}, status_code=503)
        return Response(r.content, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    # The mission log clock (the `t` of every log row); scripts/replay_record.py
    # stamps each recorded video frame with it.
    @app.get("/api/clock")
    def clock():
        return {"t": round(orch.now(), 3)}

    @app.get("/api/snapshot/{sid}.jpg")
    def snapshot(sid: int):
        s = orch.survivors.get(sid)
        if s is None or s.snapshot is None:
            return JSONResponse({"error": "no snapshot"}, status_code=404)
        return Response(s.snapshot, media_type="image/jpeg")

    @app.get("/api/metrics")
    def metrics():
        return {"expected_found": orch.expected_found(), **orch.metrics,
                "survivors": [vars(s) | {"snapshot": None} for s in orch.survivors.values()]}

    # ----------------------------------------------------------------- replay
    # Side-by-side replay of two methods on the same scenario and seed
    # (plan section 11, step 7). See docs/PROTOCOL.md section 6.
    # The mission frame origin places the finds of old logs on the map.
    replay.HOME_ORIGIN = tuple(cfg.home)

    @app.get("/api/replays")
    def replays():
        return {"replays": replay.list_replays(ROOT)}

    @app.get("/api/replay/{replay_id}")
    def replay_detail(replay_id: str):
        data = replay.get_replay(replay_id, ROOT)
        if data is None:
            return JSONResponse({"error": f"no replay {replay_id}"}, status_code=404)
        return data

    # Recorded videos of packed replays (scripts/replay_pack.py). FileResponse
    # answers Range requests, so the browser can seek.
    @app.get("/api/replay_media/{folder}/{name}")
    def replay_media(folder: str, name: str):
        p = replay.media_path(folder, name, ROOT)
        if p is None:
            return JSONResponse({"error": "no such replay video"}, status_code=404)
        return FileResponse(p, media_type="video/mp4")
    # ------------------------------------------------------------------ video
    # The two 3D views and the drone camera panel (plan 11.1). Without a bridge they
    # answer 404 immediately so the UI shows its placeholder instead of retrying forever.
    async def _stream(path: str, boxes: int | None = None) -> Response:
        if not bridge:
            return JSONResponse({"error": "no bridge configured"}, status_code=404)
        if boxes is None:
            gen = frames(f"{bridge}/stream/{path}")
        else:
            gen = frames_with_boxes(bridge, boxes, orch)
        return StreamingResponse(gen, headers=stream_headers())

    @app.get("/video/follow.mjpg")
    async def video_follow(request: Request):
        return await _stream("follow")

    @app.get("/video/overview.mjpg")
    async def video_overview(request: Request):
        return await _stream("overview")

    @app.get("/video/camera/{drone}.mjpg")
    async def video_camera(drone: int, request: Request):
        if not any(d.id == drone for d in orch.drones):
            return JSONResponse({"error": f"no drone {drone}"}, status_code=404)
        return await _stream(f"cam{drone}", boxes=drone)

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        q: asyncio.Queue = asyncio.Queue()
        orch.subscribers.append(q)

        async def sender():
            await sock.send_text(json.dumps(orch.state_msg(truth=args.show_truth)))
            if orch.mission:
                await sock.send_text(json.dumps(orch.belief_msg()))
                await sock.send_text(json.dumps(orch.prior_msg()))
            for ev in orch.events[-50:]:
                await sock.send_text(json.dumps(ev))
            while True:
                msg = await q.get()
                if msg["type"] == "state" and args.show_truth:
                    msg = orch.state_msg(truth=True)
                await sock.send_text(json.dumps(msg))

        task = asyncio.create_task(sender())
        try:
            while True:
                msg = json.loads(await sock.receive_text())
                try:
                    await orch.command(msg)
                except Exception as exc:
                    await sock.send_text(json.dumps({"type": "event", "t": round(orch.t_mission(), 1),
                                                     "level": "error", "text": f"{msg.get('type')}: {exc}"}))
                else:
                    await _after_command(msg)
        except WebSocketDisconnect:
            pass
        finally:
            task.cancel()
            orch.subscribers.remove(q)

    async def _after_command(msg: dict) -> None:
        """Side effects the UI expects from a command but the orchestrator does not own."""
        if msg.get("type") != "select":
            return
        if not bridge:
            return
        drone = msg.get("drone")
        if drone is None and msg.get("survivor") is not None:
            selector = getattr(orch, "select", None)
            with contextlib.suppress(Exception):
                drone = selector() if callable(selector) else None
        if drone is None:
            return
        import httpx
        with contextlib.suppress(Exception):
            async with httpx.AsyncClient(timeout=3.0) as client:
                await set_view(bridge, int(drone), client)

    dist = ROOT / "ui" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="ui")
    return app


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["kinematic", "flooduav"], default="kinematic")
    ap.add_argument("--roles", default="search,search,rescue")
    ap.add_argument("--speed", type=float, default=1.0, help="sim speed factor (kinematic)")
    ap.add_argument("--ckpt", default=str(ROOT / "models/ft_best/seed1"))
    ap.add_argument("--detector", choices=["sim"], default="sim")
    ap.add_argument("--pd", type=float, default=0.8)
    ap.add_argument("--fp", type=float, default=0.01)
    ap.add_argument("--truth-from-prior", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--poisson", action="store_true")
    ap.add_argument("--show-truth", action="store_true")
    ap.add_argument("--bridge", default="",
                    help="sim bridge base URL for the video endpoints (plan 11.1); "
                         "empty disables /video/*, which then answer 404")
    ap.add_argument("--flood-layers", default=str(ROOT / "data/regions/demo_flood.npz"),
                    help="flood town layer npz for the flood prior (plan 6.4); empty: flood uses the Koester rings")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--home", default="", help="launch site as lon,lat; it must be the world origin")
    ap.add_argument("--log", default=str(ROOT / "runs/mission/mission.jsonl"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    return ap


def main():
    args = parser().parse_args()
    orch = build(args)
    uvicorn.run(make_app(args, orch), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
