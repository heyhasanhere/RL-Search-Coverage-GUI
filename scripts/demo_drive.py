"""Drive the mission server over the WebSocket: start a mission, confirm detections,
dispatch the rescue drone and print the numbers (plan 21, step 3).

  uv run --extra mission python scripts/demo_drive.py --mode rl --seed 1000 \
      --url ws://127.0.0.1:8000/ws --t-max 900

The operator loop is the point of the demo: a detection stops a drone, the operator
confirms the candidate and the rescue drone is dispatched. This script does what a
person does in the UI, so the same code path is exercised.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import numpy as np
import websockets

M_PER_DEG = 111_320.0


def near_truth(pos, truth, radius_m):
    """True if a true victim (lon, lat) lies within radius_m of pos; None without truth."""
    if not truth or pos is None:
        return None
    lon, lat = pos
    c = np.cos(np.radians(lat))
    return any(np.hypot((t[0] - lon) * M_PER_DEG * c, (t[1] - lat) * M_PER_DEG) <= radius_m for t in truth)


def poly_around(home, half_deg=0.006):
    """A square search area around the launch site (lon, lat)."""
    lon, lat = home
    return [[lon - half_deg, lat - half_deg], [lon + half_deg, lat - half_deg],
            [lon + half_deg, lat + half_deg], [lon - half_deg, lat + half_deg]]


async def main_async(args):
    import httpx
    cfg = (await httpx.AsyncClient(timeout=10.0).get(f"{args.http}/api/config")).json()
    home = cfg["home"]
    async with websockets.connect(args.url, max_size=16 * 1024 * 1024,
                                  open_timeout=60, ping_interval=None) as ws:
        state = {"mission": None, "survivors": [], "events": []}

        async def reader():
            try:
                while True:
                    msg = json.loads(await ws.recv())
                    t = msg.get("type")
                    if t == "state":
                        state["mission"] = msg.get("mission")
                        state["drones"] = msg.get("drones")
                        state["survivors"] = msg.get("survivors", [])
                        state["t"] = msg.get("t")
                        state["truth"] = msg.get("truth")
                    elif t == "event":
                        state["events"].append(msg)
                        print(f"[{msg.get('t')}] {msg.get('level')}: {msg.get('text')}", flush=True)
                    elif t == "error":
                        print(f"ERROR {msg}", flush=True)
            except Exception as exc:
                print(f"reader stopped: {exc!r}", flush=True)

        task = asyncio.create_task(reader())
        await asyncio.sleep(1.0)

        # a mission left over from an earlier run would make start_mission fail
        if (state.get("mission") or {}).get("status") == "running":
            print("stopping the mission that is still running", flush=True)
            await ws.send(json.dumps({"type": "stop_mission"}))
            await asyncio.sleep(8.0)

        start = {"type": "start_mission", "polygon": poly_around(home), "lkp": home,
                 "mode": args.mode, "n_search": 3, "expected_survivors": args.expected,
                 "scenario": "wilderness", "lkp_time_h": 6.0,
                 "options": {"auto_dispatch": False, "auto_replace": True, "auto_stage": False}}
        print("start mission:", json.dumps(start), flush=True)
        await ws.send(json.dumps(start))

        rescue = [d["id"] for d in cfg["drones"] if d["role"] == "rescue"]
        t0, confirmed, dispatched, last = time.time(), set(), set(), []
        last_print = 0.0
        started = False
        t_start = None
        while time.time() - t0 < args.t_max:
            await asyncio.sleep(1.0)
            m = state.get("mission")
            if m is None:
                if time.time() - last_print > 30:
                    last_print = time.time()
                    print("... waiting for the mission to start", flush=True)
                continue
            last = m
            if m.get("status") == "running":
                if not started:
                    started = True
                    t_start = state.get("t")
            for s in state.get("survivors", []):
                sid = s["id"]
                if s.get("status") == "candidate" and sid not in confirmed:
                    confirmed.add(sid)
                    # the operator checks the camera crop; with --show-truth the server
                    # sends the true victims, which stand in for that check
                    ok = near_truth(s.get("pos"), state.get("truth"), args.check_m)
                    if ok is False and args.operator_check:
                        await ws.send(json.dumps({"type": "reject", "survivor": sid}))
                        print(f"operator rejected candidate #{sid} at {s.get('pos')} (no person in view)", flush=True)
                        continue
                    await ws.send(json.dumps({"type": "confirm", "survivor": sid}))
                    print(f"operator confirmed candidate #{sid} at {s.get('pos')}", flush=True)
                elif s.get("status") == "confirmed" and sid not in dispatched and rescue:
                    dispatched.add(sid)
                    await ws.send(json.dumps({"type": "dispatch", "drone": rescue[0], "survivor": sid}))
                    print(f"rescue dispatch for #{sid}: drone {rescue[0]}", flush=True)
            if time.time() - last_print > 30:
                last_print = time.time()
                print(f"t={m.get('expected_found')} found {m.get('expected_found')}/3, "
                      f"drones {[(d['id'], d['state']) for d in state.get('drones', [])]}", flush=True)
            # the mission ends when it stops running, but only after it has started:
            # the first state message still shows the idle mission
            if started and m.get("status") != "running":
                break
            # --t-mission gives every mode the same flight time, the sim clock is slower
            # than the wall clock
            if args.t_mission > 0 and t_start is not None and state.get("t") is not None:
                if state["t"] - t_start >= args.t_mission:
                    print(f"mission time {args.t_mission} s reached", flush=True)
                    break
        if last and last.get("status") == "running":
            await ws.send(json.dumps({"type": "stop_mission"}))
            await asyncio.sleep(5.0)
        task.cancel()
    print(json.dumps({"mode": args.mode,
                      "status": last.get("status") if last else None,
                      "found_expected": last.get("expected_found") if last else None,
                      "covered_fraction": last.get("covered_fraction") if last else None,
                      "t_end": state.get("t"),
                      "confirmed": sorted(confirmed), "dispatched": sorted(dispatched),
                      "survivors": state.get("survivors", [])},
                     indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    ap.add_argument("--http", default="http://127.0.0.1:8000")
    ap.add_argument("--mode", default="rl")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--expected", type=float, default=3.0)
    ap.add_argument("--operator-check", action=argparse.BooleanOptionalAction, default=True,
                    help="reject candidates with no true victim within --check-m (needs --show-truth)")
    ap.add_argument("--check-m", type=float, default=30.0)
    ap.add_argument("--t-max", type=float, default=900.0)
    ap.add_argument("--t-mission", type=float, default=0.0,
                    help="stop after this many seconds of mission time (the sim runs "
                         "slower than real time, so --t-max alone would give the modes "
                         "different flight times)")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
