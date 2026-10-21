# Operator UI

React + MapLibre GL + Terra Draw front end for the mission orchestrator. Protocol: `../docs/PROTOCOL.md`.

## Commands

```bash
npm install
npm run dev     # Vite on :5173, proxies /api and /ws to http://localhost:8000
npm run build   # type-check and build to ui/dist (relative asset paths, served by the Python server at /)
npm run mock    # mock orchestrator on :8000 (fake drones, belief and prior PNGs, synthetic MJPEG); also serves ui/dist when built
```

Without Python: run `npm run mock` and `npm run dev`, then open http://localhost:5173.

With Python:

```bash
cd .. && uv run python -m sar.mission.server --backend kinematic --speed 5
```

The FloodUAV backend (`--backend flooduav --bridge http://<host>:8101`) gives the three video panels.
Without it they show a placeholder.

## Layout

| Panel | Position | Content |
|---|---|---|
| Status bar | top | Scenario, mission, mode, time, found, candidates, covered, remaining |
| 3D view | large, top left | Follow camera on the selected drone, with a Follow / Overview toggle |
| Drone camera | top right, upper | Downward camera of the selected drone, with detector boxes drawn server side |
| Map and grid | top right, lower | Area, belief, coverage, routes, altitude level per drone, prior layer toggles |
| Mission and drones | bottom left | Scenario inputs and per-drone state, level, AGL, battery, sortie, recall, standby |
| Survivors and events | bottom right | Candidates with snapshot, confirm and reject, confirmed and rescued, event log |

## Using it

- Map tools (top left): `Area` draws the search polygon, `LKP` places the last-known point,
  `Region` draws an operator priority region, `Edit` moves vertices, `Clear` removes everything.
- The region weight slider (bottom left) sets the weight of newly drawn regions. Default 0.3.
- Draw several regions to mark more than one priority area.
- Start is disabled until the area is within `max_area_km2` and the bounding-box side is
  within `max_grid * cell_m`. Wilderness also needs a last-seen point.
- Clicking a drone in the list or on the map sends `{"type":"select","drone":i}` and switches
  both video panels to it. Clicking a survivor sends `{"type":"select","survivor":id}`.
- Drone `Standby`: click the map to choose the point, or `At home`.

## Screenshots

```bash
node scripts/shot_running.cjs http://localhost:8000/ /tmp/ui.png 30
```

Draws an area, an LKP and a region, starts a mission, waits 30 s of mission time and
screenshots. Needs `npm install` and a running server (mock or Python).