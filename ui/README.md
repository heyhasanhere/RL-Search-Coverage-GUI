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
| Video panes | left, two stacked | Each pane shows the 3D follow view or the downward camera of one drone. Both start on 3D follow, the top pane on D1 and the bottom pane on D2. The 3D Follow / Drone Camera toggle and the drone tabs change each pane. Video is H.264 decoded by the browser (WebCodecs), or MJPEG where the browser cannot decode it, and stops while the tab is hidden |
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
- Clicking a drone in the list or on the map sends `{"type":"select","drone":i}`.
  Clicking a survivor sends `{"type":"select","survivor":id}`.
- Drone `Standby`: click the map to choose the point, or `At home`.
- 3D follow and drone camera (FloodUAV): scroll to zoom, drag to turn, double-click to reset.
- `Restart` resets the world and starts the same mission again. `Reset` resets the world and waits for Start.
- Survivor terrain mix (FloodUAV): percent of new survivors on roofs, high ground, bridges, water and anywhere
  (default 50 / 25 / 10 / 10 / 5). `New survivors` places them by it, and the search prior uses the same mix.
- Mission options (Mission panel): `Auto switch to rescue` lets a search drone fly a rescue when that is faster than waiting for a rescue drone, then search again.
  It does nothing in `rl_unity` mode.
  The rule is in `docs/PROTOCOL.md` section 4.3.
  `Re-check detections` makes a search drone look again from beside a detection before it is confirmed (hidden in `rl_unity`).
  Defaults: Expected survivors 15; auto switch, auto dispatch, auto replace and auto stage on; re-check off.
- Drone speed: the cap starts at 15 m/s, or 3 m/s in `rl_unity`. Changing the mode before Start resets it to that mode's default.
  In `rl_unity` it applies to the agent only when `Training flight limits` is unticked.
- `Settings` (top bar): show or hide the true survivor positions on the map.

## Screenshots

```bash
node scripts/shot_running.cjs http://localhost:8000/ /tmp/ui.png 30
```

Draws an area, an LKP and a region, starts a mission, waits 30 s of mission time and
screenshots. Needs `npm install` and a running server (mock or Python).