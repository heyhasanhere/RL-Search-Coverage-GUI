# Orchestrator - UI Protocol

Scope: the JSON messages between the mission orchestrator (`sar/mission/server.py`) and the web UI (`ui/`).
Coordinates are always `[lon, lat]` (GeoJSON order). Times are seconds of mission time.

## 1. Transport

| Item | Value |
|---|---|
| WebSocket | `ws://<host>:8000/ws` |
| Config | `GET /api/config` |
| Camera snapshot | `GET /api/snapshot/<survivor_id>.jpg` (404 if none) |
| Metrics | `GET /api/metrics` |
| Replay list | `GET /api/replays` |
| Replay detail | `GET /api/replay/<id>` |
| 3D follow camera | `GET /video/follow.mjpg` |
| 3D overview camera | `GET /video/overview.mjpg` |
| Drone camera with detector boxes | `GET /video/camera/<drone>.mjpg` |
| Static UI | `GET /` serves `ui/dist` when it exists |

Every message is one JSON object with a `type` field.

The video endpoints proxy the sim bridge (`sim/flooduav/bridge.py`). They are enabled by the
server argument `--bridge <url>`; without it all three answer 404 with
`{"error": "no bridge configured"}` and the UI shows a placeholder. The server also posts
`{"drone": i}` to `<bridge>/view` when the operator selects a drone.
`GET /api/flood_overview.jpg` is the FloodUAV top-down map image under the map panel. It is
rendered once by Unity at high resolution (`<bridge>/map.jpg`, 4096 px by default).

## 2. `GET /api/config`

```json
{
  "home": [149.58, -33.42],
  "cell_m": 30.0,
  "max_grid": 96,
  "max_area_km2": 8.29,
  "modes": ["rl", "planner", "greedy", "lawnmower"],
  "drones": [{"id": 0, "name": "S1", "role": "search"}, {"id": 1, "name": "S2", "role": "search"}, {"id": 2, "name": "R1", "role": "rescue"}],
  "backend": "kinematic | flooduav",
  "detector": "sim",
  "bridge": false,
  "survivor_layouts": false,
  "scenarios": ["wilderness", "flood"]
}
```

`bridge` tells the UI whether the video panels can stream. `modes` is the full list of
supported search modes; the UI shows exactly this list.

## 3. Server to Client

### 3.1 `state` (about 5 Hz)

```json
{
  "type": "state",
  "t": 123.4,
  "speed": 1.0,
  "mission": {
    "status": "idle | running | finished",
    "stopping": false,
    "mode": "rl",
    "polygon": [[lon, lat], ...],
    "lkp": [lon, lat] | null,
    "expected_total": 3.0,
    "remaining_fraction": 0.62,
    "expected_found": 1.14,
    "covered_fraction": 0.40
  },
  "drones": [{
    "id": 0, "name": "S1", "role": "search | rescue",
    "state": "IDLE | STANDBY | SEARCH | RESCUE_TRANSIT | ON_SURVIVOR | RETURNING | ABORT | LANDED",
    "pos": [lon, lat], "alt": 25.0, "heading": 90.0,
    "battery_s": 1320.0, "battery_frac": 0.88,
    "segment": [[lon, lat], ...],
    "target": [lon, lat] | null,
    "standby_point": [lon, lat] | null,
    "task": "text",
    "level": "low | high | transit | null",
    "alt_agl": 25.0,
    "sortie": 1,
    "swap_remaining": null
  }],
  "survivors": [{
    "id": 4, "pos": [lon, lat],
    "status": "candidate | confirmed | rejected | assigned | reached",
    "score": 0.87, "views": 2, "has_snapshot": true,
    "assigned": 2 | null, "t_detected": 88.2, "t_confirmed": 97.0 | null, "t_reached": null
  }],
  "options": {"auto_dispatch": true, "auto_replace": false, "auto_stage": false},
  "truth": [[lon, lat], ...] | null
}
```

`truth` is the simulated ground truth; it is sent only when the server runs with `--show-truth`.

The four extra drone fields are optional and older clients ignore them.

| Field | Meaning |
|---|---|
| `level` | Search altitude level the agent chose: `low`, `high`, or `transit` for return and rescue legs. Null while the drone is on the ground. |
| `alt_agl` | Height above the ground or canopy under the drone, in meters. The search levels are 23 m and 69 m. |
| `sortie` | Battery sortie number, 1 on the first flight. It goes up after each battery swap. |
| `swap_remaining` | Seconds left of a battery swap, or null when no swap is running. The UI shows a countdown and the drone state stays `LANDED`. |

### 3.2 `belief` (about 1 Hz while a mission runs)

```json
{
  "type": "belief",
  "corners": [[lon, lat], [lon, lat], [lon, lat], [lon, lat]],
  "png": "<base64 RGBA PNG>",
  "coverage_png": "<base64 RGBA PNG>"
}
```

`corners` are top-left, top-right, bottom-right, bottom-left, the order a MapLibre image source expects.
The belief PNG is a heatmap (transparent outside the area); the coverage PNG marks looked-at cells. Every airborne drone at search height or above looks at the cell below it, also on rescue, return and staging flights, and a look there counts as a search look.

### 3.3 `prior` (once per mission)

```json
{
  "type": "prior",
  "corners": [[lon, lat], [lon, lat], [lon, lat], [lon, lat]],
  "nx": 64,
  "ny": 64,
  "cell_m": 30.0,
  "scenario": "wilderness",
  "blend": {"key": "blend", "label": "Blend", "png": "<base64 RGBA PNG>"},
  "layers": [
    {"key": "koester", "label": "Koester", "png": "<base64 RGBA PNG>"},
    {"key": "walker", "label": "Walker", "png": "<base64 RGBA PNG>"}
  ]
}
```

| Field | Meaning |
|---|---|
| `corners` | Same order and meaning as in `belief`. Every PNG covers this rectangle. |
| `nx`, `ny` | Grid size in cells. Each PNG is `nx` by `ny` pixels, one pixel per cell. |
| `cell_m` | Cell size in meters. |
| `scenario` | `wilderness` or `flood`. The status bar shows it. |
| `blend` | The weighted mix of all components. Always present. |
| `layers` | One entry per prior component. Empty is allowed when the server has no fitted prior yet. |

Each `png` is a base64 RGBA PNG, transparent where the prior is zero, colormapped so that
brighter means more likely. The UI adds one checkbox per entry plus one for `blend`, and
overlays only the checked layers on the map. Keys are stable across missions so a checkbox
keeps its state.

The server sends `prior` once when the mission is built, right after the first `state`.
Later missions send it again.

### 3.4 `event`

```json
{"type": "event", "t": 12.0, "level": "info | warn | error", "text": "S1 candidate at ..."}
```

## 4. Client to Server

| `type` | Fields | Effect |
|---|---|---|
| `start_mission` | `polygon` ([lon, lat] ring), `lkp` (optional), `mode`, `n_search` (1-3), `expected_survivors`, `options`, `scenario`, `lkp_time_h`, `water_level_m`, `regions` | Builds the grid and prior, launches the search drones |
| `stop_mission` | - | All drones return home |
| `continue_mission` | - | Resumes a stopped or finished mission: belief, coverage and survivors are kept, the search drones take off again. Batteries are not recharged |
| `restart_mission` | - | Resets the world and starts the last mission again |
| `reset_mission` | - | Resets the world to before the start and does not start: drones on their pads, batteries full, survivors cleared, survivor numbers start again at 1 |
| `set_mode` | `mode` | Changes the search mode mid-mission |
| `new_survivors` | `count` (1-100, optional, default the build's 15), `mix` (optional) | FloodUAV only (`survivor_layouts` in the config): `reset_mission`, then `count` survivors at new random positions in Unity. Later resets keep them. `mix` is the survivor terrain mix: weights 0 or more for `roofs`, `high_ground`, `bridges`, `water`, `uniform` (missing keys are 0, not all 0), normalized; without it the default 50 / 25 / 10 / 10 / 5. The flood prior of later missions uses the same mix. An event names the count, the layout seed and the mix |
| `recall` | `drone` | Drone returns home (RETURNING) |
| `standby` | `drone`, `point` ([lon, lat], optional, default home) | Drone parks at the point (STANDBY) |
| `launch` | `drone`, `task` (`search` or `rescue`), `survivor` (for rescue, optional) | Launches a standby or idle drone |
| `dispatch` | `drone`, `survivor` | Manual rescue dispatch, overrides automatic dispatch |
| `confirm` | `survivor` | Operator confirms a candidate |
| `reject` | `survivor` | Operator rejects a candidate |
| `set_options` | any of `auto_dispatch`, `auto_replace`, `auto_stage` | Updates options |
| `set_speed` | `speed` | Kinematic backend only: simulation speed factor |
| `select` | `drone` or `survivor` | Points the follow camera and the drone camera panel |
| `camera` | `view: "follow"` with `distance` (m), `yaw`, `pitch` (degrees), or `view: "drone"` with `drone`, `fov`, `pan`, `tilt` (degrees) | Zoom and angle of the 3D follow camera or of a drone camera. FloodUAV only; the server posts it to `<bridge>/camera` |

### 4.1 Scenario inputs

`start_mission` takes four extra fields. All of them are optional, and a server that does
not know them ignores them, so an old orchestrator still runs an old UI.

| Field | Type | Used for | Default |
|---|---|---|---|
| `scenario` | `wilderness` or `flood` | Chooses the prior, the world and the survivor model | `wilderness` |
| `lkp_time_h` | hours, float | Wilderness only. Sets the walker time budget, so the map spreads as time passes. | 6 |
| `water_level_m` | meters, float | Flood only. Drives the water surface and the flood prior, so the two always match. | 0 |
| `regions` | list of `{polygon, weight}` | Operator priority regions, as search teams draw them. The weight is the share the prior blend gives them. | `[]` |

```json
{
  "type": "start_mission",
  "polygon": [[149.57, -33.43], [149.59, -33.43], [149.59, -33.41], [149.57, -33.41]],
  "lkp": [149.58, -33.42],
  "mode": "rl_search",
  "n_search": 3,
  "expected_survivors": 3,
  "options": {"auto_dispatch": true, "auto_replace": false, "auto_stage": false},
  "scenario": "wilderness",
  "lkp_time_h": 6,
  "regions": [{"polygon": [[149.575, -33.425], [149.585, -33.425], [149.585, -33.415], [149.575, -33.415]], "weight": 0.3}]
}
```

`weight` is a share between 0 and 1. The UI default is 0.3.

### 4.2 `select`

```json
{"type": "select", "drone": 1}
{"type": "select", "survivor": 4}
```

With `drone`, the follow camera tracks that drone and the drone camera panel shows it.
With `survivor`, the follow camera moves to the survivor and the camera panel follows the
nearest drone to it. The server tells the sim bridge through `POST <bridge>/view`.

The server answers an invalid command with an `event` of level `error`.

## 5. Replay

The split replay compares two methods on the same scenario and seed.
It reads the mission logs the orchestrator writes as JSON lines under `runs/mission/`
and the packed recordings under `runs/replays/`.

### 5.1 `GET /api/replays`

The list of missions that can be replayed.

```json
{
  "replays": [{
    "id": "demo_mcts",
    "path": "runs/mission/demo_mcts.jsonl",
    "scenario": "wilderness",
    "mode": "mcts",
    "seed": 1000,
    "started": 141.28,
    "duration": 1858.93,
    "grid": 64,
    "expected": 3.0,
    "n_frames": 1186,
    "n_finds": 2
  }]
}
```

| Field | Meaning |
|---|---|
| `id` | The id to pass to `GET /api/replay/<id>`. A log file is appended to, so it can hold several missions; the second one in the same file gets the id `<file id>#2`. |
| `path` | The log file, relative to the project root. |
| `scenario` | `wilderness` or `flood`, when the log says so. |
| `mode` | The search mode the mission ran with. |
| `seed` | The seed the mission ran with. |
| `started` | Wall-clock time the mission started, in seconds. |
| `duration` | Mission duration in seconds, from the first to the last track. |
| `grid` | Grid side in cells. |
| `expected` | Expected survivors the mission was given. |
| `n_frames` | Number of track frames in the log. |
| `n_finds` | Number of survivor detections in the log. |

Missions with nothing to show (no tracks, finds or belief) are left out.

### 5.2 `GET /api/replay/<id>`

One mission as a time series the UI can scrub.

```json
{
  "meta": {"id": "demo_mcts", "path": "runs/mission/demo_mcts.jsonl", "scenario": "wilderness",
           "mode": "mcts", "seed": 1000, "started": 141.28, "duration": 1858.93,
           "grid": 64, "expected": 3.0, "n_frames": 1186, "n_finds": 2,
           "polygon": [[149.57, -33.43], [149.59, -33.43], [149.59, -33.41], [149.57, -33.41]]},
  "frames": [{
    "t": 0.0,
    "found": 0.0,
    "drones": [{"id": 0, "name": "S1", "state": "SEARCH", "pos": [149.58, -33.42],
                "alt": 23.0, "heading": 90.0}]
  }],
  "belief": [{"t": 0.0, "corners": [[149.57, -33.43], [149.59, -33.43], [149.59, -33.41], [149.57, -33.41]],
              "png": "<base64 RGBA PNG>", "coverage_png": "<base64 RGBA PNG>"}],
  "finds": [{"t": 68.8, "sid": 0, "pos": [149.5805, -33.4205], "status": "candidate",
             "drone": 0, "true": true}],
  "events": [{"t": 12.0, "level": "info", "text": "S1 candidate at ..."}],
  "curve": [[0.0, 0.0], [1.05, 0.0], [2.09, 0.0]]
}
```

| Field | Meaning |
|---|---|
| `meta` | The same fields as one row of `GET /api/replays`, plus the search area `polygon` in `[lon, lat]`. |
| `frames` | The drone positions and the found-expected fraction, sampled about once a second. Times are seconds since the first frame. `found` is the fraction of the expected survivors found, so the two runs of a comparison are on the same scale. |
| `belief` | The belief and coverage heat maps, subsampled to at most 60 snapshots per mission. Same PNG format and corner order as the `belief` message. |
| `finds` | The survivor detections: when they happened, where, and their status (`candidate`, `confirmed` or `reached`). `true` says whether the detection was a real survivor, when the log knows. |
| `events` | The event log, with mission times in seconds. |
| `curve` | The found-expected curve as `[[t, value], ...]`. This is the line the UI draws in the replay chart. |

The server answers an unknown id with a 404 and `{"error": "no replay <id>"}`.
