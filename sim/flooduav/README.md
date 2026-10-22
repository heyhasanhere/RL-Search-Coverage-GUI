# FloodUAV in the Operator UI

The FloodUAV Unity town (`unity/SearchPickupDefault`, Linux build) runs as a mission backend.
The operator UI (`ui/`) shows its views and drones.
The Unity build is used only for its scene and visuals. Its rewards and agent are not used.

## Run (on the Linux GPU machine)

```bash
sim/flooduav/run.sh up      # Xvfb :97, bridge :8101, UI and mission server :8001
sim/flooduav/run.sh down
```

Open `http://<host>:8001`. The search area is preset to the Unity boundary, and the map shows only a top-down render of the town (no real-world tiles).
Press Start (scenario `flood`). Logs are in `runs/flooduav_ui/`, and the mission log is `runs/mission/flooduav_ui.jsonl`.
Run `down` in its own SSH call: its `pkill` patterns can match a command line that contains them.

## Parts

| Part | File | Role |
|---|---|---|
| Patch | `patch/FloodPatch/Hook.cs`, `patch/Injector/`, `patch/build.sh` | Hard-linked copy of the build at `unity/SearchPickupPatched` with hooks injected (Mono.Cecil) |
| Fleet | `patch/FloodPatch/Fleet.cs` | Takes over the build's pickup drone (pilot and coordinator off) and flies it from a side channel |
| Bridge | `bridge.py` (`.venv-fly`) | Steps Unity, flies both drones to setpoints, lands them, streams the screen regions as MJPEG |
| Backend | `sar/mission/flooduav.py`, `server.py --backend flooduav` | Two drones (D1 search drone, D2 pickup drone); Unity's finds and D2's camera footprint become detections |

The patch reads environment variables at mission start:

- It raises the speed. The airframe's 100 m2 wing area caps hover-mode speed near 2.5 m/s. The bridge sets it to 1 m2 and 15 m/s.
- It removes the time limit and sets `runInBackground`.
- It splits the screen (2640x720): follow view (left), D1 and D2 cameras (middle, straight down) and a square overview (right).
- It turns contact failures off (`FLOODUAV_SOFT`) so drones can land, and sets the follow camera offset (`FLOODUAV_FOLLOW`, default 0,30,-30: about 45 degrees down).
- It writes the boundary, start point and survivor positions to a JSON file.

Unity renders with Vulkan on the NVIDIA GPU (`-force-vulkan`); OpenGL under Xvfb falls back to llvmpipe at about 3 fps.
The mission server runs JAX on GPU 1 (`FLOODUAV_GPU`) without preallocation, so GPU 0 stays free for Unity.

## Behavior and limits

- Mission frame: east = boundary +x, north = +z, origin at the drone start. The lon/lat are made up (`MissionConfig.home`).
- Heights: the drone never flies below 40 m above the start (bridges at about 12 m end the mission). The vehicle maps the 23 m search height to 40 m.
- Landing is real: the drone descends straight down and keeps pushing down on the ground. Over open water (no ground within 150 m) it hovers instead.
- Unity's visibility sensor confirms survivors up to 150 m away. The detector reports each found survivor near the drone, so the re-look confirms it.
- If Unity ends its mission (collision or leaving the boundary), the drone resets to the start and the UI logs a warning.
- Two drones under our control, 15 fixed survivors. D2 has no sensor in the build: it detects survivors inside its camera footprint.
- Restart (`POST /reset`) puts both drones back on their pads and clears Unity's finds. Speed cap: `POST /speed`. Follow target: `POST /view`.
- The build's mission map goes to a second display, which Xvfb does not have.
- Speed: `--rate 2` (sim seconds per wall second) with Unity time scale 2. A time scale above the rate drops the video to a few fps.
- Replay of a FloodUAV mission draws the tracks on a plain map: the log does not keep the Unity top-down image or the search area outline.
