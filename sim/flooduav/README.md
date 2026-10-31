# FloodUAV in the Operator UI

The FloodUAV Unity town runs as a mission backend: the Linux build (`unity/SearchPickupDefault`) or the macOS build (`unity/FloodUAV.app`).
The operator UI (`ui/`) shows its views and drones.
The Unity build is used only for its scene and visuals. Its rewards and agent are not used.

## Run (on the Linux GPU machine or a Mac)

```bash
sim/flooduav/run.sh up      # bridge :8101, UI and mission server :8001 (Linux: on Xvfb :97)
sim/flooduav/run.sh down
```

Mac setup: `docs/SETUP.md` section 7.

Open `http://<host>:8001` (`http://localhost:8001` on a Mac). The search area is preset to the Unity boundary, and the map shows only a top-down render of the town (no real-world tiles).
Press Start (scenario `flood`). Logs are in `runs/flooduav_ui/`, and the mission log is `runs/mission/flooduav_ui.jsonl`.
Over SSH, run `down` in its own SSH call: its `pkill` patterns can match a command line that contains them.

Operator password: put `SAR_CONTROL_PASSWORD=...` in `~/.config/sar-search/control.env` (or the file named by `SAR_ENV_FILE`) before `up`.
With it set, anyone can watch and only a browser logged in with the password (status bar, Log in) can send commands.

## Parts

| Part | File | Role |
|---|---|---|
| Patch | `patch/FloodPatch/Hook.cs`, `Layout.cs`, `patch/Injector/`, `patch/build.sh` | Copy of the build with hooks injected (Mono.Cecil): `unity/SearchPickupPatched` (Linux, hard links) or `unity/FloodUAVPatched.app` (macOS, clones, signed ad hoc) |
| Capture | `patch/FloodPatch/Capture.cs` | macOS: the split views render into a 3920x720 texture instead of the screen, and its frames go to a file the bridge reads (`--capture file`) |
| Fleet | `patch/FloodPatch/Fleet.cs` | Takes over the build's pickup drone (pilot and coordinator off) and flies it from a side channel |
| Bridge | `bridge.py` (`.venv-fly`) | Steps Unity, flies both drones to setpoints, lands them, streams the screen regions (or the captured frames) as MJPEG and H.264 |
| Trained agent | `models/unity_survivor_search/survivor_search_ppo_19m.onnx` | The build's own search agent (ML-Agents PPO, `train_config.yaml`); the bridge can fly D1 with it |
| More agents | anywhere under `models/` | Drop-in: ML-Agents ONNX files and Stable-Baselines3 folders (DQN, TRPO) on the same observation, found automatically |
| Backend | `sar/mission/flooduav.py`, `server.py --backend flooduav` | Two drones (D1 search drone, D2 pickup drone); survivors inside each drone's detection box become detections |

The patch reads environment variables at mission start:

- It raises the speed. The airframe's 100 m2 wing area caps hover-mode speed near 2.5 m/s. The bridge sets it to 1 m2 and 15 m/s.
- It removes the time limit and sets `runInBackground`.
- It splits the screen (3920x720): D1 follow view (left), D1 and D2 cameras (straight down), a square overview, and D2 follow view (right).
- It turns contact failures off (`FLOODUAV_SOFT`) so drones can land, and sets the follow camera offset (`FLOODUAV_FOLLOW`, default 0,30,-30: about 45 degrees down).
- It places the survivors (`FLOODUAV_SURVIVOR_SEED`, see Survivor positions below).
- It writes the boundary, start point and survivor positions to a JSON file.

Linux: Unity renders with Vulkan on the NVIDIA GPU (`-force-vulkan`); OpenGL under Xvfb falls back to llvmpipe at about 3 fps.
The mission server runs JAX on GPU 1 (`FLOODUAV_GPU`) without preallocation, so GPU 0 stays free for Unity.

macOS: Unity renders with Metal in a small window (784x200) that shows a preview; the views go to the frame file (`--frames`, default `/tmp/flooduav_frames.raw`).
Unity draws up to 60 frames a second, and up to 3 GPU readbacks run at once, so the capture keeps up with the preset's frame rate.
`--capture-scale 0.5` renders the views at half size (1960x360); it lowers the GPU copy and encoding work, not Unity's CPU use.
A frame covers at most 0.05 s of game time (`FLOODUAV_MAX_DT`). Unity counts its wait for the bridge as frame time, and without the cap a frame ran many physics steps, so the video fell to 2-3 fps.
The patched app has App Nap off (`LSAppNapIsDisabled`), so macOS does not slow it down while it is hidden.
The window must stay open: Unity draws only with a window, so it cannot run headless (`-nographics` turns off the video and the map).
Do not quit it with Cmd+Q or the Dock: the player can hang while quitting. If Unity stops, the bridge exits too; restart with `down`, then `up`. On macOS `down` force-kills a player that does not quit.
The frame file is a 32-byte header (frame number, width, height, slot) and two RGBA slots, bottom row first (`Capture.cs`).
The mission server runs JAX on the CPU.

Graphics presets (`--graphics`, default `auto`; the UI changes it under Settings > Simulation):

| Preset | Render scale | Anti-aliasing | Video fps | H.264 bit rate |
|---|---|---|---|---|
| low | 0.75 | off | 10 | 2 Mbit/s |
| medium | 1.0 | 2x | 20 | 4 Mbit/s |
| high | 1.0 | 4x | 30 | 6 Mbit/s |

`auto` picks `high` on an Apple Silicon Mac or on Linux with an NVIDIA GPU (`nvidia-smi`), else `medium`.
`GET /graphics` returns the preset, the Auto choice and the values in use; `POST /graphics` with `{"preset": ...}` switches it while running.
The bridge sends the render scale and anti-aliasing to Unity over the side channel, and `unity.log` names them.
On a switch the video encoders restart, and viewers wait for the next key frame.

## Behavior and limits

- Mission frame: east = boundary +x, north = +z, origin at the drone start. The lon/lat are made up (`MissionConfig.home`).
- Heights: the drone never flies below 40 m above the start (bridges at about 12 m end the mission). The vehicle maps the 23 m search height to 40 m.
- Landing is real: the drone descends straight down and keeps pushing down on the ground. Over open water (no ground within 150 m) it hovers instead.
- Detection: a drone detects every survivor inside its detection box, the box drawn around it on the map. The box is the downward camera's ground view from the bridge's `--min-alt` (46 x 82 m at 40 m) and keeps that size at any height. The rule uses the true survivor positions with no line of sight; Unity's own visibility sensor is not used.
  The search coverage and belief use the same box: a look counts every cell whose center is inside it, once per pass (`docs/PROTOCOL.md` section 4.3).
  Because the positions are exact, detections merge into one survivor only within 0.5 m (`merge_m`; the kinematic sim keeps 8 m for its 2 m position noise). Survivors a few meters apart stay separate.
- `rl_unity` mode: search drones never switch to rescue on their own (`auto_switch` is skipped). The operator dispatches by hand.
- If a drone leaves the boundary (sides or top, about 83 m above the start), Unity ends its mission and freezes both drones. Stop and Recall cannot move them then; Reset puts them back on their pads. Collisions do not end the mission (`FLOODUAV_SOFT`).
- Setpoint flight: Unity's drone follows a speed command about 0.4 s late and then takes about 1.7 s to reach it, and it brakes at about 4 m/s2 at most (25 degree tilt).
  The bridge steers from where the drone will be `--lead` seconds ahead (default 1.5), so it brakes early and does not swing around a stop or a turn.
  The drone trails a moving setpoint by about its speed x `--lead`. `GET /info` reports `lead`, and the mission server allows that trailing before it pauses the path.
- Top speed: the drone reaches about 14 m/s at most, even with a higher `--max-speed`. With a 15 m/s cap it flies about 10-11 m/s and stays within about 5 m of the path; above that it drifts up to about 9 m off the path.
  So the bridge's `--max-speed` and the UI's drone speed cap both default to 15 m/s (3 m/s in `rl_unity`).
- Commands to the bridge are sent up to 3 times before the drone's task fails.
- Lost link: a flying drone with no position update for 3 s is reported lost and aborted. After a pause of the mission server, the check waits one more 3 s first.
- Two drones under our control, 15 survivors by default (1-100, see Survivor positions). Both drones use the same detection box.
- Restart (`POST /reset`) puts both drones back on their pads and clears Unity's finds. It keeps the survivor positions. Speed cap: `POST /speed`.
- Camera zoom and angle: `POST /camera` with `{"view": "follow", "drone", "distance", "yaw", "pitch"}` or `{"view": "drone", "drone", "fov", "pan", "tilt"}`.
- Video: `GET /stream/<region>` is MJPEG (`?small=1`: 640 px wide, 6 frames a second). `GET /h264/<region>` is H.264 from the GPU encoder (NVENC on Linux, VideoToolbox on macOS, libx264 if that fails), at the graphics preset's bit rate with a key frame every second. Each H.264 encoder runs only while someone watches it.
- Map image: `GET /map.jpg` renders the top-down overview camera once at `--map-px` (default 4096) and caches it.
- The build's mission map (`MissionMapDisplay`) goes to a second display, which Xvfb does not have. With offscreen capture (macOS) the patch turns it off, or it would open as a full-screen window in its own desktop space.
- Speed: `--rate 2` (sim seconds per wall second) with Unity time scale 2. A time scale above the rate drops the video to a few fps.
- Trained agent (`rl_unity` mode): `POST /pilot` with `{"pilot": "onnx"}` lets the build's agent fly D1 on Unity's 936-value observation.
  `"agent": <id>` picks another agent; `GET /agents` lists them (`agents`, with a label such as "DQN 5M").
  The bridge scans `--models` (default `models/`) recursively on each list, so a model copied in shows up without a restart.
  `--agent` sets the default agent (its id is the path under `models/` without the suffix).
  An agent is an ML-Agents `.onnx` file, or a Stable-Baselines3 folder with `model.zip` and `vecnormalize.pkl` (DQN also `action_mapping.json`).
  `sim/flooduav/agents.py` runs the SB3 checkpoints in numpy, so the bridge needs neither torch nor SB3.
  The next setpoint, hold or landing for D1 takes it back.
  With `"limits": "training"` D1 flies with the build's own limits (10 m/s, 3 m/s climb, 20 degree tilt), the ones the agent was trained with.
  With `"limits": "bridge"` it uses `--climb`, `--tilt` and the `/speed` cap.
  The bridge's height floor and obstacle climb do not apply while the agent flies.
  When the agent would leave the boundary within 3 s, the bridge steers D1 back inside and then gives control back to the agent. The boundary has a top too, about 83 m above the start.
  When D1 moves less than 0.5 m in 4 s under the agent (for example resting on a pipe or roof), the bridge climbs it at full power for 2 s.
  The agent needs `onnxruntime` in `.venv-fly` (`docs/SETUP.md`).
- Replay of a FloodUAV mission draws the tracks on a plain map: the log does not keep the Unity top-down image or the search area outline.

## Survivor positions

The build has 15 fixed survivors. The patch (`patch/FloodPatch/Layout.cs`) can move them to random positions drawn from a seed.

- The UI button New survivors (or `POST /survivors` on the bridge) resets the mission and makes a new random layout. Reset and Restart keep it.
- Survivors in map (`Settings` > Simulation) sets how many survivors New survivors places, 1-100. Fewer than 15 hides some of the build's survivors; more adds copies of its first survivor (`Survivor_016` and on).
- Survivor terrain mix (`Settings` > Mission settings) sets how likely each survivor lands on each kind in the table below. It is five percent fields, by default 50 / 25 / 10 / 10 / 5, normalized to 100%. New survivors uses it, and the search prior uses the same mix.
- The same seed gives the same layout. Start with one: `sim/flooduav/run.sh up --survivor-seed 7 --survivor-count 30`. The defaults `-1` keep the build's 15 survivors. With the build's layout the count can only go down.
- Survivors stand at least 10 m apart and 25 m from the drone pads. Nobody is inside a building.

At the first mission start the patch samples the search area every 10 m and gives each sample a cell type:

| Code | Cell type |
|---|---|
| `.` | Dry ground (an elevated road over dry land counts as ground) |
| `w` | Flood water |
| `r` | Flooded house whose roof (any slope) is at least 0.5 m above the water |
| `s` | Building under water |
| `b` | Bridge deck (an object named "bridge") over the flood, at least 0.5 m above the water |
| `h` | Roof of a dry house; roof survivors and the roof prior use it like `r` |
| `o` | Nothing hit |

Houses are the building prefabs of the scene (barns, cabins, churches, villas, industrial buildings and so on). Walls, fences, flood barriers, props and invisible blockers are looked through, so the house, bridge, ground or water under them sets the type.

The layout uses the survivor terrain mix. The default is the shares of `sar/prior/flood_bank.py`:

| Kind | Default share | Where |
|---|---|---|
| Roof | 50% | An `r` sample, on a flat spot |
| High ground | 25% | A `.` sample, weight `exp(-d^2 / (2 * 90^2))` by the distance `d` to the flood, none beyond 240 m, doubled on dry islands |
| Bridge | 10% | A `b` sample, on a flat spot |
| Water | 10% | A `w` sample, standing 0.6 m below the water surface |
| Random | 5% | Any sample except `o` and `s` |

A kind the scene has no samples of gives its share to the others. A mix of only such kinds places the survivors anywhere, like Random. The default shares are hand-set: there is no case data to fit them.
`GET /info` reports the layout seed (`survivor_seed`), the terrain mix (`survivor_shares`), each survivor's kind (`kinds`) and the cell types (`cells`).
The mission server builds the flood prior from the cell types with the same mix, so the operator does not mark area types.

