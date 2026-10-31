# Search Coverage: Operator GUI

A web GUI for running multi-drone search and rescue missions.
Search drones sweep an area for survivors, and rescue drones fly to the ones they find.
The operator watches the map, the detections and the drone video, and can step in at any time.

The GUI is a React web app (`ui/`) served by a Python mission server (`sar/mission/server.py`).
The UI can also replay a mission log or a packed recording from `runs/replays/`.

## Backends

The server flies the drones through one of two backends.

| Backend | Sim | Runs on |
|---|---|---|
| `kinematic` | Simple point-mass drones with a simulated detector | Any machine |
| `flooduav` | FloodUAV, a 3D Unity flood town with two drones and live video | Linux with an NVIDIA GPU, or a Mac with Apple silicon |

## Search Modes

The search mode decides where each search drone flies next.
It can be changed before and during a mission.

| Mode | How it picks the next move | Weights |
|---|---|---|
| `rl` | MAPPO policy trained on flood towns | `models/mappo_flood/seed1` |
| `mcts` | Tree search with a hand-written prior and greedy rollouts | None |
| `greedy` | Best expected find in the next step | None |
| `lawnmower` | Fixed back-and-forth sweep | None |
| `rl_unity` | FloodUAV only: a trained agent flies the first search drone directly | `models/ppo_19m.onnx`, `models/DQN`, `models/TRPO` |

Start the server with `--ckpt models/mappo_general/seed1` to run `rl` with the general MAPPO policy instead of the flood one.
In `rl_unity` the agent can be changed from the UI settings.
Any ML-Agents `.onnx` file or Stable-Baselines3 folder copied into `models/` appears in that list (`sim/flooduav/README.md`).

## Quick Start

All commands run from the project root.
Setup for each machine is in `docs/SETUP.md`.

### Kinematic drones (any machine)

```bash
uv sync --extra mission --extra prior
cd ui && npm install && npm run build && cd ..
uv run python -m sar.mission.server --backend kinematic --speed 5 --show-truth
```

Open http://localhost:8000.

At startup the server compiles the search policies in the background and prints `search policies ready`.
This takes about 30 s on a laptop CPU.
Later starts take about 10 s, because compiled code is cached in `runs/jax_cache`.
A mission started before then waits for the compile.
`--no-warm` skips the startup compile, and `--jax-cache ""` turns the cache off.

### FloodUAV (Linux GPU machine or Mac)

FloodUAV needs a second venv for the bridge (`docs/SETUP.md` section 1.1).

```bash
scripts/get_unity.sh       # once: downloads the Unity build for this OS (about 500 MB) into unity/
sim/flooduav/run.sh up     # start Unity, the bridge and the mission server
sim/flooduav/run.sh down   # stop them
```

Open `http://<host>:8001`, or http://localhost:8001 on a Mac (`docs/SETUP.md` section 7).

## Tests

```bash
uv run --group dev python -m pytest
```

## Folder Structure

```
ui/                        React and MapLibre web app (Vite)
sar/                       Python package
  mission/                 mission server, orchestrator, vehicles, detectors, video, replay
  env/, search/, train/    grid environments and networks used by the RL search modes
  baselines/               lawnmower, greedy and planner
  prior/                   flood prior
  eval/tier0.py            scenario maps used by the mission server
  config.py, config_v2.py  environment settings
sim/flooduav/              FloodUAV bridge, Unity build patch (C#), run.sh
models/                    trained weights (see Search Modes)
data/
  pd_table.json            detection probability table
  regions/demo_flood.npz   flood town layers for the flood prior
scripts/                   kinematic demo, mission driver, Unity build download, replay recorder and packer
tests/                     Python tests
docs/                      setup and protocol
unity/                     FloodUAV builds from scripts/get_unity.sh (not in git)
runs/                      logs, mission logs and packed replays (output, not in git)
```

## Docs

| File | Content |
|---|---|
| `docs/SETUP.md` | Environments, system installs, running on the GPU machine or a Mac |
| `docs/PROTOCOL.md` | Server API and WebSocket messages |
| `sim/flooduav/README.md` | FloodUAV backend details |
| `ui/README.md` | Web app layout and controls |
