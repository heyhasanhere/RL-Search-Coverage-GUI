# Search Coverage: Operator GUI

This project holds the multi-drone search and rescue operator GUI and everything it needs to run.

The GUI is a web app (`ui/`) served by a Python mission server (`sar/mission/server.py`).
The server flies the drones through one of two backends:

| Backend | Sim | Where |
|---|---|---|
| `kinematic` | Simple point-mass drones, simulated detector | Any machine |
| `flooduav` | FloodUAV Unity town, two drones | Linux with an NVIDIA GPU |

FloodUAV is the only 3D sim.
The UI can also replay a mission log or a packed recording from `runs/replays/`.

## Folder Structure

```
ui/                      React and MapLibre web app (Vite)
sar/                     Python package
  mission/               server, orchestrator, vehicles, detectors, video, replay
  env/, search/, train/  grid envs and networks the RL search modes run
  baselines/             lawnmower and greedy
  prior/                 flood prior
  eval/tier0.py          scenario maps the mission server imports
  config.py, config_v2.py
sim/
  flooduav/              Unity bridge, build patch (C#), run.sh
unity/SearchPickupDefault/  FloodUAV Linux build, downloaded by scripts/get_unity.sh (not in git)
models/
  ft_flood/seed1/        RL search policy fine-tuned on flood towns (default `rl` mode)
  ft_best/seed1/         previous RL search policy (`--ckpt models/ft_best/seed1`)
  v2_m4/final/           v2 policy for the altitude and tree-search modes
  unity_survivor_search/ FloodUAV's own trained agent (`rl_unity` mode)
  RL_PPO-TRPO-DQN/       more `rl_unity` agents: PPO checkpoints, DQN and TRPO
data/
  pd_table.json          detection probability table
  regions/demo_flood.npz flood town layers for the flood prior
scripts/                 kinematic demo, mission driver, Unity build download, replay recorder and packer
tests/                   Python tests
docs/                    setup, protocol, status
runs/                    logs, mission logs and packed replays (output)
```

## Quick Start

All commands run from the project root.
Setup for each machine is in `docs/SETUP.md`.

Kinematic drones on any machine:

```bash
uv sync --extra mission --extra prior
cd ui && npm install && npm run build && cd ..
uv run python -m sar.mission.server --backend kinematic --speed 5 --show-truth
```

Open http://localhost:8000.

FloodUAV on the Linux GPU machine (UI on port 8001):

```bash
scripts/get_unity.sh       # once: downloads the Unity build (512 MB) into unity/
sim/flooduav/run.sh up
sim/flooduav/run.sh down
```

Open `http://<host>:8001`.

## Docs

| File | Content |
|---|---|
| `docs/SETUP.md` | Environments, system installs, running on the GPU machine |
| `docs/PROTOCOL.md` | Server API and WebSocket messages |
| `sim/flooduav/README.md` | FloodUAV backend details |
| `ui/README.md` | Web app notes |
