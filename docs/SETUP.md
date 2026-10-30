# Setup

All commands run from the project root.
Any machine runs the kinematic backend, the UI build and the tests.
The FloodUAV backend needs Linux with an NVIDIA GPU (tested on Ubuntu with two RTX 3060).

## 1. Python Environments

| Venv | Used by | Create |
|---|---|---|
| `.venv` | mission server, tests, scripts | `uv sync --extra mission --extra prior` (on the GPU machine add `--extra cuda`) |
| `.venv-fly` | FloodUAV bridge (`sim/flooduav/bridge.py`) | see 1.1 |

`uv sync --all-extras` fails on macOS because `jax-cuda12-plugin` is Linux only.
A plain `uv sync` on Linux installs CPU-only JAX, so name `--extra cuda` there.

### 1.1 FloodUAV Bridge Venv

`mlagents-envs` 1.1.0 pins old packages, so it is installed without its dependencies.

```bash
uv venv .venv-fly --python 3.12
uv pip install --python .venv-fly/bin/python --no-deps mlagents-envs==1.1.0
uv pip install --python .venv-fly/bin/python grpcio==1.62.2 protobuf==3.20.3 "numpy<2" setuptools cloudpickle pyyaml filelock pillow
uv pip install --python .venv-fly/bin/python onnxruntime==1.19.2    # the rl_unity mode's trained agent
```

`setuptools` is needed because `mlagents-envs` imports `distutils`.
`sim/flooduav/run.sh` sets `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.

## 2. Web UI

Node.js is needed.

```bash
cd ui && npm install && npm run build && cd ..
```

The server serves `ui/dist`.
For UI work, `npm run dev` in `ui/` runs Vite against a running server.

## 3. FloodUAV Unity Build

The Unity build is too large for git, so it is attached to the GitHub Release `unity-v1` as `FloodUAV-linux.tar.gz` (512 MB).

```bash
scripts/get_unity.sh
```

The script downloads the archive, checks its SHA-256 and unpacks it into `unity/SearchPickupDefault`.
It reads the repository from `git remote get-url origin`; set `UNITY_REPO=<owner>/<name>` or `UNITY_TAG` to override.
`sim/flooduav/run.sh up` then builds the patched copy `unity/SearchPickupPatched` on first use.

## 4. System Installs on the GPU Machine

These are outside the project folder.

| Install | Needed for | Notes |
|---|---|---|
| `xvfb`, `ffmpeg`, `xdotool` (apt) | FloodUAV | virtual screen, video streams, window focus |
| NVIDIA driver with Vulkan | FloodUAV | Unity runs with `-force-vulkan`; OpenGL under Xvfb falls back to about 3 fps |
| .NET 8 SDK in `~/.dotnet` | FloodUAV patch build | `sim/flooduav/patch/build.sh` |

## 5. Copy to the GPU Machine

```bash
rsync -a --exclude .venv --exclude ui/node_modules ./ <host>:~/search-coverage/
```

Then on that machine create the venvs (section 1), build the UI (section 2) and get the Unity build (section 3).
`sim/flooduav/run.sh up` builds the patched Unity player on first use (`unity/SearchPickupPatched`).
Run `sim/flooduav/run.sh down` in its own SSH call, because its `pkill` patterns can match a command line that contains them.
`down` leaves the virtual screen running for the next `up`; `down all` stops it too.

## 6. GPUs

Unity renders on GPU 0.
The mission server runs JAX on GPU 1 (`FLOODUAV_GPU`) without preallocation.
Ports: UI 8001, bridge 8101.
