# Setup

All commands run from the project root.
Any machine runs the kinematic backend, the UI build and the tests.
The FloodUAV backend runs on Linux with an NVIDIA GPU (two GPUs: section 6) or on a Mac with Apple silicon (section 7).

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

The Unity build is too large for git, so it is attached to the GitHub Release `unity-v1`: `FloodUAV-linux.tar.gz` (500 MB) and `FloodUAV-macos.tar.gz` (480 MB).

```bash
scripts/get_unity.sh
```

The script downloads the archive for this OS, checks its SHA-256 and unpacks it into `unity/SearchPickupDefault` (Linux) or `unity/FloodUAV.app` (macOS).
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
At startup it compiles the search policies for every grid size there, and `runs/flooduav_ui/server.log` shows `search policies ready` when done.
Ports: UI 8001, bridge 8101.

## 7. macOS

FloodUAV also runs on a Mac with Apple silicon, from a macOS build of the same Unity project.
Unity renders with Metal, the patch renders the views offscreen and the bridge encodes them with VideoToolbox, so no X server or screen capture is needed.
The mission server runs JAX on the CPU.

1. Install Homebrew, `uv` and Node.js. Create both venvs (section 1, without `--extra cuda`) and build the UI (section 2).
2. Install the tools:

   ```bash
   brew install ffmpeg
   curl -sSL https://dot.net/v1/dotnet-install.sh | bash -s -- --channel 8.0 --install-dir ~/.dotnet
   ```

3. Get the macOS build: `scripts/get_unity.sh` (section 3) unpacks it into `unity/FloodUAV.app`.
   To use your own build instead, build the player in Unity (File > Build Profiles > macOS, Apple silicon, Mono scripting backend) and save it there.
   The patch needs Mono: IL2CPP builds have no `Managed/Assembly-CSharp.dll`.
4. Start and stop as on Linux:

   ```bash
   sim/flooduav/run.sh up      # builds unity/FloodUAVPatched.app on first use
   sim/flooduav/run.sh down
   ```

Open `http://localhost:8001`. The first `up` takes about a minute until `search policies ready` appears in `runs/flooduav_ui/server.log`.
`build.sh` signs the patched app ad hoc and clears the download quarantine, so macOS starts it without a Gatekeeper prompt.
Unity opens a small FloodUAV window with a preview of the views. Leave it open and stop everything with `run.sh down`, not Cmd+Q (`sim/flooduav/README.md`).
After replacing `unity/FloodUAV.app`, delete `unity/FloodUAVPatched.app` so `up` patches the new build.
