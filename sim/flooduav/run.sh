#!/usr/bin/env bash
# FloodUAV with the operator UI (sim/flooduav/README.md), on the Linux GPU machine or a Mac.
# Usage: sim/flooduav/run.sh [up|down [all]]
# up: the Unity bridge on :8101 and the mission server with the UI on :8001.
#     Linux: Unity on an Xvfb screen, video grabbed from it. macOS: video from the patch's offscreen frames.
# Logs: runs/flooduav_ui/{bridge,server,unity}.log
set -u
cd "$(dirname "$0")/../.."
DISP=${FLOODUAV_DISPLAY:-:97}
BPORT=${FLOODUAV_BRIDGE_PORT:-8101}
PORT=${FLOODUAV_PORT:-8001}
LOG=runs/flooduav_ui
MAC=$([ "$(uname)" = Darwin ] && echo 1 || echo 0)
if [ "${1:-up}" = down ]; then
  pkill -f "[s]im/flooduav/bridge.py"; pkill -f "[b]ackend flooduav"
  if [ $MAC = 1 ]; then
    pkill -f "[F]loodUAVPatched.app/Contents/MacOS"
    sleep 2; pkill -9 -f "[F]loodUAVPatched.app/Contents/MacOS"    # a player stuck while quitting
    exit 0
  fi
  pkill -x FloodUAV.x86_64
  pkill -f "[x]11grab.*$DISP"; pkill -f "[F]LOODUAV_FOCUS"
  [ "${2:-}" = all ] && pkill -f "[X]vfb $DISP"     # down all: also the X server (needed after a screen size change)
  exit 0
fi
mkdir -p $LOG
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
if [ $MAC = 1 ]; then
  [ -x unity/FloodUAVPatched.app/Contents/MacOS/FloodUAV ] || bash sim/flooduav/patch/build.sh
else
  [ -x unity/SearchPickupPatched/FloodUAV.x86_64 ] || bash sim/flooduav/patch/build.sh
  [ -e /tmp/.X11-unix/X${DISP#:} ] || { Xvfb $DISP -screen 0 3920x720x24 >/dev/null 2>&1 & sleep 2; }
  export DISPLAY=$DISP
  # The patch sets runInBackground; the focus loop is a fallback for players that still pause.
  ( exec -a FLOODUAV_FOCUS bash -c 'while true; do for w in $(xdotool search --name FloodUAV 2>/dev/null); do xdotool windowfocus $w 2>/dev/null; done; sleep 5; done' ) &
fi
nohup .venv-fly/bin/python -u sim/flooduav/bridge.py --port $BPORT --unity-log "$PWD/$LOG/unity.log" "${@:2}" >$LOG/bridge.log 2>&1 &
for i in $(seq 120); do curl -sf localhost:$BPORT/time >/dev/null && break; sleep 2; done
curl -sf localhost:$BPORT/time >/dev/null || { echo "bridge did not start, see $LOG/bridge.log"; exit 1; }
# Public deployment: the operator password (SAR_CONTROL_PASSWORD=...) lives outside the project
ENV_FILE=${SAR_ENV_FILE:-$HOME/.config/sar-search/control.env}
[ -f "$ENV_FILE" ] && { set -a; . "$ENV_FILE"; set +a; }
# Linux: JAX on its own GPU without preallocation, so GPU 0 stays free for Unity. macOS: JAX on the CPU.
[ $MAC = 0 ] && export CUDA_VISIBLE_DEVICES=${FLOODUAV_GPU:-1}
XLA_PYTHON_CLIENT_PREALLOCATE=false \
nohup uv run --no-sync --extra mission python -W ignore -m sar.mission.server --backend flooduav \
  --bridge http://127.0.0.1:$BPORT --port $PORT --show-truth --log runs/mission/flooduav_ui.jsonl >$LOG/server.log 2>&1 &
for i in $(seq 60); do curl -sf localhost:$PORT/api/config >/dev/null && break; sleep 2; done
curl -sf localhost:$PORT/api/config >/dev/null && echo "UI on http://$(hostname):$PORT" || echo "server did not start, see $LOG/server.log"
