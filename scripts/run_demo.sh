#!/usr/bin/env bash
# One-command demo on the Mac or any machine: kinematic drones (3 search, 1 rescue), simulated detector,
# the fine-tuned RL policy models/ft_flood/seed1 (the server default).
# Usage: scripts/run_demo.sh [port]
set -eu
cd "$(dirname "$0")/.."
PORT=${1:-8000}
[ -f ui/dist/index.html ] || (cd ui && npm install && npm run build)
exec uv run python -m sar.mission.server --backend kinematic --speed 5 --roles search,search,search,rescue --show-truth --port "$PORT"
