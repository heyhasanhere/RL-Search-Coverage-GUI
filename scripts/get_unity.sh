#!/usr/bin/env bash
# Download the FloodUAV Linux build from the GitHub Release into unity/SearchPickupDefault.
# Usage: scripts/get_unity.sh
# Env: UNITY_TAG (release tag, default unity-v1), UNITY_REPO (owner/name, default from git origin)
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${UNITY_TAG:-unity-v1}
ASSET=FloodUAV-linux.tar.gz
SHA256=1bca95ba404d84b311a13274c0d06866ece3bd266f08c901839fda1a47adc8cd
if [ -e unity/SearchPickupDefault/FloodUAV.x86_64 ]; then
  echo "unity/SearchPickupDefault is already there"; exit 0
fi
REPO=${UNITY_REPO:-$(git remote get-url origin | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##')}
URL=https://github.com/$REPO/releases/download/$TAG/$ASSET
mkdir -p unity runs
echo "== downloading $URL"
curl -fL --retry 3 -o runs/$ASSET "$URL"
echo "$SHA256  runs/$ASSET" | sha256sum -c -
tar -C unity -xzf runs/$ASSET
chmod +x unity/SearchPickupDefault/FloodUAV.x86_64   # the archive was packed on macOS without the exec bit
rm runs/$ASSET
echo "== unity/SearchPickupDefault ready"
