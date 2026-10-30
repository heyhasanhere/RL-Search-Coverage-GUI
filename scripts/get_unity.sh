#!/usr/bin/env bash
# Download the FloodUAV build from the GitHub Release into unity/.
# Linux: unity/SearchPickupDefault. macOS: unity/FloodUAV.app.
# Usage: scripts/get_unity.sh
# Env: UNITY_TAG (release tag, default unity-v1), UNITY_REPO (owner/name, default from git origin)
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${UNITY_TAG:-unity-v1}
if [ "$(uname)" = Darwin ]; then
  ASSET=FloodUAV-macos.tar.gz
  SHA256=163d41a0c4a2b5a5bcbce8d9b03f82efdd900eae8ba315bdf338e2eba5b09061
  DIR=unity/FloodUAV.app EXE=unity/FloodUAV.app/Contents/MacOS/FloodUAV
else
  ASSET=FloodUAV-linux.tar.gz
  SHA256=1bca95ba404d84b311a13274c0d06866ece3bd266f08c901839fda1a47adc8cd
  DIR=unity/SearchPickupDefault EXE=unity/SearchPickupDefault/FloodUAV.x86_64
fi
if [ -e "$EXE" ]; then
  echo "$DIR is already there"; exit 0
fi
REPO=${UNITY_REPO:-$(git remote get-url origin | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##')}
URL=https://github.com/$REPO/releases/download/$TAG/$ASSET
mkdir -p unity runs
echo "== downloading $URL"
curl -fL --retry 3 -o runs/$ASSET "$URL"
SUM=$(command -v sha256sum || echo "shasum -a 256")
echo "$SHA256  runs/$ASSET" | $SUM -c -
tar -C unity -xzf runs/$ASSET
chmod +x "$EXE"   # the archives were packed on macOS without the exec bit
rm runs/$ASSET
echo "== $DIR ready"
