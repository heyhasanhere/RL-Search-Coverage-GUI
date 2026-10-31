#!/usr/bin/env bash
# Build the patched FloodUAV player (sim/flooduav/README.md).
# Usage: sim/flooduav/patch/build.sh [original build] [patched build]
# Linux: build folders (default unity/SearchPickupDefault -> unity/SearchPickupPatched).
# macOS: app bundles (default unity/FloodUAV.app -> unity/FloodUAVPatched.app).
# The patched build is a copy of the original (hard links on Linux, clones on macOS);
# only Managed/Assembly-CSharp.dll is replaced, and FloodPatch.dll is added.
set -eu
cd "$(dirname "$0")"
ROOT=$(cd ../../.. && pwd)
export PATH=$HOME/.dotnet:$PATH DOTNET_ROOT=$HOME/.dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
if [ "$(uname)" = Darwin ]; then
  SRC=${1:-$ROOT/unity/FloodUAV.app}
  DST=${2:-$ROOT/unity/FloodUAVPatched.app}
  M=$SRC/Contents/Resources/Data/Managed
  OUT=$DST/Contents/Resources/Data/Managed
else
  SRC=${1:-$ROOT/unity/SearchPickupDefault}
  DST=${2:-$ROOT/unity/SearchPickupPatched}
  M=$SRC/FloodUAV_Data/Managed
  OUT=$DST/FloodUAV_Data/Managed
fi
dotnet build FloodPatch -c Release -o out/hook -p:Managed="$M" -v q
dotnet build Injector -c Release -o out/injector -v q
rm -rf "$DST"
if [ "$(uname)" = Darwin ]; then
  cp -cR "$SRC" "$DST"
  xattr -cr "$DST"                       # a downloaded build is quarantined
  chmod +x "$DST"/Contents/MacOS/*
else
  cp -al "$SRC" "$DST"
  chmod +x "$DST/FloodUAV.x86_64"
fi
rm "$OUT/Assembly-CSharp.dll"
dotnet out/injector/Injector.dll "$M/Assembly-CSharp.dll" out/hook/FloodPatch.dll "$OUT/Assembly-CSharp.dll"
cp out/hook/FloodPatch.dll "$OUT/"
# macOS slows a hidden app down (App Nap), and with it the video
[ "$(uname)" = Darwin ] && /usr/libexec/PlistBuddy -c "Add :LSAppNapIsDisabled bool true" "$DST/Contents/Info.plist"
# the changed bundle needs a new (ad hoc) signature, or macOS refuses to start it
[ "$(uname)" = Darwin ] && codesign --force --deep --sign - "$DST" >/dev/null 2>&1
echo "patched build: $DST"
