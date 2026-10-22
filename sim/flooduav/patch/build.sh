#!/usr/bin/env bash
# Build the patched FloodUAV player on the Linux GPU machine (sim/flooduav/README.md).
# Usage: sim/flooduav/patch/build.sh [original build dir] [patched build dir]
# The patched dir is a hard-linked copy of the original; only Managed/Assembly-CSharp.dll is replaced.
set -eu
cd "$(dirname "$0")"
ROOT=$(cd ../../.. && pwd)
SRC=${1:-$ROOT/unity/SearchPickupDefault}
DST=${2:-$ROOT/unity/SearchPickupPatched}
export PATH=$HOME/.dotnet:$PATH DOTNET_ROOT=$HOME/.dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
M=$SRC/FloodUAV_Data/Managed
dotnet build FloodPatch -c Release -o out/hook -p:Managed="$M" -v q
dotnet build Injector -c Release -o out/injector -v q
rm -rf "$DST"
cp -al "$SRC" "$DST"
chmod +x "$DST/FloodUAV.x86_64"
rm "$DST/FloodUAV_Data/Managed/Assembly-CSharp.dll"
dotnet out/injector/Injector.dll "$M/Assembly-CSharp.dll" out/hook/FloodPatch.dll "$DST/FloodUAV_Data/Managed/Assembly-CSharp.dll"
cp out/hook/FloodPatch.dll "$DST/FloodUAV_Data/Managed/"
echo "patched build: $DST/FloodUAV.x86_64"
