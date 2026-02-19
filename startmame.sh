#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUA_SCRIPT="$SCRIPT_DIR/Scripts/main.lua"

if [[ "${1:-}" == "-kill" ]]; then
    echo "Killing all running MAME instances..."
    pids=$(ps ax | grep '[m]ame.*tempest1' | awk '{print $1}')
    if [[ -n "$pids" ]]; then
        echo "$pids" | xargs kill -9
        echo "Killed PIDs: $(echo $pids | tr '\n' ' ')"
    else
        echo "No MAME instances found."
    fi
    exit 0
fi

COUNT="${1:-1}"

if ! [[ "$COUNT" =~ ^[0-9]+$ ]] || [[ "$COUNT" -lt 1 ]]; then
    echo "Usage: $0 [COUNT | -kill]"
    echo "  COUNT   Number of MAME instances to launch (default: 1)"
    echo "  -kill   Kill all running MAME instances"
    exit 1
fi

SOUND_FLAG=""
if [[ "$COUNT" -gt 1 ]]; then
    SOUND_FLAG="-sound none"
fi

echo "Launching $COUNT MAME instance(s)..."
for i in $(seq 1 "$COUNT"); do
    mame tempest1 -nothrottle $SOUND_FLAG -skip_gameinfo -autoboot_script "$LUA_SCRIPT" &
    echo "  Started instance $i (PID $!)"
done
echo "All instances launched."

