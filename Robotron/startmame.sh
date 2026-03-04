#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUA_SCRIPT="$SCRIPT_DIR/Scripts/main.lua"
LOG_DIR="$SCRIPT_DIR/logs"

usage() {
    echo "Usage: $0 [COUNT] [--fg] [-kill]"
    echo "  COUNT   Number of MAME instances to launch (default: 1, background mode only)"
    echo "  --fg    Run one MAME instance in foreground"
    echo "  -kill   Kill all running Robotron MAME instances"
}

FOREGROUND=0
COUNT="1"
COUNT_SET=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -kill)
            echo "Killing all running Robotron MAME instances..."
            pids=$(pgrep -f 'mame.*robotron' || true)
            if [[ -n "$pids" ]]; then
                echo "$pids" | xargs kill -9
                echo "Killed PIDs: $(echo "$pids" | tr '\n' ' ')"
            else
                echo "No Robotron MAME instances found."
            fi
            exit 0
            ;;
        --fg)
            FOREGROUND=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [[ "$COUNT_SET" -eq 1 ]]; then
                echo "error: multiple COUNT values provided" >&2
                usage >&2
                exit 1
            fi
            COUNT="$1"
            COUNT_SET=1
            shift
            ;;
    esac
done

if ! [[ "$COUNT" =~ ^[0-9]+$ ]] || [[ "$COUNT" -lt 1 ]]; then
    echo "error: COUNT must be a positive integer" >&2
    usage >&2
    exit 1
fi

if [[ "$FOREGROUND" -eq 1 && "$COUNT" -ne 1 ]]; then
    echo "error: --fg supports COUNT=1 only" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

SOUND_FLAG=""
if [[ "$COUNT" -gt 1 ]]; then
    SOUND_FLAG="-sound none"
fi

if [[ "$FOREGROUND" -eq 1 ]]; then
    echo "Mode: foreground"
    echo "Launching 1 MAME instance (attached)..."
    exec mame robotron -nothrottle $SOUND_FLAG -window -skip_gameinfo -autoboot_script "$LUA_SCRIPT"
fi

echo "Mode: background"
echo "Launching $COUNT MAME instance(s)..."
declare -a PIDS=()
for i in $(seq 1 "$COUNT"); do
    mame robotron -nothrottle $SOUND_FLAG -window -skip_gameinfo -autoboot_script "$LUA_SCRIPT" &
    pid=$!
    PIDS+=("$pid")
    echo "  Started instance $i (PID $pid)"
done

echo "All instances launched. Checking process liveness..."
sleep 1
for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        echo "  RUNNING: PID $pid"
    else
        echo "  NOT RUNNING: PID $pid"
    fi
done

echo "Script exits now; MAME continues running in background."
