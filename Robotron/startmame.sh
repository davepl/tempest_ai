#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUA_SCRIPT="$SCRIPT_DIR/Scripts/main.lua"
LOG_DIR="$SCRIPT_DIR/logs"
ROM_DIR="$SCRIPT_DIR/roms"
MAME_BIN="${MAME_BIN:-mame}"

# Default to project-local ROMs; allow callers to append/override via MAME_ROMPATH.
if [[ -n "${MAME_ROMPATH:-}" ]]; then
    ROMPATH="$ROM_DIR;$MAME_ROMPATH"
else
    ROMPATH="$ROM_DIR"
fi

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

# Guard against stacking multiple wavwrite-enabled Robotron launches.
if pgrep -f "mame.*robotron.*wavwrite" > /dev/null 2>&1; then
    echo "ERROR: MAME with audio output is already running!"
    echo "Kill existing instances with: killall mame"
    exit 1
fi

WARNING_FLAG=""
if "$MAME_BIN" -showusage 2>&1 | grep -q -- "-skip_warnings"; then
    WARNING_FLAG="-skip_warnings"
else
    echo "Note: this MAME build does not support -skip_warnings; continuing without it."
fi

if [[ ! -d "$ROM_DIR" ]]; then
    echo "error: ROM directory not found: $ROM_DIR" >&2
    exit 1
fi

if ! "$MAME_BIN" -rompath "$ROMPATH" -verifyroms robotron >/dev/null 2>&1; then
    echo "error: Robotron ROM verification failed for rompath: $ROMPATH" >&2
    "$MAME_BIN" -rompath "$ROMPATH" -verifyroms robotron || true
    exit 1
fi

if [[ "$FOREGROUND" -eq 1 ]]; then
    AUDIO_WAV="/tmp/robotron_audio_client0.wav"
    rm -f "$AUDIO_WAV" 2>/dev/null || true
    echo "Mode: foreground"
    echo "Launching 1 MAME instance (attached) with preview audio/video enabled, throttled to real time..."
    SOUND_FLAG="-wavwrite $AUDIO_WAV -samplerate 48000 -audio_latency 1"
    THROTTLE_FLAG="-throttle -speed 1.0"
    VIDEO_FLAG="-video soft"
    exec env ROBOTRON_PREVIEW_CLIENT=1 ROBOTRON_CLIENT_SLOT=0 "$MAME_BIN" robotron -rompath "$ROMPATH" $THROTTLE_FLAG $SOUND_FLAG $VIDEO_FLAG -window -skip_gameinfo $WARNING_FLAG -autoboot_script "$LUA_SCRIPT"
fi

echo "Mode: background"
if [[ "$COUNT" -eq 1 ]]; then
    echo "Launching 1 MAME instance (client 0) with preview audio/video enabled, throttled to real time..."
else
    echo "Launching $COUNT MAME instance(s): client 0 preview-capable, others training-only..."
fi
declare -a PIDS=()
for i in $(seq 1 "$COUNT"); do
    CLIENT_SLOT=$((i-1))
    if [[ $i -eq 1 ]]; then
        AUDIO_WAV="/tmp/robotron_audio_client${CLIENT_SLOT}.wav"
        rm -f "$AUDIO_WAV" 2>/dev/null || true
        SOUND_FLAG="-wavwrite $AUDIO_WAV -samplerate 48000 -audio_latency 1"
        THROTTLE_FLAG="-throttle -speed 1.0"
        VIDEO_FLAG="-video soft"
        PREVIEW_CLIENT_FLAG="1"
    else
        SOUND_FLAG="-sound none"
        THROTTLE_FLAG="-nothrottle"
        VIDEO_FLAG="-video none"
        PREVIEW_CLIENT_FLAG="0"
    fi
    LOG_FILE="$LOG_DIR/mame_instance_${CLIENT_SLOT}.log"
    if [[ $i -eq 1 ]]; then
        # Keep the first instance attached to the terminal so one clean set of init lines stays visible.
        ROBOTRON_PREVIEW_CLIENT="$PREVIEW_CLIENT_FLAG" ROBOTRON_CLIENT_SLOT="$CLIENT_SLOT" "$MAME_BIN" robotron -rompath "$ROMPATH" $THROTTLE_FLAG $SOUND_FLAG $VIDEO_FLAG -skip_gameinfo $WARNING_FLAG -autoboot_script "$LUA_SCRIPT" &
    else
        # Additional clients stay backgrounded and log to per-instance files.
        ROBOTRON_PREVIEW_CLIENT="$PREVIEW_CLIENT_FLAG" ROBOTRON_CLIENT_SLOT="$CLIENT_SLOT" "$MAME_BIN" robotron -rompath "$ROMPATH" $THROTTLE_FLAG $SOUND_FLAG $VIDEO_FLAG -skip_gameinfo $WARNING_FLAG -autoboot_script "$LUA_SCRIPT" >> "$LOG_FILE" 2>&1 &
    fi
    pid=$!
    PIDS+=("$pid")
    if [[ $i -eq 1 ]]; then
        echo "  Started instance $i (client $CLIENT_SLOT - audio/video, throttled) PID $pid  log: $LOG_FILE"
    else
        echo "  Started instance $i (client $CLIENT_SLOT - training-only, unthrottled) PID $pid  log: $LOG_FILE"
    fi
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
