#!/usr/bin/env python3
"""Robotron AI v3 — Socket server bridge.

TCP server bridging Lua (MAME) ↔ Python for the v3 PPO architecture.
Preserves the exact binary wire protocol from v2 so the Lua side
is completely unchanged.

Wire protocol:
  Inbound (Lua → Python):
    4-byte big-endian length
    Header: >HddBIBBBIBB (n_params, subj_reward, obj_reward, done,
            score, player_alive, save, start_pressed, replay_level,
            num_lasers, wave_number)
    State: n × float32 big-endian
    Optional: preview data

  Outbound (Python → Lua):
    5 bytes: move_cmd(i8), fire_cmd(i8), source_byte(u8),
             start_advanced(u8), start_level_min(u8)
"""

import os
import sys
import time
import struct
import socket
import select
import threading
import traceback
import random
import numpy as np
from collections import deque
from typing import Optional
from dataclasses import dataclass

from .config import CONFIG, GAME_SETTINGS, WIRE_PARAMS_COUNT
from .agent import PPOAgent
from .expert import get_expert_action
from .reward import shape_reward

# ── Constants ───────────────────────────────────────────────────────────────

FIRE_HOLD_FRAMES = 4
_MAX_FRAME_PAYLOAD_BYTES = 4 * 1024 * 1024
_DIAG = 0.70710678
_FIRE_DIR_VECTORS = (
    (0.0, -1.0), (_DIAG, -_DIAG), (1.0, 0.0), (_DIAG, _DIAG),
    (0.0, 1.0), (-_DIAG, _DIAG), (-1.0, 0.0), (-_DIAG, -_DIAG),
)
_START_PULSE_VALID_FRAMES = 240
_GAMEPLAY_RESET_DEAD_FRAMES = 180
_GAMEPLAY_PLAUSIBLE_START_STREAK = 8


# ── Frame data (parsed from wire) ──────────────────────────────────────────

@dataclass
class FrameData:
    state: np.ndarray
    subjreward: float
    objreward: float
    done: bool
    player_alive: bool
    save_signal: bool
    start_pressed: bool = False
    level_number: int = 0
    game_score: int = 0
    next_replay_level: int = 0
    num_lasers: int = 0


def parse_frame_data(data: bytes) -> Optional[FrameData]:
    """Parse the binary wire protocol from Lua."""
    fmt = ">HddBIBBBIBB"
    hdr_size = struct.calcsize(fmt)
    if not data or len(data) < hdr_size:
        return None

    vals = struct.unpack(fmt, data[:hdr_size])
    n, subj, obj, done, score, alive, save, start, replay, lasers, wave = vals

    base_len = hdr_size + n * 4
    if len(data) < base_len:
        return None

    state = np.frombuffer(data[hdr_size:base_len], dtype=">f4", count=n).astype(np.float32)
    if state.shape[0] != n:
        return None

    return FrameData(
        state=state,
        subjreward=subj,
        objreward=obj,
        done=bool(done),
        player_alive=bool(alive),
        save_signal=bool(save),
        start_pressed=bool(start),
        level_number=int(wave),
        game_score=int(score),
        next_replay_level=int(replay),
        num_lasers=int(lasers),
    )


# ── Action encoding (matching game's joystick directions) ──────────────────

def encode_action_to_game(move_dir: int, fire_dir: int) -> tuple[int, int]:
    """Convert model action indices to game joystick commands.

    move/fire 0-7 map to game directions 0-7.
    move/fire 8 (idle) maps to game -1 (no input).
    """
    move_cmd = int(move_dir) if 0 <= move_dir <= 7 else -1
    fire_cmd = int(fire_dir) if 0 <= fire_dir <= 7 else -1
    return move_cmd, fire_cmd


# ── Fire hold logic ─────────────────────────────────────────────────────────

def _apply_fire_hold(cs: dict, raw_fire: int) -> int:
    """Fixed-cadence fire hold for LSPROC's 3-stable-frame requirement."""
    cs["fire_pending_dir"] = int(raw_fire)
    count = cs.get("fire_hold_count", 0)
    if count > 0:
        cs["fire_hold_count"] = count - 1
        return cs.get("fire_hold_dir", raw_fire)
    next_fire = int(cs.get("fire_pending_dir", raw_fire))
    cs["fire_hold_dir"] = next_fire
    cs["fire_hold_count"] = FIRE_HOLD_FRAMES - 1
    return next_fire


# ── Metrics (lightweight rolling stats) ─────────────────────────────────────

class Metrics:
    """Thread-safe rolling metrics for the v3 system."""

    def __init__(self):
        self.lock = threading.Lock()
        self.total_frames = 0
        self.episode_rewards = deque(maxlen=200)
        self.episode_lengths = deque(maxlen=200)
        self.fps_window = deque(maxlen=60)
        self.peak_game_score = 0
        self._last_fps_time = time.time()
        self._fps_frames = 0

    def update_frame(self):
        with self.lock:
            self.total_frames += 1
            self._fps_frames += 1
            now = time.time()
            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                self.fps_window.append(self._fps_frames / elapsed)
                self._fps_frames = 0
                self._last_fps_time = now

    def add_episode(self, reward: float, length: int):
        with self.lock:
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)

    @property
    def avg_reward(self) -> float:
        with self.lock:
            if not self.episode_rewards:
                return 0.0
            return sum(self.episode_rewards) / len(self.episode_rewards)

    @property
    def avg_ep_len(self) -> float:
        with self.lock:
            if not self.episode_lengths:
                return 0.0
            return sum(self.episode_lengths) / len(self.episode_lengths)

    @property
    def fps(self) -> float:
        with self.lock:
            if not self.fps_window:
                return 0.0
            return sum(self.fps_window) / len(self.fps_window)


# ── Socket Server ───────────────────────────────────────────────────────────

class SocketServer:
    """TCP server bridging MAME/Lua clients to the PPO agent.

    Each MAME instance connects on its own TCP socket. The server
    manages per-client state, frame processing, action selection,
    and rollout collection.
    """

    def __init__(
        self,
        agent: PPOAgent,
        host: str = None,
        port: int = None,
        max_clients: int = None,
    ):
        cfg = CONFIG.server
        self.agent = agent
        self.host = host or cfg.host
        self.port = port or cfg.port
        self.max_clients = max_clients or cfg.max_clients

        self.metrics = Metrics()
        self.running = False
        self.shutdown_event = threading.Event()
        self.client_states: dict[int, dict] = {}
        self.client_lock = threading.Lock()
        self._next_cid = 0

        # Rollout collection
        self._rollout_step = 0
        self._rollout_lock = threading.Lock()

    def start(self):
        """Start the server (blocking)."""
        self.running = True
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.settimeout(1.0)

        server_sock.bind((self.host, self.port))
        server_sock.listen(self.max_clients)
        print(f"v3 Socket server listening on {self.host}:{self.port}")

        try:
            while self.running and not self.shutdown_event.is_set():
                try:
                    sock, addr = server_sock.accept()
                except socket.timeout:
                    continue

                with self.client_lock:
                    cid = self._next_cid
                    self._next_cid += 1
                    self.client_states[cid] = self._new_client_state()

                thread = threading.Thread(
                    target=self._handle_client,
                    args=(sock, cid),
                    daemon=True,
                )
                thread.start()
                print(f"Client {cid} connected from {addr}")
        finally:
            self.running = False
            server_sock.close()

    def stop(self):
        """Signal shutdown."""
        self.running = False
        self.shutdown_event.set()

    def _new_client_state(self) -> dict:
        return {
            "frames": 0,
            "player_alive": False,
            "alive_streak": 0,
            "dead_streak": 0,
            "gameplay_seen": False,
            "start_pulse_window": 0,
            "plausible_start_streak": 0,
            "level_number": 0,
            "start_wave": 1,
            "game_score": 0,
            "num_lasers": 0,
            "last_time": time.time(),
            "fps": 0.0,
            "was_done": False,
            "total_reward": 0.0,
            "ep_frames": 0,
            "episode_id": 1,
            "fire_hold_dir": -1,
            "fire_hold_count": 0,
            "fire_pending_dir": -1,
            "last_state": None,
            "last_action": None,
            "last_player_alive": False,
            "prev_action_source": None,
            "last_alive_game_score": 0,
        }

    def _recv_exact(self, sock, n: int, timeout_s: float = 0.5) -> Optional[bytes]:
        """Read exactly n bytes from socket."""
        chunks = []
        remaining = n
        deadline = time.time() + timeout_s
        while remaining > 0:
            if time.time() > deadline:
                return None
            ready = select.select([sock], [], [], max(0.001, deadline - time.time()))
            if not ready[0]:
                continue
            chunk = sock.recv(min(remaining, 65536))
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _pack_action(
        self,
        move_cmd: int,
        fire_cmd: int,
        source_byte: int,
    ) -> bytes:
        """Pack 5-byte action response for Lua."""
        start_adv = 1 if GAME_SETTINGS.start_advanced else 0
        start_level = max(1, GAME_SETTINGS.start_level_min)
        return struct.pack(
            ">bbbBB",
            int(move_cmd),
            int(fire_cmd),
            int(source_byte),
            int(start_adv),
            int(start_level),
        )

    def _handle_client(self, sock: socket.socket, cid: int):
        """Per-client frame loop."""
        try:
            sock.setblocking(False)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            # Handshake
            sock.setblocking(True)
            sock.settimeout(5.0)
            ping = sock.recv(2)
            if not ping or len(ping) < 2:
                raise ConnectionError("No handshake")
            sock.setblocking(False)
            sock.settimeout(None)

            while self.running and not self.shutdown_event.is_set():
                ready = select.select([sock], [], [], 0.002)
                if not ready[0]:
                    continue

                # Read length header
                hdr = self._recv_exact(sock, 4, timeout_s=0.25)
                if hdr is None or len(hdr) < 4:
                    raise ConnectionError("EOF")
                dlen = struct.unpack(">I", hdr)[0]
                if dlen <= 0 or dlen > _MAX_FRAME_PAYLOAD_BYTES:
                    raise ConnectionError(f"Invalid payload: {dlen}")

                data = self._recv_exact(sock, dlen, timeout_s=0.5)
                if data is None:
                    raise ConnectionError("Broken")

                # Validate param count
                if len(data) >= 2:
                    n = struct.unpack(">H", data[:2])[0]
                    if n != WIRE_PARAMS_COUNT:
                        print(f"Client {cid}: param mismatch {n} != {WIRE_PARAMS_COUNT}")
                        break

                frame = parse_frame_data(data)
                if not frame:
                    sock.sendall(self._pack_action(-1, -1, 0))
                    continue

                with self.client_lock:
                    if cid not in self.client_states:
                        break
                    cs = self.client_states[cid]
                    cs["frames"] += 1
                    cs["player_alive"] = frame.player_alive

                    # Track alive/dead streaks
                    if frame.player_alive:
                        cs["alive_streak"] = cs.get("alive_streak", 0) + 1
                        cs["dead_streak"] = 0
                    else:
                        cs["alive_streak"] = 0
                        cs["dead_streak"] = cs.get("dead_streak", 0) + 1

                    if cs["dead_streak"] >= _GAMEPLAY_RESET_DEAD_FRAMES:
                        cs["gameplay_seen"] = False
                        cs["start_pulse_window"] = 0
                        cs["level_number"] = 0
                        cs["start_wave"] = 1
                        cs["game_score"] = 0

                    # Detect game start
                    if frame.start_pressed:
                        cs["start_pulse_window"] = _START_PULSE_VALID_FRAMES
                    elif cs.get("start_pulse_window", 0) > 0:
                        cs["start_pulse_window"] -= 1

                    plausible = frame.player_alive and frame.game_score >= 0
                    if cs.get("start_pulse_window", 0) > 0 and plausible:
                        cs["plausible_start_streak"] = cs.get("plausible_start_streak", 0) + 1
                    else:
                        cs["plausible_start_streak"] = 0

                    if (
                        cs.get("start_pulse_window", 0) > 0
                        and cs.get("alive_streak", 0) >= 15
                        and cs.get("plausible_start_streak", 0) >= _GAMEPLAY_PLAUSIBLE_START_STREAK
                    ):
                        if not cs.get("gameplay_seen"):
                            cs["gameplay_seen"] = True
                            cs["ep_frames"] = 0
                            cs["start_wave"] = max(1, frame.level_number)
                        cs["start_pulse_window"] = 0

                    if frame.player_alive and cs.get("gameplay_seen"):
                        cs["level_number"] = frame.level_number
                        cs["game_score"] = max(0, frame.game_score)

                    if (
                        frame.player_alive
                        and cs.get("gameplay_seen")
                        and frame.num_lasers == 0
                        and frame.game_score > self.metrics.peak_game_score
                    ):
                        self.metrics.peak_game_score = frame.game_score

                    # Track per-game score
                    if frame.player_alive:
                        if frame.game_score < cs.get("last_alive_game_score", 0):
                            cs["ep_frames"] = 0
                            cs["start_wave"] = max(1, frame.level_number)
                        cs["last_alive_game_score"] = frame.game_score

                self.metrics.update_frame()
                self.agent.total_frames = self.metrics.total_frames

                # ── Process previous step reward ────────────────────────
                if cs.get("last_state") is not None and cs.get("last_action") is not None:
                    reward = shape_reward(frame.objreward, frame.subjreward, frame.done)
                    cs["total_reward"] += reward
                    cs["ep_frames"] = cs.get("ep_frames", 0) + 1

                # ── Terminal ────────────────────────────────────────────
                if frame.done:
                    self.agent._reset_frame_buffer(cid)
                    cs["fire_hold_dir"] = -1
                    cs["fire_hold_count"] = 0
                    cs["fire_pending_dir"] = -1

                    if not cs.get("was_done"):
                        self.metrics.add_episode(
                            cs["total_reward"],
                            cs.get("ep_frames", 0),
                        )
                    cs["was_done"] = True
                    sock.sendall(self._pack_action(-1, -1, 0))
                    cs["last_state"] = cs["last_action"] = None
                    cs["last_player_alive"] = False
                    cs["prev_action_source"] = None
                    cs["episode_id"] = cs.get("episode_id", 1) + 1
                    cs["total_reward"] = 0.0
                    cs["ep_frames"] = 0
                    continue

                if cs.get("was_done"):
                    cs["was_done"] = False
                    cs["total_reward"] = 0.0
                    cs["ep_frames"] = 0

                # Skip dead/attract frames
                if not frame.player_alive:
                    self.agent._reset_frame_buffer(cid)
                    cs["last_state"] = cs["last_action"] = None
                    cs["fire_hold_dir"] = -1
                    cs["fire_hold_count"] = 0
                    sock.sendall(self._pack_action(-1, -1, 0))
                    continue

                # ── Choose action ───────────────────────────────────────
                wire_state = frame.state
                epsilon = self.agent.get_epsilon()
                expert_ratio = self.agent.get_expert_ratio()

                fire_update_open = cs.get("fire_hold_count", 0) <= 0
                locked_fire = None
                if not fire_update_open:
                    held = cs.get("fire_hold_dir", -1)
                    locked_fire = max(0, min(8, held)) if held >= 0 else 8

                # Decide: expert vs policy
                use_expert = random.random() < expert_ratio
                action_source = "none"
                is_epsilon = False

                if use_expert:
                    wave = max(1, cs.get("level_number", 1))
                    move_idx, fire_idx = get_expert_action(wire_state, wave_number=wave)
                    if locked_fire is not None:
                        fire_idx = locked_fire
                    action_source = "expert"
                else:
                    move_idx, fire_idx, is_epsilon = self.agent.act(
                        wire_state,
                        epsilon=epsilon,
                        client_id=cid,
                        locked_fire=locked_fire,
                    )
                    action_source = "dqn"

                # Apply fire hold
                effective_fire = _apply_fire_hold(cs, fire_idx)

                cs["last_state"] = wire_state
                cs["last_action"] = (move_idx, effective_fire)
                cs["last_player_alive"] = frame.player_alive
                cs["prev_action_source"] = action_source

                # Save signal
                if frame.save_signal:
                    self.agent.save()

                # Send action
                move_cmd, fire_cmd = encode_action_to_game(move_idx, effective_fire)
                if action_source == "expert":
                    source_byte = 3
                elif is_epsilon:
                    source_byte = 2
                elif action_source == "dqn":
                    source_byte = 1
                else:
                    source_byte = 0

                sock.sendall(self._pack_action(move_cmd, fire_cmd, source_byte))

        except Exception as e:
            is_expected = isinstance(e, (ConnectionError, BrokenPipeError, ConnectionResetError, TimeoutError))
            if not is_expected:
                print(f"Client {cid} error: {e}")
                traceback.print_exc()
        finally:
            with self.client_lock:
                self.client_states.pop(cid, None)
            try:
                sock.close()
            except Exception:
                pass
            print(f"Client {cid} disconnected")
