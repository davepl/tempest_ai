#!/usr/bin/env python3
# ==================================================================================================================
# ||  ROBOTRON AI v2 • CONFIGURATION                                                                             ||
# ||  Rainbow engine with factored dual-joystick action heads                                                     ||
# ==================================================================================================================
"""Central configuration: server, RL hyper-parameters, and metrics."""

if __name__ == "__main__":
    print("This is not the main application, run 'main.py' instead")
    exit(1)

import os, sys, time, threading, math, json
from dataclasses import dataclass, field
from typing import Deque
from collections import deque

IS_INTERACTIVE = sys.stdin.isatty()
RESET_METRICS = False
FORCE_FRESH_MODEL = False

MODEL_DIR = "models"
LATEST_MODEL_PATH = f"{MODEL_DIR}/robotron_model_latest.pt"
SETTINGS_PATH = f"{MODEL_DIR}/game_settings.json"


def _default_webrtc_ice_servers() -> list[dict]:
    """Built-in ICE defaults for dashboard WebRTC preview."""
    return [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": [
                "turn:turn.example.com:3478?transport=udp",
                "turn:turn.example.com:3478?transport=tcp",
            ],
            "username": "robotron",
            "credential": "b7K2q9VxM4pN8tR1yL6cZ3wH5dF0sJ",
        },
    ]


def _parse_webrtc_ice_servers_env() -> list[dict]:
    """Parse ROBOTRON_WEBRTC_ICE_SERVERS JSON env override, else defaults."""
    raw = (os.getenv("ROBOTRON_WEBRTC_ICE_SERVERS") or "").strip()
    if not raw:
        return _default_webrtc_ice_servers()
    try:
        data = json.loads(raw)
    except Exception:
        return _default_webrtc_ice_servers()
    if not isinstance(data, list):
        return _default_webrtc_ice_servers()
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        urls = item.get("urls")
        if isinstance(urls, str):
            urls = [urls]
        if not isinstance(urls, list) or not urls:
            continue
        urls_norm = [str(u).strip() for u in urls if isinstance(u, str) and str(u).strip()]
        if not urls_norm:
            continue
        ent = {"urls": urls_norm}
        if isinstance(item.get("username"), str):
            ent["username"] = str(item.get("username"))
        if isinstance(item.get("credential"), str):
            ent["credential"] = str(item.get("credential"))
        out.append(ent)
    return out if out else _default_webrtc_ice_servers()


# Dashboard WebRTC ICE server list (TURN/STUN). Env override takes precedence.
WEBRTC_ICE_SERVERS = _parse_webrtc_ice_servers_env()

# ---------------------------------------------------------------------------
@dataclass
class ServerConfigData:
    host: str = "0.0.0.0"
    port: int = 9998
    max_clients: int = 36
    # State layout:
    #   5 core (alive, score/1e6, replay/100, lasers/9, wave/40)
    #   + 2 player position (norm_pos_x, norm_pos_y: 0..1 over playfield)
    #   + 2 player velocity (frame-delta dx/range, dy/range: ~[-1,+1])
    #   + 50 ELIST enemy state bytes (/255)
    #   + 9 per-type entity categories, each: 1 occupancy + N slots × 4 features
    #   = 9 + 50 + 585 = 644 floats
    #
    # Per-slot features: present, dx, dy, distance_norm
    # dx/dy are player-relative (entity - player), normalised over playfield range.
    # Type is implicit in category position (no type feature needed).
    # Slots sorted by distance to player (nearest first).
    #
    # Entity categories (order matches Lua ENTITY_CATEGORIES):
    #   0. grunt      (40 slots) - grunts                          peak 80
    #   1. hulk       (16 slots) - indestructible hulks             peak 25
    #   2. brain      (16 slots) - brains                          peak 25
    #   3. tank       ( 8 slots) - tanks (growing + full)          peak ~14
    #   4. spawner    ( 8 slots) - circles, squares/quarks         peak 14
    #   5. enforcer   (12 slots) - enforcers                       peak ~10
    #   6. projectile (12 slots) - sparks, shells, cruise, progs
    #   7. human      (16 slots) - mom, dad, kid                   peak 30
    #   8. electrode  (16 slots) - electrodes/posts                peak 25
    params_count: int = 644

SERVER_CONFIG = ServerConfigData()

# ---------------------------------------------------------------------------
@dataclass
class RLConfigData:
    # ── state / action ──────────────────────────────────────────────────
    # Base per-frame state from Lua wire protocol (644 floats).
    base_state_size: int = SERVER_CONFIG.params_count
    # Temporal context window fed to the network.
    frame_stack: int = 4
    # Effective model input width after stacking.
    state_size: int = SERVER_CONFIG.params_count * 4

    # Factored action space for Robotron dual sticks:
    #   movement_direction (0..7 directions, 8 = idle/no-move) × firing_direction (0..7)
    #   = 9 × 8 = 72 joint actions
    num_move_actions: int = 9
    num_fire_actions: int = 8
    # Decode greedy action from the SAME joint Q head used for training.
    # Factored greedy (axis-wise argmax) can pick a low-value pair that is
    # not the joint argmax action actually optimized by C51.
    factored_greedy_action: bool = False

    @property
    def num_joint_actions(self) -> int:
        return self.num_move_actions * self.num_fire_actions

    # ── network architecture ────────────────────────────────────────────
    trunk_hidden: int = 512
    trunk_layers: int = 3
    use_layer_norm: bool = True
    dropout: float = 0.0

    # Object-slot self-attention over 9 typed categories = 144 tokens.
    use_enemy_attention: bool = True
    # Per-type entity categories: name, slots — matches Lua ENTITY_CATEGORIES.
    entity_categories: list = field(default_factory=lambda: [
        ("grunt",      40),
        ("hulk",       16),
        ("brain",      16),
        ("tank",        8),
        ("spawner",     8),
        ("enforcer",   12),
        ("projectile", 12),
        ("human",      16),
        ("electrode",  16),
    ])
    object_slots: int = 144             # total slots across all 9 categories
    # Token features: dx, dy, dist, ddx, ddy, category_id_norm, present
    object_token_features: int = 7
    slot_state_features: int = 4        # present, dx, dy, dist (in state vector)
    attn_heads: int = 8
    attn_dim: int = 128

    # Distributional C51
    # Robotron per-frame rewards: grunt=100, brain=500, human rescue=1000-5000.
    # With obj_reward_scale=0.02 and reward_clip=100, a 5000-pt rescue scales
    # to 100 and passes through unclipped.  Widened support accommodates
    # n-step=12 returns (worst-case ~1136).
    use_distributional: bool = True
    num_atoms: int = 51
    v_min: float = -1200.0
    v_max: float = 1200.0

    use_dueling: bool = True

    # ── training ────────────────────────────────────────────────────────
    batch_size: int = 512
    lr: float = 1e-4
    lr_min: float = 4e-5
    lr_warmup_steps: int = 5_000
    lr_cosine_period: int = 3_000_000       # Longer period to prevent destructive restarts
    lr_use_restarts: bool = True           # Periodic warm restarts to escape plateaus
    gamma: float = 0.99
    n_step: int = 12
    max_samples_per_frame: float = 16

    # Replay (PER with proportional priorities)
    memory_size: int = 20_000_000
    # True = keep replay arrays as persistent np.memmap files and only save
    # compact metadata/priorities on checkpoint (fast restart/save path).
    replay_use_memmap_storage: bool = True
    # Empty string means derive from latest checkpoint path (<model>_replay).
    replay_memmap_dir: str = ""
    priority_alpha: float = 0.7
    priority_beta_start: float = 0.4
    priority_beta_frames: int = 10_000_000
    priority_eps: float = 1e-6
    per_new_priority_cap_multiplier: float = 3.0  # Cap new-entry priority vs current mean to reduce recency runaway
    # Delay training until replay has enough diversity for stable updates.
    min_replay_to_train: int = 25_000

    # Target network (periodic hard sync; moderate refresh for stable learning)
    target_update_period: int = 2_500
    target_tau: float = 1.0

    # Gradient
    grad_clip_norm: float = 5.0            # Tighter clipping dampens large updates

    # ── exploration ─────────────────────────────────────────────────────
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay_frames: int = 8_000_000
    # Manual epsilon pulse (fired with P key, runs for N frames then auto-stops).
    manual_pulse_epsilon: float = 0.25
    manual_pulse_duration_frames: int = 750_000
    epsilon: float = 1.0

    # Expert guidance
    expert_ratio_start: float = 0.80
    expert_ratio_end: float = 0.00
    expert_ratio_decay_frames: int = 10_000_000
    expert_ratio: float = 0.80
    # No special zoom handling for Robotron; keep multipliers neutral.
    expert_ratio_zoom_multiplier: float = 1.0
    expert_ratio_zoom_gamestate: int = 0x00
    epsilon_zoom_multiplier: float = 1.0

    # Moderate BC anchor from heuristic expert transitions.
    expert_bc_weight: float = 0.2
    expert_bc_decay_start: int = 500_000
    expert_bc_decay_frames: int = 4_000_000
    expert_bc_min_weight: float = 0.01

    # ── reward ──────────────────────────────────────────────────────────
    # Scale objective rewards to fit C51 support while keeping TD targets stable.
    # Human rescue (5000) -> 100; no longer clipped.  reward_clip=100
    # accommodates the full rescue signal.
    obj_reward_scale: float = 0.02
    point_reward_scale: float = 1.0 / obj_reward_scale
    subj_reward_scale: float = 0.0025
    reward_clip: float = 100.0
    death_reward_clip: float = 100.0

    # ── death attribution ───────────────────────────────────────────────
    death_priority_boost: float = 5.0      # Lower terminal boost to reduce over-focusing on death tails
    pre_death_lookback: int = 120          # Boost priorities of N frames before each death
    pre_death_priority_boost: float = 2.0  # Multiplicative boost for pre-death frames

    # ── inference ───────────────────────────────────────────────────────
    use_separate_inference_model: bool = True
    # Keep inference on GPU when available; CPU inference can become a bottleneck
    # at higher frame rates even with low overall system utilization.
    inference_on_cpu: bool = False
    # Device placement (CUDA only): useful on multi-GPU hosts.
    train_cuda_device_index: int = 0
    inference_cuda_device_index: int = 0
    inference_sync_steps: int = 100
    # Micro-batch inference requests across clients to increase GPU work per launch.
    inference_batching_enabled: bool = True
    inference_batch_max_size: int = 128
    inference_batch_wait_ms: float = 1.0
    inference_request_timeout_ms: float = 50.0

    # ── background training ─────────────────────────────────────────────
    training_steps_per_cycle: int = 16
    save_interval: int = 10_000

    enable_amp: bool = True

    def __post_init__(self):
        self.frame_stack = max(1, int(self.frame_stack))
        self.base_state_size = int(self.base_state_size)
        self.state_size = self.base_state_size * self.frame_stack


RL_CONFIG = RLConfigData()

# ---------------------------------------------------------------------------
#  Game Settings (shared between dashboard, socket server, and LUA clients)
# ---------------------------------------------------------------------------
# Legacy dashboard list retained for compatibility with existing UI controls.
# Robotron level-select mapping is not wired yet, so expose a simple placeholder range.
ROBOTRON_SELECTABLE_LEVELS = list(range(1, 82))

class GameSettings:
    """Thread-safe container for operator-adjustable game settings."""
    def __init__(self):
        self._lock = threading.Lock()
        self._start_advanced: bool = False
        self._start_level_min: int = 1
        self._epsilon_pct: int = -1   # -1 = auto (follow decay), 0-100 = manual override %
        self._expert_pct: int = -1    # -1 = auto (follow decay), 0-100 = manual override %
        self._auto_curriculum: bool = False

    @property
    def start_advanced(self) -> bool:
        with self._lock:
            return self._start_advanced

    @start_advanced.setter
    def start_advanced(self, value: bool):
        with self._lock:
            self._start_advanced = bool(value)

    @property
    def start_level_min(self) -> int:
        with self._lock:
            return self._start_level_min

    @start_level_min.setter
    def start_level_min(self, value: int):
        with self._lock:
            self._start_level_min = max(1, min(81, int(value)))

    @property
    def epsilon_pct(self) -> int:
        with self._lock:
            return self._epsilon_pct

    @epsilon_pct.setter
    def epsilon_pct(self, value: int):
        with self._lock:
            self._epsilon_pct = max(-1, min(100, int(value)))

    @property
    def expert_pct(self) -> int:
        with self._lock:
            return self._expert_pct

    @expert_pct.setter
    def expert_pct(self, value: int):
        with self._lock:
            self._expert_pct = max(-1, min(100, int(value)))

    @property
    def auto_curriculum(self) -> bool:
        with self._lock:
            return self._auto_curriculum

    @auto_curriculum.setter
    def auto_curriculum(self, value: bool):
        with self._lock:
            self._auto_curriculum = bool(value)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "start_advanced": self._start_advanced,
                "start_level_min": self._start_level_min,
                "epsilon_pct": self._epsilon_pct,
                "expert_pct": self._expert_pct,
                "auto_curriculum": self._auto_curriculum,
            }

    def reset(self) -> None:
        """Restore all settings to initial defaults (fresh-start)."""
        with self._lock:
            self._start_advanced = False
            self._start_level_min = 1
            self._epsilon_pct = -1
            self._expert_pct = -1
            self._auto_curriculum = False

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: str = SETTINGS_PATH) -> None:
        """Write current settings to a JSON file."""
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            data = self.snapshot()
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
        except Exception:
            pass  # best-effort; don't crash the server

    def load(self, path: str = SETTINGS_PATH) -> None:
        """Restore settings from a JSON file if it exists."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            with self._lock:
                if "start_advanced" in data:
                    self._start_advanced = bool(data["start_advanced"])
                if "start_level_min" in data:
                    self._start_level_min = max(1, min(81, int(data["start_level_min"])))
                if "epsilon_pct" in data:
                    self._epsilon_pct = max(-1, min(100, int(data["epsilon_pct"])))
                if "expert_pct" in data:
                    self._expert_pct = max(-1, min(100, int(data["expert_pct"])))
                if "auto_curriculum" in data:
                    self._auto_curriculum = bool(data["auto_curriculum"])
        except FileNotFoundError:
            pass  # first run — use defaults
        except Exception:
            pass  # corrupted file — use defaults

game_settings = GameSettings()
game_settings.load()

# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------
@dataclass
class MetricsData:
    frame_count: int = 0
    total_controls: int = 0
    total_training_steps: int = 0
    memory_buffer_size: int = 0
    client_count: int = 0
    web_client_count: int = 0

    epsilon: float = RL_CONFIG.epsilon_start
    expert_ratio: float = RL_CONFIG.expert_ratio_start

    episode_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    dqn_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    expert_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    subj_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    obj_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    losses: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    fps: float = 0.0
    frames_last_second: int = 0
    last_fps_time: float = 0.0

    # Interval accumulators (reset each display row)
    loss_sum_interval: float = 0.0
    loss_count_interval: int = 0
    agree_sum_interval: float = 0.0
    agree_count_interval: int = 0
    reward_sum_interval: float = 0.0
    reward_count_interval: int = 0
    reward_sum_interval_dqn: float = 0.0
    reward_count_interval_dqn: int = 0
    reward_sum_interval_subj: float = 0.0
    reward_count_interval_subj: int = 0
    reward_sum_interval_obj: float = 0.0
    reward_count_interval_obj: int = 0
    training_steps_interval: int = 0
    frames_count_interval: int = 0
    episode_length_sum_interval: int = 0
    episode_length_count_interval: int = 0
    level_sum_interval: float = 0.0
    level_count_interval: int = 0

    total_inference_time: float = 0.0
    total_inference_requests: int = 0

    last_grad_norm: float = 0.0
    last_loss: float = 0.0
    last_q_mean: float = 0.0
    last_bc_loss: float = 0.0
    last_priority_mean: float = 0.0
    last_agreement: float = 0.0

    average_level: float = 0.0
    peak_level: int = 0
    peak_episode_reward: float = 0.0
    peak_game_score: int = 0
    game_scores: Deque[int] = field(default_factory=lambda: deque(maxlen=100))
    avg_game_score: float = 0.0
    total_games_played: int = 0
    episodes_this_run: int = 0
    last_target_update_step: int = 0
    last_target_update_time: float = 0.0
    loaded_frame_count: int = 0
    game_preview_seq: int = 0
    game_preview_client_id: int = -1
    game_preview_width: int = 0
    game_preview_height: int = 0
    game_preview_format: str = ""
    game_preview_data_b64: str = ""
    game_preview_updated_ts: float = 0.0
    game_preview_source_format: str = ""
    game_preview_encoded_bytes: int = 0
    game_preview_raw_bytes: int = 0
    game_preview_compression_ratio: float = 1.0
    game_preview_fps: float = 0.0

    # UI toggles
    override_expert: bool = False
    expert_mode: bool = False
    manual_expert_override: bool = False
    override_epsilon: bool = False
    manual_epsilon_override: bool = False
    manual_pulse_active: bool = False
    manual_pulse_frames_remaining: int = 0
    training_enabled: bool = True
    verbose_mode: bool = False
    saved_expert_ratio: float = RL_CONFIG.expert_ratio_start

    global_server: object = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    # ── helpers ─────────────────────────────────────────────────────────
    def update_frame_count(self, delta: int = 1):
        with self.lock:
            d = max(0, int(delta))
            if d <= 0:
                return
            self.frame_count += d
            self.frames_count_interval += d
            self.frames_last_second += d
            now = time.time()
            if self.last_fps_time == 0:
                self.last_fps_time = now
            elapsed = now - self.last_fps_time
            if elapsed >= 1.0:
                self.fps = self.frames_last_second / elapsed
                self.frames_last_second = 0
                self.last_fps_time = now

    def get_fps(self) -> float:
        """Return current FPS, decaying to 0 if no frames arrive for >2s."""
        with self.lock:
            if self.last_fps_time > 0:
                stale = time.time() - self.last_fps_time
                if stale >= 2.0:
                    self.fps = 0.0
            return float(self.fps)

    def get_epsilon(self):
        with self.lock:
            return float(self.epsilon)

    def get_effective_epsilon(self) -> float:
        with self.lock:
            ep = game_settings.epsilon_pct
            if ep >= 0:
                return ep / 100.0
            return 0.0 if self.override_epsilon else float(self.epsilon)

    @staticmethod
    def _natural_epsilon_for_frame(frame_count: int) -> float:
        progress = min(1.0, frame_count / max(1, RL_CONFIG.epsilon_decay_frames))
        return RL_CONFIG.epsilon_start + progress * (RL_CONFIG.epsilon_end - RL_CONFIG.epsilon_start)

    def update_epsilon(self):
        with self.lock:
            if self.manual_epsilon_override:
                return self.epsilon
            base = self._natural_epsilon_for_frame(int(self.frame_count))
            if self.manual_pulse_active:
                self.manual_pulse_frames_remaining -= 1
                if self.manual_pulse_frames_remaining <= 0:
                    self.manual_pulse_active = False
                    self.manual_pulse_frames_remaining = 0
                    self.epsilon = base
                else:
                    self.epsilon = max(base, float(RL_CONFIG.manual_pulse_epsilon))
            else:
                self.epsilon = base
            return self.epsilon

    def get_expert_ratio(self):
        with self.lock:
            if self.override_expert:
                return 0.0
            xp = game_settings.expert_pct
            if xp >= 0:
                return xp / 100.0
            return float(self.expert_ratio)

    def update_expert_ratio(self):
        with self.lock:
            if self.expert_mode or self.override_expert or self.manual_expert_override:
                return self.expert_ratio
            progress = min(1.0, self.frame_count / max(1, RL_CONFIG.expert_ratio_decay_frames))
            self.expert_ratio = RL_CONFIG.expert_ratio_start + progress * (RL_CONFIG.expert_ratio_end - RL_CONFIG.expert_ratio_start)
            return self.expert_ratio

    def add_episode_reward(self, total, dqn, expert, subj=None, obj=None, length=0):
        with self.lock:
            self.episodes_this_run += 1
            self.episode_rewards.append(float(total))
            self.dqn_rewards.append(float(dqn))
            self.expert_rewards.append(float(expert))
            if subj is not None:
                self.subj_rewards.append(float(subj))
            if obj is not None:
                self.obj_rewards.append(float(obj))
            self.reward_sum_interval += float(total)
            self.reward_count_interval += 1
            self.reward_sum_interval_dqn += float(dqn)
            self.reward_count_interval_dqn += 1
            if subj is not None:
                self.reward_sum_interval_subj += float(subj)
                self.reward_count_interval_subj += 1
            if obj is not None:
                self.reward_sum_interval_obj += float(obj)
                self.reward_count_interval_obj += 1
            if length > 0:
                self.episode_length_sum_interval += length
                self.episode_length_count_interval += 1
            if float(total) > self.peak_episode_reward:
                self.peak_episode_reward = float(total)

    def add_game_score(self, score: int):
        """Record a completed full-game score (all lives) into rolling window."""
        with self.lock:
            self.game_scores.append(int(score))
            self.total_games_played += 1
            if self.game_scores:
                self.avg_game_score = float(sum(self.game_scores)) / len(self.game_scores)

    def increment_total_controls(self):
        with self.lock:
            self.total_controls += 1

    def update_game_state(self, enemy_seg, open_level):
        pass  # compat stub

    def add_inference_time(self, t: float):
        with self.lock:
            self.total_inference_time += t
            self.total_inference_requests += 1

    # ── UI toggle methods ───────────────────────────────────────────────
    def toggle_override(self, kb=None):
        with self.lock:
            self.override_expert = not self.override_expert
            if self.override_expert:
                self.saved_expert_ratio = self.expert_ratio
                self.expert_ratio = 0.0
            else:
                self.expert_ratio = self.saved_expert_ratio
            game_settings.expert_pct = -1   # keyboard wins → clear dashboard

    def toggle_expert_mode(self, kb=None):
        with self.lock:
            self.expert_mode = not self.expert_mode
            if self.expert_mode:
                self.saved_expert_ratio = self.expert_ratio
                self.expert_ratio = 1.0
            else:
                self.expert_ratio = self.saved_expert_ratio
            game_settings.expert_pct = -1   # keyboard wins → clear dashboard

    def toggle_training_mode(self, kb=None):
        with self.lock:
            self.training_enabled = not self.training_enabled

    def toggle_epsilon_override(self, kb=None):
        with self.lock:
            self.override_epsilon = not self.override_epsilon
            game_settings.epsilon_pct = -1   # keyboard wins → clear dashboard

    def toggle_verbose_mode(self, kb=None):
        with self.lock:
            self.verbose_mode = not self.verbose_mode

    def toggle_epsilon_pulse(self, kb=None):
        """Fire or cancel the manual epsilon pulse."""
        with self.lock:
            if self.manual_pulse_active:
                # Cancel the running pulse
                self.manual_pulse_active = False
                self.manual_pulse_frames_remaining = 0
            else:
                # Start a new pulse
                self.manual_pulse_active = True
                self.manual_pulse_frames_remaining = int(RL_CONFIG.manual_pulse_duration_frames)
            game_settings.epsilon_pct = -1   # keyboard wins → clear dashboard

    def increase_expert_ratio(self, kb=None):
        with self.lock:
            p = int(self.expert_ratio * 100)
            p = min(100, p + (1 if p < 10 else 5))
            self.expert_ratio = p / 100.0
            self.manual_expert_override = True
            game_settings.expert_pct = -1   # keyboard wins → clear dashboard

    def decrease_expert_ratio(self, kb=None):
        with self.lock:
            p = int(self.expert_ratio * 100)
            p = max(0, p - (1 if p <= 10 else 5))
            self.expert_ratio = p / 100.0
            self.manual_expert_override = True
            game_settings.expert_pct = -1   # keyboard wins → clear dashboard

    def restore_natural_expert_ratio(self, kb=None):
        with self.lock:
            self.manual_expert_override = False
            progress = min(1.0, self.frame_count / max(1, RL_CONFIG.expert_ratio_decay_frames))
            self.expert_ratio = RL_CONFIG.expert_ratio_start + progress * (RL_CONFIG.expert_ratio_end - RL_CONFIG.expert_ratio_start)
            game_settings.expert_pct = -1   # keyboard wins → clear dashboard

    def increase_epsilon(self, kb=None):
        with self.lock:
            p = int(self.epsilon * 100)
            p = min(100, p + (1 if p < 10 else 5))
            self.epsilon = p / 100.0
            self.manual_epsilon_override = True
            game_settings.epsilon_pct = -1   # keyboard wins → clear dashboard

    def decrease_epsilon(self, kb=None):
        with self.lock:
            p = int(self.epsilon * 100)
            p = max(0, p - (1 if p <= 10 else 5))
            self.epsilon = p / 100.0
            self.manual_epsilon_override = True
            game_settings.epsilon_pct = -1   # keyboard wins → clear dashboard

    def restore_natural_epsilon(self, kb=None):
        with self.lock:
            self.manual_epsilon_override = False
            self.epsilon = self._natural_epsilon_for_frame(int(self.frame_count))
            game_settings.epsilon_pct = -1   # keyboard wins → clear dashboard


metrics = MetricsData()


# ── PlateauPulser stub ──────────────────────────────────────────────────────
# Provides the interface expected by the dashboard, backed by our manual pulse.
class PlateauPulser:
    WATCHING   = "watching"
    PULSING    = "pulsing"
    RECOVERING = "recovering"

    @property
    def state(self) -> str:
        return self.PULSING if metrics.manual_pulse_active else self.WATCHING

    @property
    def total_pulses(self) -> int:
        return 0                       # manual pulse doesn't track lifetime count

    pulse_start_frame: int = 0
    pulse_end_frame: int = 0
    cooldown_multiplier: float = 1.0


plateau_pulser = PlateauPulser()
