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
from pathlib import Path
from dataclasses import dataclass, field
from typing import Deque
from collections import deque

IS_INTERACTIVE = sys.stdin.isatty()
RESET_METRICS = False
FORCE_FRESH_MODEL = False

_CONFIG_DIR = Path(__file__).resolve().parent
_ROBOTRON_DIR = _CONFIG_DIR.parent
_REPO_ROOT = _ROBOTRON_DIR.parent


def _score_model_dir(path: Path) -> tuple[int, float]:
    """Rank model directories by completeness, then recency."""
    model_path = path / "robotron_model_latest.pt"
    replay_meta = path / "robotron_model_latest_replay" / "_meta.npy"
    replay_pri = path / "robotron_model_latest_replay" / "priorities.npy"
    settings_path = path / "game_settings.json"
    existing = [p for p in (model_path, replay_meta, replay_pri, settings_path) if p.exists()]
    if not existing:
        return (0, 0.0)
    completeness = sum(int(p.exists()) for p in (model_path, replay_meta, replay_pri))
    newest_mtime = max(p.stat().st_mtime for p in existing)
    return (completeness, newest_mtime)


def _resolve_model_dir() -> Path:
    """Pick a stable model directory regardless of launch cwd.

    Historically this project used a relative "models" path, which caused
    separate checkpoint trees to appear under different working directories.
    We now scan the known legacy locations and consistently pick the best one.
    """
    env_dir = (os.getenv("ROBOTRON_MODEL_DIR") or "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    candidates = []
    for cand in (
        _ROBOTRON_DIR / "models",
        _REPO_ROOT / "models",
        _CONFIG_DIR / "models",
    ):
        if cand not in candidates:
            candidates.append(cand)

    populated = [cand for cand in candidates if _score_model_dir(cand)[0] > 0]
    if populated:
        return max(populated, key=_score_model_dir)
    return _ROBOTRON_DIR / "models"


MODEL_DIR = str(_resolve_model_dir())
LATEST_MODEL_PATH = str(Path(MODEL_DIR) / "robotron_model_latest.pt")
SETTINGS_PATH = str(Path(MODEL_DIR) / "game_settings.json")


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

LEGACY_ENTITY_CATEGORIES: tuple[tuple[str, int], ...] = (
    ("grunt", 40),
    ("hulk", 16),
    ("brain", 16),
    ("tank", 8),
    ("spawner", 8),
    ("enforcer", 12),
    ("projectile", 12),
    ("human", 16),
    ("electrode", 16),
)
LEGACY_SLOT_STATE_FEATURES = 8
LEGACY_ELIST_FEATURES = 50
LEGACY_CORE_FEATURES = 9
LEGACY_TOTAL_SLOTS = sum(int(slots) for _name, slots in LEGACY_ENTITY_CATEGORIES)
LEGACY_PARAMS_COUNT = LEGACY_CORE_FEATURES + LEGACY_ELIST_FEATURES + len(LEGACY_ENTITY_CATEGORIES) + (LEGACY_TOTAL_SLOTS * LEGACY_SLOT_STATE_FEATURES)

# ---------------------------------------------------------------------------
@dataclass
class ServerConfigData:
    host: str = "0.0.0.0"
    port: int = 9998
    max_clients: int = 36
    # State layout:
    #   5 core (alive, score, replay, lasers, wave)
    #   + 2 player position
    #   + 2 player velocity
    #   + 50 ELIST bytes
    #   + 9 typed entity buckets, each: 1 occupancy + N slots x 8 features
    #     (present, dx, dy, dist, hit_w, hit_h, vx, vy)
    #   = 1220 floats total
    params_count: int = LEGACY_PARAMS_COUNT

SERVER_CONFIG = ServerConfigData()

# ---------------------------------------------------------------------------
@dataclass
class RLConfigData:
    # ── state / action ──────────────────────────────────────────────────
    # Base per-frame state from Lua wire protocol.
    base_state_size: int = SERVER_CONFIG.params_count
    # Structured slot encoder: all stacked frames are consumed directly.
    frame_stack: int = 4
    # Effective model input width after stacking.
    state_size: int = SERVER_CONFIG.params_count * 4

    # Factored action space for Robotron dual sticks:
    #   movement_direction (0..7 directions, 8 = idle/no-move) × firing_direction (0..7)
    #   = 9 × 8 = 72 joint actions
    num_move_actions: int = 9
    num_fire_actions: int = 8
    # Learn separate move/fire action heads and reconstruct the joint table
    # from them. This matches Robotron's dual-stick structure better than a
    # monolithic 72-way action head.
    factorized_action_heads: bool = True
    # With factorized move/fire heads, axis-wise greedy decode matches the
    # optimized additive action head, so we enable it by default.
    factored_greedy_action: bool = True

    @property
    def num_joint_actions(self) -> int:
        return self.num_move_actions * self.num_fire_actions

    # ── observation layout / network architecture ──────────────────────
    # These hybrid-shape fields are retained so expert/test tooling can still
    # understand the newer 2212-float format when synthetic states use it.
    global_feature_count: int = 100
    grid_width: int = 12
    grid_height: int = 12
    grid_channels: int = 8
    object_token_count: int = 64
    # Token features:
    #   present, dx, dy, vx, vy, dist, dir_x, dir_y, threat,
    #   size_x, size_y, category_norm, is_human, is_dangerous, approach
    object_token_features: int = 15

    # MLP trunk over the full stacked flat state. When mlp_with_attention is
    # enabled, a parallel object-set branch is fused in before the C51 head.
    pure_mlp: bool = True
    mlp_with_attention: bool = True
    mlp_hidden_layers: list = field(default_factory=lambda: [1024, 512])
    mlp_output_dim: int = 256
    trunk_hidden: int = 256
    trunk_layers: int = 2
    use_layer_norm: bool = True
    dropout: float = 0.0

    # Structured learner: compact slot encoding over typed entity buckets.
    use_enemy_attention: bool = True
    entity_categories: list = field(default_factory=lambda: list(LEGACY_ENTITY_CATEGORIES))
    object_slots: int = LEGACY_TOTAL_SLOTS
    slot_state_features: int = LEGACY_SLOT_STATE_FEATURES
    # Per-object set-encoder features for the compact legacy slot state:
    #   dx, dy, vx, vy, dist, hit_w, hit_h, category_norm, is_human, is_dangerous
    legacy_slot_token_features: int = 10
    attn_heads: int = 4
    attn_dim: int = 96
    attn_layers: int = 1
    # False = attention only on the most recent frame; the flat MLP still sees
    # all stacked frames. This keeps the attention branch focused on "what
    # matters now" while preserving temporal context in the dense path.
    # Changing this alters the entity_proj input width, so it is an
    # architecture change and requires a fresh checkpoint.
    attn_all_frames: bool = False
    grid_hidden_channels: int = 32
    global_hidden: int = 128
    category_summary_dim: int = 48
    entity_hidden: int = 192

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
    lr_cosine_period: int = 250_000       # Longer period to prevent destructive restarts
    lr_use_restarts: bool = True           # Periodic warm restarts to escape plateaus
    gamma: float = 0.99
    n_step: int = 12
    max_samples_per_frame: float = 16

    # Replay (PER with proportional priorities)
    memory_size: int = 10_000_000
    # True = keep replay arrays as persistent np.memmap files and only save
    # compact metadata/priorities on checkpoint (fast restart/save path).
    replay_use_memmap_storage: bool = True
    # Empty string means derive from latest checkpoint path (<model>_replay).
    replay_memmap_dir: str = ""
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.55
    priority_beta_frames: int = 5_000_000
    priority_eps: float = 1e-6
    per_new_priority_cap_multiplier: float = 2.0  # Cap new-entry priority vs current mean to reduce recency runaway
    # Delay training until replay has enough diversity for stable updates.
    min_replay_to_train: int = 25_000
    # Wave-aware replay quotas: preserve frontier/high-wave DQN experience so
    # it does not get drowned by abundant easy-wave traffic.
    replay_wave_sampling_enabled: bool = True
    replay_wave_frontier_frac: float = 0.35
    replay_wave_high_frac: float = 0.15
    replay_wave_frontier_margin: int = 1
    replay_wave_high_offset: int = 2
    replay_wave_candidate_multiplier: int = 10
    replay_wave_min_frontier: int = 8
    # Guaranteed floor for expert transitions in a sampled batch when enough
    # expert candidates exist in replay. These are reserved before the DQN-only
    # wave quotas are filled.
    replay_expert_min_frac: float = 0.25
    # Cap, not floor: expert transitions may be below this fraction only if
    # replay does not contain enough expert candidates.
    replay_expert_max_frac: float = 0.25

    # Target network (periodic hard sync; moderate refresh for stable learning)
    target_update_period: int = 2_500
    target_tau: float = 1.0

    # Gradient
    grad_clip_norm: float = 5.0            # Tighter clipping dampens large updates

    # ── exploration ─────────────────────────────────────────────────────
    epsilon_start: float = 1.0
    epsilon_end: float = 0.00
    epsilon_decay_frames: int = 10_000_000
    # Manual epsilon pulse (fired with P key, runs for N frames then auto-stops).
    manual_pulse_epsilon: float = 0.25
    manual_pulse_duration_frames: int = 750_000
    # Automatic plateau pulser: temporarily raises epsilon and forces
    # frontier starts when DQN reward and reached-wave metrics stop improving.
    plateau_pulse_enabled: bool = False
    plateau_pulse_epsilon: float = 0.14
    plateau_pulse_frames: int = 300_000
    plateau_confirm_frames: int = 1_000_000
    plateau_cooldown_frames: int = 1_500_000
    plateau_min_frame: int = 4_000_000
    plateau_reward_delta: float = 0.75
    plateau_level_delta: float = 0.25
    plateau_curriculum_wave_offset: int = 1
    epsilon: float = 1.0

    # Expert guidance
    expert_ratio_start: float = 0.99
    expert_ratio_end: float = 0.20
    expert_ratio_decay_frames: int = 30_000_000
    expert_ratio: float = 0.99

    # No special zoom handling for Robotron; keep multipliers neutral.
    expert_ratio_zoom_multiplier: float = 1.0
    expert_ratio_zoom_gamestate: int = 0x00
    epsilon_zoom_multiplier: float = 1.0

    # Stronger BC anchor from heuristic expert transitions; keep it alive well
    # into training so the learner does not lose competent aiming/movement
    # before it can stand on its own.
    expert_bc_weight: float = 0.4
    expert_bc_decay_start: int = 10_000_000
    # Decays from 10M to 30M frames.
    expert_bc_decay_frames: int = 50_000_000
    expert_bc_min_weight: float = 0.05
    expert_profile: str = "simple"
    expert_hold_fire_for_last_enemy_rescue: bool = True

    # ── reward ──────────────────────────────────────────────────────────
    # Scale objective rewards to fit C51 support while keeping TD targets stable.
    # At 0.03, a 5000-point rescue scales to 150 and is clipped to the
    # reward_clip of 100. This intentionally boosts ordinary kill signal
    # versus the earlier 0.02 setting, while still bounding very large events.
    obj_reward_scale: float = 0.03
    point_reward_scale: float = 1.0 / obj_reward_scale
    subj_reward_scale: float = 0.001
    reward_clip: float = 100.0
    death_reward_clip: float = 60.0

    # ── death attribution ───────────────────────────────────────────────
    death_priority_boost: float = 5.0      # Lower terminal boost to reduce over-focusing on death tails
    pre_death_lookback: int = 120          # Boost priorities of N frames before each death
    pre_death_priority_boost: float = 2.0  # Multiplicative boost for pre-death frames

    # ── level-based priority ─────────────────────────────────────────────
    # Priority multiplier = max(1.0, log10(wave_number) * level_priority_log_scale).
    # At scale=1.0: wave 10 → 1.0×, wave 100 → 2.0×, wave 1000 → 3.0×.
    # At scale=5.0: wave 100 → 10.0×, matching a "10× at level 100" goal.
    # Set to 0.0 to disable.
    level_priority_log_scale: float = 0.0

    # ── inference ───────────────────────────────────────────────────────
    use_separate_inference_model: bool = True
    # Keep inference on GPU when available; CPU inference can become a bottleneck
    # at higher frame rates even with low overall system utilization.
    inference_on_cpu: bool = False
    # Device placement (CUDA only): useful on multi-GPU hosts.
    train_cuda_device_index: int = 0
    inference_cuda_device_index: int = 1
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
        self.point_reward_scale = 1.0 / max(1e-12, float(self.obj_reward_scale))


RL_CONFIG = RLConfigData()

# ---------------------------------------------------------------------------
#  Game Settings (shared between dashboard, socket server, and LUA clients)
# ---------------------------------------------------------------------------
# Legacy dashboard list retained for compatibility with existing UI controls.
# Robotron now accepts direct wave numbers 1..81 for curriculum/start-level control.
ROBOTRON_SELECTABLE_LEVELS = list(range(1, 82))


def compute_robotron_auto_curriculum_level(average_level: float) -> int:
    """Map dashboard average level to the Robotron curriculum start wave."""
    try:
        avg = float(average_level)
    except Exception:
        avg = 1.0
    return max(1, min(81, int(math.floor(avg)) - 1))

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

    # Preview capture control (admin-controlled via dashboard)
    preview_capture_enabled: bool = True
    hud_enabled: bool = False
    last_priority_mean: float = 0.0
    last_agreement: float = 0.0

    average_level: float = 0.0
    peak_level: int = 0
    peak_level_verified: bool = False
    peak_episode_reward: float = 0.0
    peak_game_score: int = 0
    records_reset_seq: int = 0
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
    game_preview_data: bytes = b""
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
            try:
                self.epsilon = max(self.epsilon, float(plateau_pulser.epsilon_floor()))
            except Exception:
                pass
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

    def reset_record_metrics(self) -> int:
        """Clear dashboard/server record-style metrics and return the new reset sequence."""
        with self.lock:
            self.peak_level = 0
            self.peak_level_verified = False
            self.peak_episode_reward = 0.0
            self.peak_game_score = 0
            self.game_scores.clear()
            self.avg_game_score = 0.0
            self.total_games_played = 0
            self.records_reset_seq += 1
            return int(self.records_reset_seq)

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


class PlateauPulser:
    WATCHING = "watching"
    PULSING = "pulsing"
    RECOVERING = "recovering"

    def __init__(self):
        self._lock = threading.Lock()
        self._state = self.WATCHING
        self._total_pulses = 0
        self._stall_start_frame = 0
        self._best_reward = float("-inf")
        self._best_level = 0.0
        self._pulse_target_level = 1
        self.pulse_start_frame = 0
        self.pulse_end_frame = 0
        self.recovery_end_frame = 0
        self.cooldown_multiplier = 1.0

    @property
    def state(self) -> str:
        with self._lock:
            state = self._state
        if state == self.WATCHING and getattr(metrics, "manual_pulse_active", False):
            return self.PULSING
        return state

    @property
    def total_pulses(self) -> int:
        with self._lock:
            return int(self._total_pulses)

    def remaining_frames(self) -> int:
        with self._lock:
            auto_remaining = 0
            if self._state == self.PULSING:
                auto_remaining = max(0, int(self.pulse_end_frame - metrics.frame_count))
        manual_remaining = int(getattr(metrics, "manual_pulse_frames_remaining", 0) or 0)
        return max(auto_remaining, manual_remaining)

    def epsilon_floor(self) -> float:
        with self._lock:
            if self._state != self.PULSING or not bool(getattr(RL_CONFIG, "plateau_pulse_enabled", False)):
                return 0.0
        return max(0.0, min(1.0, float(getattr(RL_CONFIG, "plateau_pulse_epsilon", 0.0))))

    def overlay_game_settings(self, snapshot: dict, average_level: float) -> dict:
        out = dict(snapshot or {})
        with self._lock:
            if self._state != self.PULSING or not bool(getattr(RL_CONFIG, "plateau_pulse_enabled", False)):
                return out
            pulse_level = int(self._pulse_target_level)
        pulse_level = max(
            pulse_level,
            compute_robotron_auto_curriculum_level(float(average_level)) + int(getattr(RL_CONFIG, "plateau_curriculum_wave_offset", 1)),
        )
        pulse_level = max(int(getattr(RL_CONFIG, "replay_wave_min_frontier", 4)), pulse_level)
        out["start_advanced"] = True
        out["start_level_min"] = max(int(out.get("start_level_min", 1) or 1), max(1, min(81, pulse_level)))
        return out

    def _set_watch_baseline(self, frame_count: int, average_level: float, reward_1m: float) -> None:
        self._best_reward = float(reward_1m)
        self._best_level = float(average_level)
        self._stall_start_frame = max(0, int(frame_count))

    def update(self, frame_count: int, average_level: float, dqn100k: float, dqn1m: float, dqn5m: float) -> None:
        del dqn100k, dqn5m  # The trigger currently uses the steadier 1M reward window plus reached-wave frontier.

        cfg = RL_CONFIG
        if not bool(getattr(cfg, "plateau_pulse_enabled", False)):
            with self._lock:
                self._state = self.WATCHING
            return

        fc = max(0, int(frame_count))
        try:
            avg = float(average_level)
        except Exception:
            avg = 0.0
        try:
            reward_1m = float(dqn1m)
        except Exception:
            reward_1m = 0.0
        if not math.isfinite(avg):
            avg = 0.0
        if not math.isfinite(reward_1m):
            reward_1m = 0.0
        reward_delta = max(0.0, float(getattr(cfg, "plateau_reward_delta", 0.0)))
        level_delta = max(0.0, float(getattr(cfg, "plateau_level_delta", 0.0)))

        with self._lock:
            if self._state == self.PULSING:
                if fc >= self.pulse_end_frame:
                    self._state = self.RECOVERING
                    cooldown_frames = int(getattr(cfg, "plateau_cooldown_frames", 0))
                    self.recovery_end_frame = fc + max(1, int(cooldown_frames * self.cooldown_multiplier))
                    self._set_watch_baseline(fc, avg, reward_1m)
                return

            if self._state == self.RECOVERING:
                if reward_1m >= (self._best_reward + reward_delta) or avg >= (self._best_level + level_delta):
                    self.cooldown_multiplier = max(1.0, self.cooldown_multiplier * 0.85)
                    self._set_watch_baseline(fc, avg, reward_1m)
                if fc >= self.recovery_end_frame:
                    self._state = self.WATCHING
                    self._set_watch_baseline(fc, avg, reward_1m)
                return

            if self._stall_start_frame <= 0 or not math.isfinite(self._best_reward):
                self._set_watch_baseline(fc, avg, reward_1m)
                return

            if fc < int(getattr(cfg, "plateau_min_frame", 0)):
                if reward_1m >= (self._best_reward + reward_delta) or avg >= (self._best_level + level_delta):
                    self._set_watch_baseline(fc, avg, reward_1m)
                return

            improved = False
            if reward_1m >= (self._best_reward + reward_delta):
                self._best_reward = reward_1m
                improved = True
            if avg >= (self._best_level + level_delta):
                self._best_level = avg
                improved = True
            if improved:
                self._stall_start_frame = fc
                self.cooldown_multiplier = max(1.0, self.cooldown_multiplier * 0.95)
                return

            if (fc - self._stall_start_frame) < int(getattr(cfg, "plateau_confirm_frames", 0)):
                return

            self._state = self.PULSING
            self._total_pulses += 1
            self.pulse_start_frame = fc
            self.pulse_end_frame = fc + max(1, int(getattr(cfg, "plateau_pulse_frames", 0)))
            frontier_wave = max(1, min(81, int(math.floor(avg)) + int(getattr(cfg, "plateau_curriculum_wave_offset", 1))))
            self._pulse_target_level = max(
                int(getattr(cfg, "replay_wave_min_frontier", 4)),
                frontier_wave,
                compute_robotron_auto_curriculum_level(avg) + int(getattr(cfg, "plateau_curriculum_wave_offset", 1)),
            )
            self.cooldown_multiplier = min(3.0, self.cooldown_multiplier + 0.15)
            self._stall_start_frame = fc


plateau_pulser = PlateauPulser()
