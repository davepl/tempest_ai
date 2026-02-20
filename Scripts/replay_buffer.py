#!/usr/bin/env python3
# ==================================================================================================================
# ||  TEMPEST AI v2 • PRIORITIZED EXPERIENCE REPLAY                                                              ||
# ||  Sum-tree backed proportional PER with per-slot storage.                                                     ||
# ==================================================================================================================
"""Prioritized replay buffer using a sum-tree for O(log N) sampling."""

import os, sys, time, struct, shutil
import numpy as np
import threading

try:
    from config import RL_CONFIG
except ImportError:
    from Scripts.config import RL_CONFIG


class SumTree:
    """Binary sum-tree for efficient proportional sampling in O(log N)."""

    __slots__ = ("capacity", "tree", "data_ptr", "size", "max_priority", "_depth")

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity, dtype=np.float64)
        self.data_ptr = 0
        self.size = 0
        self.max_priority = 1.0
        self._depth = int(np.ceil(np.log2(max(2, self.capacity))))

    def _propagate(self, idx: int):
        parent = idx >> 1
        while parent >= 1:
            self.tree[parent] = self.tree[parent * 2] + self.tree[parent * 2 + 1]
            parent >>= 1

    def total(self) -> float:
        return float(self.tree[1])

    def add(self, priority: float) -> int:
        """Add a new entry and return its data index."""
        idx = self.data_ptr
        tree_idx = idx + self.capacity
        self.tree[tree_idx] = float(priority)
        self._propagate(tree_idx)
        self.data_ptr = (self.data_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        if priority > self.max_priority:
            self.max_priority = float(priority)
        return idx

    def update(self, data_idx: int, priority: float):
        tree_idx = data_idx + self.capacity
        self.tree[tree_idx] = float(priority)
        self._propagate(tree_idx)
        if priority > self.max_priority:
            self.max_priority = float(priority)

    def get(self, value: float) -> int:
        """Sample a data index proportional to priority."""
        idx = 1
        while idx < self.capacity:
            left = idx * 2
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        return idx - self.capacity

    def batch_get(self, values: np.ndarray) -> np.ndarray:
        """Vectorised batch sampling — all queries traverse the tree in lockstep.

        Instead of a Python loop over batch_size items each doing O(log N)
        scalar traversals, this performs log N numpy-vectorised steps.
        """
        n = len(values)
        indices = np.ones(n, dtype=np.int64)
        remaining = values.astype(np.float64, copy=True)
        cap = self.capacity
        for _ in range(self._depth):
            # Mask: True where index is still an internal node
            mask = indices < cap
            if not mask.any():
                break
            # Safe left-child indices (use 0 for already-resolved leaves)
            left = np.where(mask, indices << 1, 0)
            left_vals = self.tree[left]
            go_right = mask & (remaining > left_vals)
            remaining -= left_vals * go_right
            indices = np.where(mask, left + go_right.astype(np.int64), indices)
        return indices - cap

    def batch_update(self, data_indices: np.ndarray, priorities: np.ndarray):
        """Vectorised batch priority update with deduped parent propagation.

        Sets all leaf priorities at once then walks up the tree one level at
        a time, merging duplicate parents with np.unique at each level.
        """
        tree_idx = data_indices.astype(np.int64) + self.capacity
        self.tree[tree_idx] = priorities.astype(np.float64)
        mx = float(priorities.max())
        if mx > self.max_priority:
            self.max_priority = mx
        # Walk parents upward, deduplicating at each level
        parents = np.unique(tree_idx >> 1)
        while len(parents) > 0 and parents[0] >= 1:
            self.tree[parents] = self.tree[parents * 2] + self.tree[parents * 2 + 1]
            parents = np.unique(parents >> 1)
            parents = parents[parents >= 1]

    def priority(self, data_idx: int) -> float:
        return float(self.tree[data_idx + self.capacity])


class PrioritizedReplayBuffer:
    """Proportional PER backed by a SumTree.

    Stores transitions as flat numpy arrays for fast vectorised sampling.
    Thread-safe via a reentrant lock.
    """

    def __init__(self, capacity: int, state_size: int, alpha: float = 0.6):
        self.capacity = int(capacity)
        self.state_size = int(state_size)
        self.alpha = float(alpha)
        self.lock = threading.Lock()

        # Storage arrays
        self.states      = np.zeros((self.capacity, self.state_size), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_size), dtype=np.float32)
        self.actions     = np.zeros(self.capacity, dtype=np.int64)
        self.rewards     = np.zeros(self.capacity, dtype=np.float32)
        self.dones       = np.zeros(self.capacity, dtype=np.float32)
        self.horizons    = np.ones(self.capacity, dtype=np.int32)
        self.is_expert   = np.zeros(self.capacity, dtype=np.uint8)

        self.tree = SumTree(self.capacity)
        self.size = 0
        self._n_expert = 0          # O(1) expert tracking

    @staticmethod
    def _progress_bar(label: str, frac: float, width: int = 28):
        frac_clamped = max(0.0, min(1.0, float(frac)))
        filled = int(round(frac_clamped * width))
        bar = "#" * filled + "-" * (width - filled)
        sys.stdout.write(f"\r{label} [{bar}] {frac_clamped * 100.0:5.1f}%")
        sys.stdout.flush()
        if frac_clamped >= 1.0:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def add(self, state, action: int, reward: float, next_state, done: bool,
            horizon: int = 1, expert: int = 0, priority_hint: float = 0.0):
        with self.lock:
            priority = self.tree.max_priority
            cap_mult = float(getattr(RL_CONFIG, "per_new_priority_cap_multiplier", 0.0))
            mean_pri = 0.0
            if cap_mult > 0.0 and self.size > 0:
                mean_pri = self.tree.total() / max(1, self.size)
                if mean_pri > 0.0:
                    priority = min(priority, mean_pri * cap_mult)
            if priority_hint != 0.0:
                hint_pri = abs(priority_hint) ** self.alpha
                if hint_pri > priority:
                    priority = hint_pri
            if cap_mult > 0.0 and mean_pri > 0.0:
                priority = min(priority, mean_pri * cap_mult)
            priority = max(1e-6, float(priority))
            # If buffer is full, undo the expert flag of the slot being recycled
            if self.tree.size >= self.capacity:
                self._n_expert -= int(self.is_expert[self.tree.data_ptr])
            idx = self.tree.add(priority)
            self.states[idx]      = np.asarray(state, dtype=np.float32)
            self.next_states[idx] = np.asarray(next_state, dtype=np.float32)
            self.actions[idx]     = int(action)
            self.rewards[idx]     = float(reward)
            self.dones[idx]       = 1.0 if done else 0.0
            self.horizons[idx]    = max(1, int(horizon))
            self.is_expert[idx]   = int(expert)
            self._n_expert += int(expert)
            self.size = self.tree.size

    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample a prioritised batch. Returns (states, actions, rewards,
        next_states, dones, horizons, is_expert, indices, weights)."""
        with self.lock:
            if self.size < batch_size:
                return None

            total = self.tree.total()
            if total <= 0:
                return None

            # Stratified sampling — one uniform draw per segment (vectorised)
            segment = total / batch_size
            lows = np.arange(batch_size, dtype=np.float64) * segment
            highs = lows + segment
            values = np.random.uniform(lows, highs)
            indices = self.tree.batch_get(values)
            np.clip(indices, 0, self.size - 1, out=indices)

            # Gather priorities in one vectorised read
            priorities = np.maximum(1e-10, self.tree.tree[indices + self.tree.capacity])

            # Importance-sampling weights
            probs = priorities / total
            weights = (self.size * probs) ** (-beta)
            weights /= weights.max()

            return (
                self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.next_states[indices],
                self.dones[indices],
                self.horizons[indices],
                self.is_expert[indices],
                indices,
                weights.astype(np.float32),
            )

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors (fully vectorised)."""
        with self.lock:
            new_p = (np.abs(td_errors.astype(np.float64)) + 1e-6) ** self.alpha
            self.tree.batch_update(np.asarray(indices, dtype=np.int64), new_p)

    def __len__(self):
        return self.size

    def get_partition_stats(self):
        """Return buffer statistics (O(1) via tracked counter)."""
        with self.lock:
            n_exp = self._n_expert
            n_dqn = self.size - n_exp
            return {
                "total_size": self.size,
                "total_capacity": self.capacity,
                "dqn": n_dqn,
                "expert": n_exp,
                "frac_dqn": n_dqn / max(1, self.size),
                "frac_expert": n_exp / max(1, self.size),
            }

    # ── Persistence ─────────────────────────────────────────────────────

    # Array names in fixed save order — must match between save() and load()
    _ARRAY_NAMES = ("states", "next_states", "actions", "rewards", "dones", "horizons", "is_expert", "priorities")

    def save(self, filepath: str, verbose: bool = True):
        """Save the replay buffer as individual .npy files in a directory (3x faster than npz)."""
        with self.lock:
            if self.size == 0:
                if verbose:
                    print("  Replay buffer is empty — nothing to save.")
                return

            n = self.size
            if verbose:
                print(f"  Saving replay buffer ({n:,} transitions)...")
            t0 = time.time()
            if verbose:
                self._progress_bar("  Replay save", 0.05)

            # Snapshot arrays under lock (slicing is a view, .copy() only on priorities)
            arrays = {
                "states":      self.states[:n],
                "next_states": self.next_states[:n],
                "actions":     self.actions[:n],
                "rewards":     self.rewards[:n],
                "dones":       self.dones[:n],
                "horizons":    self.horizons[:n],
                "is_expert":   self.is_expert[:n],
                "priorities":  self.tree.tree[self.tree.capacity:self.tree.capacity + n].copy(),
            }
            data_ptr = int(self.tree.data_ptr)
            max_priority = float(self.tree.max_priority)

        if verbose:
            self._progress_bar("  Replay save", 0.10)

        # Write to a temp directory, then atomically swap
        save_dir = filepath.replace(".npz", "")  # strip .npz if present
        tmp_dir = save_dir + ".tmp"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)

        try:
            # Save each array as a raw .npy file — no ZIP overhead
            total_arrays = len(self._ARRAY_NAMES)
            for idx, name in enumerate(self._ARRAY_NAMES):
                np.save(os.path.join(tmp_dir, f"{name}.npy"), arrays[name])
                if verbose:
                    frac = 0.10 + 0.75 * ((idx + 1) / total_arrays)
                    self._progress_bar("  Replay save", frac)

            # Save scalar metadata as a small .npy file
            meta = np.array([n, data_ptr, max_priority], dtype=np.float64)
            np.save(os.path.join(tmp_dir, "_meta.npy"), meta)

            # fsync the directory to ensure durability
            try:
                fd = os.open(tmp_dir, os.O_RDONLY)
                os.fsync(fd)
                os.close(fd)
            except OSError:
                pass

            if verbose:
                self._progress_bar("  Replay save", 0.90)

            # Atomic swap: remove old dir, rename tmp into place
            if os.path.isdir(save_dir):
                shutil.rmtree(save_dir, ignore_errors=True)
            # Also clean up legacy .npz if present
            if os.path.isfile(filepath) and filepath.endswith(".npz"):
                try:
                    os.remove(filepath)
                except OSError:
                    pass
            os.rename(tmp_dir, save_dir)

            if verbose:
                self._progress_bar("  Replay save", 1.0)
        except Exception:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        elapsed = time.time() - t0
        total_bytes = sum(os.path.getsize(os.path.join(save_dir, f)) for f in os.listdir(save_dir))
        mb = total_bytes / (1024 * 1024)
        if verbose:
            print(f"  Replay buffer saved: {mb:.0f} MB in {elapsed:.1f}s")

    def load(self, filepath: str, verbose: bool = True) -> bool:
        """Load a replay buffer from disk. Supports new directory format and legacy .npz."""
        # Determine which format exists
        dir_path = filepath.replace(".npz", "") if filepath.endswith(".npz") else filepath
        npz_path = filepath if filepath.endswith(".npz") else filepath + ".npz"

        use_dir = os.path.isdir(dir_path)
        use_npz = (not use_dir) and os.path.isfile(npz_path)

        if not use_dir and not use_npz:
            return False

        if verbose:
            fmt = "directory" if use_dir else "npz (legacy)"
            print(f"  Loading replay buffer ({fmt}) from {dir_path if use_dir else npz_path}...")
        t0 = time.time()
        if verbose:
            self._progress_bar("  Replay load", 0.05)

        try:
            if use_dir:
                arch = self._load_from_dir(dir_path, verbose)
            else:
                arch = self._load_from_npz(npz_path, verbose)
        except Exception as e:
            print(f"  Failed to read replay buffer: {e}")
            return False

        if verbose:
            self._progress_bar("  Replay load", 0.20)

        n = len(arch["states"])
        if n == 0:
            print("  Replay buffer file is empty.")
            return False

        # Validate state size compatibility
        saved_state_size = arch["states"].shape[1]
        if saved_state_size != self.state_size:
            print(f"  State size mismatch: saved={saved_state_size}, expected={self.state_size}")
            return False

        # If saved data exceeds our capacity, take only the most recent entries
        if n > self.capacity:
            print(f"  Saved buffer ({n:,}) exceeds capacity ({self.capacity:,}), truncating to most recent.")
            offset = n - self.capacity
            n = self.capacity
        else:
            offset = 0
        if verbose:
            self._progress_bar("  Replay load", 0.35)

        with self.lock:
            if verbose:
                self._progress_bar("  Replay load", 0.45)

            self.states[:n]      = arch["states"][offset:offset + n]
            self.next_states[:n] = arch["next_states"][offset:offset + n]
            self.actions[:n]     = arch["actions"][offset:offset + n]
            self.rewards[:n]     = arch["rewards"][offset:offset + n]
            self.dones[:n]       = arch["dones"][offset:offset + n]
            self.horizons[:n]    = arch["horizons"][offset:offset + n]
            self.is_expert[:n]   = arch["is_expert"][offset:offset + n]
            if verbose:
                self._progress_bar("  Replay load", 0.62)

            # Restore SumTree state
            priorities = arch["priorities"][offset:offset + n]
            self.tree.size = n
            self.tree.data_ptr = int(arch["data_ptr"]) if offset == 0 else n % self.capacity
            self.tree.max_priority = float(arch["max_priority"])

            # Write all leaf priorities and rebuild the tree
            self.tree.tree[self.tree.capacity:self.tree.capacity + n] = priorities.astype(np.float64)
            # Zero out unused leaves
            if n < self.capacity:
                self.tree.tree[self.tree.capacity + n:] = 0.0

            # Rebuild internal nodes bottom-up
            total_nodes = max(1, self.tree.capacity - 1)
            update_every = max(1, total_nodes // 64)
            for i in range(self.tree.capacity - 1, 0, -1):
                self.tree.tree[i] = self.tree.tree[2 * i] + self.tree.tree[2 * i + 1]
                if verbose and ((self.tree.capacity - i) % update_every == 0):
                    rebuilt = self.tree.capacity - i
                    frac = 0.62 + (0.33 * (rebuilt / total_nodes))
                    self._progress_bar("  Replay load", frac)

            self.size = n
            self._n_expert = int(self.is_expert[:n].sum())

        elapsed = time.time() - t0
        if use_dir:
            total_bytes = sum(os.path.getsize(os.path.join(dir_path, f)) for f in os.listdir(dir_path))
            mb = total_bytes / (1024 * 1024)
        else:
            mb = os.path.getsize(npz_path) / (1024 * 1024)
        if verbose:
            self._progress_bar("  Replay load", 1.0)
            print(f"  Replay buffer loaded: {mb:.0f} MB in {elapsed:.1f}s")
        return True

    @staticmethod
    def _load_from_dir(dir_path: str, verbose: bool) -> dict:
        """Load arrays from the fast directory format."""
        meta = np.load(os.path.join(dir_path, "_meta.npy"))
        result = {
            "data_ptr": int(meta[1]),
            "max_priority": float(meta[2]),
        }
        for name in PrioritizedReplayBuffer._ARRAY_NAMES:
            result[name] = np.load(os.path.join(dir_path, f"{name}.npy"))
        return result

    @staticmethod
    def _load_from_npz(npz_path: str, verbose: bool) -> dict:
        """Load from legacy .npz format."""
        arch = np.load(npz_path, allow_pickle=False)
        return {
            "states": arch["states"],
            "next_states": arch["next_states"],
            "actions": arch["actions"],
            "rewards": arch["rewards"],
            "dones": arch["dones"],
            "horizons": arch["horizons"],
            "is_expert": arch["is_expert"],
            "priorities": arch["priorities"],
            "data_ptr": int(arch["data_ptr"]),
            "max_priority": float(arch["max_priority"]),
        }

    def flush(self):
        """Clear the entire replay buffer."""
        with self.lock:
            self.tree = SumTree(self.capacity)
            self.size = 0
            self._n_expert = 0
            # Zero the storage arrays so stale data can't leak
            self.states.fill(0)
            self.next_states.fill(0)
            self.actions.fill(0)
            self.rewards.fill(0)
            self.dones.fill(0)
            self.horizons.fill(1)
            self.is_expert.fill(0)
        print("  Replay buffer flushed.")
