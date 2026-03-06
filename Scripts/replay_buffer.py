#!/usr/bin/env python3
# ==================================================================================================================
# ||  TEMPEST AI v2 • PRIORITIZED EXPERIENCE REPLAY                                                              ||
# ||  Sum-tree backed proportional PER with per-slot storage.                                                     ||
# ==================================================================================================================
"""Prioritized replay buffer using a sum-tree for O(log N) sampling."""

import os, sys, time, shutil
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

    # Builds the initial state for PrioritizedReplayBuffer and wires the dependencies it needs.
    # Keeping setup in one place avoids partially initialized objects in hot paths.
    def __init__(self, capacity: int, state_size: int, alpha: float = 0.6,
                 memmap_dir: str = None):
        self.capacity = int(capacity)
        self.state_size = int(state_size)
        self.alpha = float(alpha)
        self.lock = threading.Lock()
        self._memmap_dir = memmap_dir or None

        # Storage arrays — either RAM-backed (np.zeros) or disk-backed (np.memmap)
        if self._memmap_dir:
            os.makedirs(self._memmap_dir, exist_ok=True)
            self.states      = self._open_memmap("states",      (self.capacity, self.state_size), np.float32)
            self.next_states = self._open_memmap("next_states", (self.capacity, self.state_size), np.float32)
            self.actions     = self._open_memmap("actions",     (self.capacity,), np.int64)
            self.rewards     = self._open_memmap("rewards",     (self.capacity,), np.float32)
            self.dones       = self._open_memmap("dones",       (self.capacity,), np.float32)
            self.horizons    = self._open_memmap("horizons",    (self.capacity,), np.int32, fill=1)
            self.is_expert   = self._open_memmap("is_expert",   (self.capacity,), np.uint8)
        else:
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

        # Auto-restore from memmap metadata if available
        if self._memmap_dir:
            self._try_restore_memmap_meta()

    # ── Memmap helpers ───────────────────────────────────────────────────

    def _open_memmap(self, name: str, shape: tuple, dtype, fill=0):
        """Open an existing memmap file (if shape matches) or create a new one."""
        path = os.path.join(self._memmap_dir, f"{name}.dat")
        expected_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if os.path.isfile(path) and os.path.getsize(path) == expected_bytes:
            return np.memmap(path, dtype=dtype, mode='r+', shape=shape)
        # Wrong size or missing — (re)create
        if os.path.exists(path):
            os.remove(path)
        mm = np.memmap(path, dtype=dtype, mode='w+', shape=shape)
        if fill != 0:
            mm.fill(fill)
            mm.flush()
        return mm

    def _try_restore_memmap_meta(self):
        """Restore buffer state (size, SumTree) from memmap metadata.

        The array data is already memory-mapped; this only reads the
        small metadata + priority files written by _save_memmap().
        """
        meta_path = os.path.join(self._memmap_dir, "_meta.npy")
        pri_path  = os.path.join(self._memmap_dir, "priorities.npy")
        if not os.path.isfile(meta_path) or not os.path.isfile(pri_path):
            return
        try:
            meta = np.load(meta_path)
            data_ptr = int(meta[0])
            n = int(meta[1])
            max_priority = float(meta[2])
            priorities = np.load(pri_path)
            if n == 0 or len(priorities) < n:
                return
            if n > self.capacity:
                # Capacity shrank since last save — keep most recent
                offset = n - self.capacity
                priorities = priorities[offset:]
                n = self.capacity
                data_ptr = n % self.capacity

            t0 = time.time()
            self.tree.size = n
            self.tree.data_ptr = data_ptr
            self.tree.max_priority = max_priority
            self.tree.tree[self.tree.capacity:self.tree.capacity + n] = priorities[:n].astype(np.float64)
            if n < self.capacity:
                self.tree.tree[self.tree.capacity + n:] = 0.0
            # Rebuild internal nodes
            for i in range(self.tree.capacity - 1, 0, -1):
                self.tree.tree[i] = self.tree.tree[2 * i] + self.tree.tree[2 * i + 1]
            self.size = n
            self._n_expert = int(self.is_expert[:n].sum())
            elapsed = time.time() - t0
            print(f"  Replay buffer restored from memmap: {n:,} transitions ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  Memmap metadata restore failed ({e}), starting with empty buffer.")

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

    # Ingests a new record into PrioritizedReplayBuffer while updating all bookkeeping fields.
    # The insert path is centralized so capacity rollover and counters stay correct.
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

    def boost_priorities(self, indices, factor: float):
        """Multiply existing priorities of the given indices by *factor*.

        Used for pre-death lookback: frames leading up to a death get their
        PER priority boosted so they are sampled more often.
        """
        if factor <= 1.0 or len(indices) == 0:
            return
        with self.lock:
            for idx in indices:
                current = self.tree.priority(int(idx))
                self.tree.update(int(idx), current * factor)

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

    def save(self, filepath: str, verbose: bool = True):
        """Save the replay buffer.

        Memmap mode:  flush dirty pages + write small metadata/priority files.
        Legacy mode:  copy & write one array at a time to avoid doubling RSS.
        """
        if self._memmap_dir:
            return self._save_memmap(verbose)
        return self._save_npy(filepath, verbose)

    # ── Memmap save (fast — just msync + tiny metadata) ─────────────────
    def _save_memmap(self, verbose: bool = True):
        with self.lock:
            if self.size == 0:
                if verbose:
                    print("  Replay buffer is empty — nothing to save.")
                return
            n = self.size
            meta = np.array([self.tree.data_ptr, n, self.tree.max_priority])
            priorities = self.tree.tree[self.tree.capacity:self.tree.capacity + n].copy()

        t0 = time.time()
        if verbose:
            self._progress_bar("  Replay save", 0.05)

        # Flush all memmap arrays to disk
        for arr in (self.states, self.next_states, self.actions,
                    self.rewards, self.dones, self.horizons, self.is_expert):
            if hasattr(arr, 'flush'):
                arr.flush()
        if verbose:
            self._progress_bar("  Replay save", 0.80)

        # Write metadata + SumTree priorities (small files, atomic via tmp+rename)
        for name, data in [("_meta", meta), ("priorities", priorities)]:
            tmp = os.path.join(self._memmap_dir, f"{name}.tmp.npy")
            dst = os.path.join(self._memmap_dir, f"{name}.npy")
            np.save(tmp, data)
            os.replace(tmp, dst)
        if verbose:
            self._progress_bar("  Replay save", 1.0)

        elapsed = time.time() - t0
        if verbose:
            print(f"  Replay buffer saved (memmap flush): {n:,} transitions in {elapsed:.1f}s")

    # ── Legacy npy save (copy one array at a time) ──────────────────────
    def _save_npy(self, filepath: str, verbose: bool = True):
        with self.lock:
            if self.size == 0:
                if verbose:
                    print("  Replay buffer is empty — nothing to save.")
                return
            n = self.size
            meta = np.array([self.tree.data_ptr, n, self.tree.max_priority])

        if verbose:
            print(f"  Saving replay buffer ({n:,} transitions)...")
        t0 = time.time()

        tmp_dir = filepath + ".tmp"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)

        array_specs = [
            ("states",      lambda: self.states[:n]),
            ("next_states", lambda: self.next_states[:n]),
            ("actions",     lambda: self.actions[:n]),
            ("rewards",     lambda: self.rewards[:n]),
            ("dones",       lambda: self.dones[:n]),
            ("horizons",    lambda: self.horizons[:n]),
            ("is_expert",   lambda: self.is_expert[:n]),
            ("priorities",  lambda: self.tree.tree[self.tree.capacity:self.tree.capacity + n]),
        ]

        total_bytes = 0
        for i, (name, src_fn) in enumerate(array_specs):
            with self.lock:
                arr = src_fn().copy()
            total_bytes += arr.nbytes
            np.save(os.path.join(tmp_dir, f"{name}.npy"), arr)
            del arr
            if verbose:
                frac = (i + 1) / len(array_specs)
                self._progress_bar("  Replay save", frac * 0.95)

        np.save(os.path.join(tmp_dir, "_meta.npy"), meta)

        if os.path.exists(filepath):
            if os.path.isdir(filepath):
                shutil.rmtree(filepath, ignore_errors=True)
            else:
                os.remove(filepath)
        os.rename(tmp_dir, filepath)
        if verbose:
            self._progress_bar("  Replay save", 1.0)

        elapsed = time.time() - t0
        mb = total_bytes / (1024 * 1024)
        if verbose:
            print(f"  Replay buffer saved: {mb:.0f} MB in {elapsed:.1f}s")

    def _load_directory(self, dirpath: str, verbose: bool = True) -> bool:
        """Load replay buffer from a directory of .npy files."""
        meta_path = os.path.join(dirpath, "_meta.npy")
        states_path = os.path.join(dirpath, "states.npy")
        if not os.path.isfile(meta_path) or not os.path.isfile(states_path):
            return False

        if verbose:
            print(f"  Loading replay buffer (directory format) from {dirpath}...")
        t0 = time.time()
        if verbose:
            self._progress_bar("  Replay load", 0.05)

        try:
            meta = np.load(meta_path)
            data_ptr = int(meta[0])
            saved_n = int(meta[1])
            max_priority = float(meta[2])
        except Exception as e:
            print(f"  Failed to read replay meta: {e}")
            return False

        # Load arrays
        names = ["states", "next_states", "actions", "rewards", "dones", "horizons", "is_expert", "priorities"]
        arch = {}
        for i, name in enumerate(names):
            fpath = os.path.join(dirpath, f"{name}.npy")
            if not os.path.isfile(fpath):
                print(f"  Missing array file: {name}.npy")
                return False
            arch[name] = np.load(fpath)
            if verbose:
                frac = 0.05 + 0.30 * ((i + 1) / len(names))
                self._progress_bar("  Replay load", frac)

        return self._restore_from_arrays(arch, data_ptr, max_priority, t0, dirpath, verbose)

    def _load_npz(self, filepath: str, verbose: bool = True) -> bool:
        """Load replay buffer from a legacy .npz file."""
        if not os.path.isfile(filepath):
            return False

        if verbose:
            print(f"  Loading replay buffer (legacy npz) from {filepath}...")
        t0 = time.time()
        if verbose:
            self._progress_bar("  Replay load", 0.05)

        try:
            arch = np.load(filepath, allow_pickle=False)
        except Exception as e:
            print(f"  Failed to read replay buffer: {e}")
            return False
        if verbose:
            self._progress_bar("  Replay load", 0.35)

        data_ptr = int(arch["data_ptr"]) if "data_ptr" in arch else 0
        max_priority = float(arch["max_priority"]) if "max_priority" in arch else 1.0
        return self._restore_from_arrays(dict(arch), data_ptr, max_priority, t0, filepath, verbose)

    def _restore_from_arrays(self, arch: dict, data_ptr: int, max_priority: float,
                              t0: float, source_path: str, verbose: bool) -> bool:
        """Common restore logic for both directory and npz formats."""
        n = len(arch["states"])
        if n == 0:
            print("  Replay buffer file is empty.")
            return False

        saved_state_size = arch["states"].shape[1]
        if saved_state_size != self.state_size:
            print(f"  State size mismatch: saved={saved_state_size}, expected={self.state_size}")
            return False

        if n > self.capacity:
            print(f"  Saved buffer ({n:,}) exceeds capacity ({self.capacity:,}), truncating to most recent.")
            offset = n - self.capacity
            n = self.capacity
        else:
            offset = 0
        if verbose:
            self._progress_bar("  Replay load", 0.40)

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

            priorities = arch["priorities"][offset:offset + n]
            self.tree.size = n
            self.tree.data_ptr = data_ptr if offset == 0 else n % self.capacity
            self.tree.max_priority = max_priority

            self.tree.tree[self.tree.capacity:self.tree.capacity + n] = priorities.astype(np.float64)
            if n < self.capacity:
                self.tree.tree[self.tree.capacity + n:] = 0.0

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
        if verbose:
            self._progress_bar("  Replay load", 1.0)
            print(f"  Replay buffer loaded: {n:,} transitions in {elapsed:.1f}s")
        return True

    def load(self, filepath: str, verbose: bool = True) -> bool:
        """Load replay buffer.

        Memmap mode:  data was auto-restored in __init__; only fall through
                      to the legacy loaders for one-time migration from old
                      .npy / .npz saves.
        Legacy mode:  tries directory format, then .npz.
        """
        # If memmap already has data from __init__, we're done.
        if self._memmap_dir and self.size > 0:
            if verbose:
                print(f"  Replay buffer live from memmap ({self.size:,} transitions)")
            return True

        # Try directory format (new fast path)
        if os.path.isdir(filepath):
            ok = self._load_directory(filepath, verbose)
            if ok and self._memmap_dir:
                self._flush_memmaps_after_migration(filepath, verbose)
            return ok
        # Try .npz at the given path
        if os.path.isfile(filepath):
            ok = self._load_npz(filepath, verbose)
            if ok and self._memmap_dir:
                self._flush_memmaps_after_migration(filepath, verbose)
            return ok
        # Try deriving the directory path from a .npz path or vice versa
        if filepath.endswith(".npz"):
            dir_path = filepath[:-4]
            if os.path.isdir(dir_path):
                ok = self._load_directory(dir_path, verbose)
                if ok and self._memmap_dir:
                    self._flush_memmaps_after_migration(dir_path, verbose)
                return ok
        else:
            npz_path = filepath + ".npz"
            if os.path.isfile(npz_path):
                ok = self._load_npz(npz_path, verbose)
                if ok and self._memmap_dir:
                    self._flush_memmaps_after_migration(npz_path, verbose)
                return ok
        return False

    def _flush_memmaps_after_migration(self, old_path: str, verbose: bool):
        """After migrating from old format into memmaps, flush and write metadata."""
        try:
            self._save_memmap(verbose=False)
            if verbose:
                print(f"  Migrated replay data into memmap at {self._memmap_dir}")
        except Exception as e:
            print(f"  Memmap migration flush failed: {e}")

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
