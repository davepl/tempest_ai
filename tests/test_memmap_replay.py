#!/usr/bin/env python3
"""Tests for memory-mapped replay buffer."""

import os, sys, shutil, tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Scripts"))
from replay_buffer import PrioritizedReplayBuffer


@pytest.fixture
def tmpdir():
    d = tempfile.mkdtemp(prefix="replay_memmap_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _fill_buffer(buf, n, state_size):
    """Add n transitions with identifiable data."""
    for i in range(n):
        s = np.full(state_size, float(i), dtype=np.float32)
        ns = np.full(state_size, float(i + 0.5), dtype=np.float32)
        buf.add(s, i % 5, float(i) * 0.1, ns, done=(i % 20 == 19))


# ── Basic memmap creation ───────────────────────────────────────────────

def test_memmap_creates_dat_files(tmpdir):
    """Memmap buffer should create .dat files in the memmap directory."""
    buf = PrioritizedReplayBuffer(capacity=100, state_size=10, memmap_dir=tmpdir)
    expected = {"states.dat", "next_states.dat", "actions.dat", "rewards.dat",
                "dones.dat", "horizons.dat", "is_expert.dat"}
    actual = {f for f in os.listdir(tmpdir) if f.endswith(".dat")}
    assert expected == actual


def test_memmap_add_and_sample(tmpdir):
    """Add transitions to memmap buffer and verify sampling works."""
    buf = PrioritizedReplayBuffer(capacity=100, state_size=8, memmap_dir=tmpdir)
    _fill_buffer(buf, 50, state_size=8)
    assert len(buf) == 50

    batch = buf.sample(16, beta=0.4)
    assert batch is not None
    states, actions, rewards, next_states, dones, horizons, is_expert, indices, weights = batch
    assert states.shape == (16, 8)
    assert len(indices) == 16


# ── Save / restore round-trip ───────────────────────────────────────────

def test_memmap_save_and_restore(tmpdir):
    """Buffer state should survive save + fresh reopen."""
    buf = PrioritizedReplayBuffer(capacity=200, state_size=4, memmap_dir=tmpdir)
    _fill_buffer(buf, 80, state_size=4)
    assert len(buf) == 80

    # Save (flush memmaps + write metadata)
    buf.save("ignored_path", verbose=False)

    # Simulate restart: create a new buffer pointing at the same dir
    buf2 = PrioritizedReplayBuffer(capacity=200, state_size=4, memmap_dir=tmpdir)
    assert len(buf2) == 80

    # Verify data integrity: first transition
    np.testing.assert_array_almost_equal(buf2.states[0], np.full(4, 0.0))
    np.testing.assert_array_almost_equal(buf2.next_states[0], np.full(4, 0.5))

    # Verify sampling works after restore
    batch = buf2.sample(16, beta=0.4)
    assert batch is not None


def test_memmap_wraps_around(tmpdir):
    """Ring buffer should wrap correctly with memmap backing."""
    cap = 50
    buf = PrioritizedReplayBuffer(capacity=cap, state_size=4, memmap_dir=tmpdir)
    _fill_buffer(buf, 120, state_size=4)  # far more than capacity
    assert len(buf) == cap

    # Save and reopen
    buf.save("ignored", verbose=False)
    buf2 = PrioritizedReplayBuffer(capacity=cap, state_size=4, memmap_dir=tmpdir)
    assert len(buf2) == cap


# ── Migration from old format ──────────────────────────────────────────

def test_migrate_from_npy_to_memmap(tmpdir):
    """Old .npy format should be loadable into a memmap buffer."""
    # Create a buffer with old format (no memmap), save it
    old_dir = os.path.join(tmpdir, "old_replay")
    buf_old = PrioritizedReplayBuffer(capacity=100, state_size=4)
    _fill_buffer(buf_old, 40, state_size=4)
    buf_old.save(old_dir, verbose=False)

    # Create a memmap buffer and load from old format
    mm_dir = os.path.join(tmpdir, "memmap")
    buf_mm = PrioritizedReplayBuffer(capacity=100, state_size=4, memmap_dir=mm_dir)
    assert len(buf_mm) == 0  # no memmap metadata yet

    ok = buf_mm.load(old_dir, verbose=False)
    assert ok
    assert len(buf_mm) == 40

    # Verify data migrated correctly
    np.testing.assert_array_almost_equal(buf_mm.states[0], np.full(4, 0.0))

    # Reopen — should now load from memmap directly
    buf_mm2 = PrioritizedReplayBuffer(capacity=100, state_size=4, memmap_dir=mm_dir)
    assert len(buf_mm2) == 40


# ── Capacity mismatch ──────────────────────────────────────────────────

def test_memmap_capacity_change_recreates(tmpdir):
    """Changing capacity should recreate memmap files (not crash)."""
    buf1 = PrioritizedReplayBuffer(capacity=100, state_size=4, memmap_dir=tmpdir)
    _fill_buffer(buf1, 50, state_size=4)
    buf1.save("ignored", verbose=False)

    # Reopen with different capacity — files should be recreated
    buf2 = PrioritizedReplayBuffer(capacity=200, state_size=4, memmap_dir=tmpdir)
    # Metadata was for cap=100, files are now cap=200 → meta restore may skip
    # but the buffer should be usable
    assert buf2.capacity == 200


# ── Flush ───────────────────────────────────────────────────────────────

def test_memmap_flush_clears(tmpdir):
    """Flushing a memmap buffer should zero all arrays."""
    buf = PrioritizedReplayBuffer(capacity=100, state_size=4, memmap_dir=tmpdir)
    _fill_buffer(buf, 50, state_size=4)
    assert len(buf) == 50
    buf.flush()
    assert len(buf) == 0
    assert buf.states[0].sum() == 0.0


# ── Non-memmap still works ──────────────────────────────────────────────

def test_non_memmap_still_works(tmpdir):
    """Passing memmap_dir=None should use plain numpy arrays (no .dat files)."""
    buf = PrioritizedReplayBuffer(capacity=100, state_size=4, memmap_dir=None)
    _fill_buffer(buf, 30, state_size=4)
    assert len(buf) == 30
    assert not isinstance(buf.states, np.memmap)

    save_path = os.path.join(tmpdir, "legacy_replay")
    buf.save(save_path, verbose=False)
    assert os.path.isdir(save_path)

    buf2 = PrioritizedReplayBuffer(capacity=100, state_size=4, memmap_dir=None)
    ok = buf2.load(save_path, verbose=False)
    assert ok
    assert len(buf2) == 30
