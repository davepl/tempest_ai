#!/usr/bin/env python3
"""
Minimal smoke test for PrioritizedReplayMemory correctness.
Runs a few add/sample/update cycles and validates invariants:
- priorities are positive and finite
- sampling probabilities are valid (implicitly via sample assertions)
- importance weights are positive and finite
- update_priorities accepts TD errors and updates without errors
"""

import os
import sys
import time
import numpy as np
import torch


def main():
	# Ensure we can import from Scripts/ where aimodel.py and config.py live
	repo_root = os.getcwd()
	scripts_dir = os.path.join(repo_root, 'Scripts')
	if scripts_dir not in sys.path:
		sys.path.insert(0, scripts_dir)

	import aimodel  # noqa: E402

	print("PER smoke test starting…")
	state_size = int(getattr(aimodel.RL_CONFIG, 'state_size', 175))
	capacity = 50000  # keep this small for quick test
	batch_size = 1024
	alpha = float(getattr(aimodel.RL_CONFIG, 'per_alpha', 0.6))
	eps = float(getattr(aimodel.RL_CONFIG, 'per_eps', 1e-6))

	mem = aimodel.PrioritizedReplayMemory(capacity, alpha=alpha, eps=eps)

	# Push enough items to enable sampling
	n_push = max(batch_size * 3, 8192)
	rng = np.random.default_rng(123)
	for i in range(n_push):
		s = rng.standard_normal(state_size, dtype=np.float32)
		a = rng.integers(0, 18)
		r = float(rng.normal())
		ns = rng.standard_normal(state_size, dtype=np.float32)
		d = bool(rng.integers(0, 2))
		mem.push(s, int(a), r, ns, d)

	size = len(mem)
	print(f"Buffer filled: size={size}")

	# Validate initial priority stats
	prios = mem.priorities[:size, 0]
	assert np.all(np.isfinite(prios)), "Found NaN/Inf in priorities after push"
	assert np.all(prios > 0), "Found non-positive priorities after push"
	print(f"Priorities OK: min={prios.min():.6g}, max={prios.max():.6g}, mean={prios.mean():.6g}")

	# Sample a batch and verify shapes and values
	beta = float(getattr(aimodel.RL_CONFIG, 'per_beta_start', 0.4))
	states, actions, rewards, next_states, dones, is_weights, indices = mem.sample(batch_size, beta=beta)

	# Basic shape checks
	assert states.shape == (batch_size, state_size)
	assert actions.shape == (batch_size, 1)
	assert rewards.shape == (batch_size, 1)
	assert next_states.shape == (batch_size, state_size)
	assert dones.shape == (batch_size, 1)
	assert is_weights.shape == (batch_size, 1)
	assert len(indices) == batch_size
	print("Sample shapes OK")

	# Weight validity
	w = is_weights.detach().cpu().numpy().reshape(-1)
	assert np.all(np.isfinite(w)), "IS weights contain NaN/Inf"
	assert np.all(w > 0), "IS weights must be strictly positive"
	print(f"IS weights OK: min={w.min():.6g}, max={w.max():.6g}")

	# Simulate TD errors and update priorities
	td = np.abs(rng.normal(loc=0.5, scale=0.5, size=batch_size)).astype(np.float32)
	td_t = torch.from_numpy(td)
	mem.update_priorities(indices, td_t)
	print("Priority update OK")

	# Spot-check updated entries
	updated = mem.priorities[indices, 0]
	assert np.all(updated > 0), "Updated priorities must be positive"
	assert np.all(np.isfinite(updated)), "Updated priorities contain NaN/Inf"
	print("Updated priorities OK")

	print("PER smoke test PASSED")


if __name__ == '__main__':
	main()
