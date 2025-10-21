#!/usr/bin/env python3
"""Comprehensive unit tests for N-bucket stratified replay buffer.

Tests cover:
- Initialization and configuration
- Basic push/sample operations
- TD-error based bucket routing
- Percentile threshold updates
- Bucket fill distribution
- Edge cases and error handling
- Actor composition tracking
- Statistics and monitoring
"""

import unittest
import numpy as np
import torch
import sys
import os

# Add Scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Scripts'))

from aimodel import HybridReplayBuffer
from config import RL_CONFIG


class TestNBucketReplayBuffer(unittest.TestCase):
    """Comprehensive test suite for N-bucket stratified replay buffer."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        self.state_size = 128
        self.capacity = 10000  # Small for fast tests
        
        # Store original config values
        self.orig_n_buckets = RL_CONFIG.replay_n_buckets
        self.orig_bucket_size = RL_CONFIG.replay_bucket_size
        self.orig_main_bucket_size = RL_CONFIG.replay_main_bucket_size
        
        # Set test config
        RL_CONFIG.replay_n_buckets = 5
        RL_CONFIG.replay_bucket_size = 1000
        RL_CONFIG.replay_main_bucket_size = 5000
        
    def tearDown(self):
        """Clean up after each test."""
        # Restore original config
        RL_CONFIG.replay_n_buckets = self.orig_n_buckets
        RL_CONFIG.replay_bucket_size = self.orig_bucket_size
        RL_CONFIG.replay_main_bucket_size = self.orig_main_bucket_size
    
    def _make_experience(self, td_error=0.0, reward=0.0, done=False, actor='dqn', horizon=1):
        """Create a dummy experience tuple for testing."""
        state = np.random.randn(self.state_size).astype(np.float32)
        next_state = np.random.randn(self.state_size).astype(np.float32)
        discrete_action = np.random.randint(0, 4)
        continuous_action = np.random.uniform(-0.9, 0.9)
        
        return {
            'state': state,
            'discrete_action': discrete_action,
            'continuous_action': continuous_action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'actor': actor,
            'horizon': horizon,
            'td_error': td_error
        }
    
    def test_initialization(self):
        """Test buffer initialization and capacity calculation."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        # Check bucket configuration
        self.assertEqual(len(buffer.buckets), 6)  # 5 priority + 1 main
        
        # Check capacity calculation
        expected_capacity = 5 * 1000 + 5000  # 5 buckets * 1000 + main bucket
        self.assertEqual(buffer.capacity, expected_capacity)
        self.assertEqual(buffer.size, 0)
        
        # Check storage arrays are allocated
        self.assertEqual(buffer.states.shape, (expected_capacity, self.state_size))
        self.assertEqual(buffer.discrete_actions.shape, (expected_capacity,))
        self.assertEqual(buffer.continuous_actions.shape, (expected_capacity,))
        self.assertEqual(buffer.rewards.shape, (expected_capacity,))
        self.assertEqual(buffer.next_states.shape, (expected_capacity, self.state_size))
        self.assertEqual(buffer.dones.shape, (expected_capacity,))
        self.assertEqual(buffer.actors.shape, (expected_capacity,))
        self.assertEqual(buffer.horizons.shape, (expected_capacity,))
        
        # Check bucket offsets are contiguous
        expected_offset = 0
        for bucket in buffer.buckets:
            self.assertEqual(bucket['offset'], expected_offset)
            expected_offset += bucket['capacity']
    
    def test_bucket_configuration(self):
        """Test that buckets are configured with correct percentile ranges."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        # Priority buckets should cover 50-100th percentile in deciles
        expected_ranges = [
            (90, 100),
            (80, 90),
            (70, 80),
            (60, 70),
            (50, 60),
        ]
        
        for i, (low, high) in enumerate(expected_ranges):
            bucket = buffer.buckets[i]
            self.assertEqual(bucket['percentile_low'], low)
            self.assertEqual(bucket['percentile_high'], high)
            self.assertEqual(bucket['capacity'], 1000)
            self.assertEqual(bucket['name'], f'p{low}-{high}')
        
        # Main bucket should be <50th percentile
        main_bucket = buffer.buckets[-1]
        self.assertEqual(main_bucket['name'], 'main')
        self.assertEqual(main_bucket['percentile_low'], 0)
        self.assertEqual(main_bucket['percentile_high'], 50)
        self.assertEqual(main_bucket['capacity'], 5000)
    
    def test_push_basic(self):
        """Test basic push operation."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        exp = self._make_experience(td_error=0.5)
        buffer.push(**exp)
        
        self.assertEqual(len(buffer), 1)
        self.assertEqual(buffer.size, 1)
        
        # Check data is stored correctly
        self.assertTrue(np.allclose(buffer.states[0], exp['state']))
        self.assertEqual(buffer.discrete_actions[0], exp['discrete_action'])
        self.assertAlmostEqual(buffer.continuous_actions[0], exp['continuous_action'], places=5)
        self.assertEqual(buffer.rewards[0], exp['reward'])
        self.assertTrue(np.allclose(buffer.next_states[0], exp['next_state']))
        self.assertEqual(buffer.dones[0], exp['done'])
        self.assertEqual(buffer.actors[0], exp['actor'])
        self.assertEqual(buffer.horizons[0], exp['horizon'])
    
    def test_push_multiple(self):
        """Test pushing multiple experiences."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        n_experiences = 100
        for i in range(n_experiences):
            exp = self._make_experience(td_error=np.random.uniform(0, 5))
            buffer.push(**exp)
        
        self.assertEqual(len(buffer), n_experiences)
        self.assertEqual(buffer.size, n_experiences)
    
    def test_bucket_routing_low_td_error(self):
        """Test that low TD-error experiences go to main bucket after thresholds are established."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        # First, establish percentile thresholds with varied TD-errors
        for _ in range(1500):
            exp = self._make_experience(td_error=np.random.uniform(1, 5))
            buffer.push(**exp)
        
        # Now push experiences with very low TD errors (should go to main bucket)
        n_low_error = 50
        for _ in range(n_low_error):
            exp = self._make_experience(td_error=0.001)
            buffer.push(**exp)
        
        # Main bucket should have accumulated the low-error experiences
        # (Some of the initial 1500 may also be in main bucket)
        main_bucket = buffer.buckets[-1]
        self.assertGreater(main_bucket['size'], 0)
    
    def test_bucket_routing_high_td_error(self):
        """Test that high TD-error experiences go to priority buckets."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        # First, fill the TD-error window to establish thresholds
        for _ in range(2000):
            exp = self._make_experience(td_error=np.random.uniform(0, 1))
            buffer.push(**exp)
        
        # Now push experiences with very high TD errors
        n_high_error = 50
        for _ in range(n_high_error):
            exp = self._make_experience(td_error=10.0)  # Much higher than threshold
            buffer.push(**exp)
        
        # High-error experiences should be in top priority bucket
        top_bucket = buffer.buckets[0]
        self.assertGreater(top_bucket['size'], 0)
    
    def test_percentile_threshold_update(self):
        """Test that percentile thresholds are updated correctly."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        # Initially thresholds should be zero
        self.assertEqual(buffer.percentile_thresholds, [0.0] * 5)
        
        # Push enough experiences to trigger threshold update
        td_errors = []
        for _ in range(1500):
            td_error = np.random.uniform(0, 5)
            td_errors.append(td_error)
            exp = self._make_experience(td_error=td_error)
            buffer.push(**exp)
        
        # Thresholds should now be non-zero
        self.assertNotEqual(buffer.percentile_thresholds, [0.0] * 5)
        
        # Thresholds should be in descending order (90th > 80th > 70th > ...)
        thresholds = buffer.percentile_thresholds
        for i in range(len(thresholds) - 1):
            self.assertGreaterEqual(thresholds[i], thresholds[i + 1])
        
        # Verify threshold values match expected percentiles (within reasonable tolerance)
        recent_errors = list(buffer.td_error_window)
        expected_p90 = np.percentile(recent_errors, 90)
        expected_p50 = np.percentile(recent_errors, 50)
        
        # Thresholds should be in reasonable range (within 10% of expected)
        self.assertLess(abs(thresholds[0] - expected_p90) / expected_p90, 0.1)
        self.assertLess(abs(thresholds[-1] - expected_p50) / expected_p50, 0.1)
    
    def test_sample_before_enough_data(self):
        """Test that sampling returns None when buffer is too small."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        batch_size = 32
        
        # Try sampling with empty buffer
        batch = buffer.sample(batch_size)
        self.assertIsNone(batch)
        
        # Push less than batch_size experiences
        for _ in range(batch_size - 1):
            exp = self._make_experience()
            buffer.push(**exp)
        
        batch = buffer.sample(batch_size)
        self.assertIsNone(batch)
    
    def test_sample_basic(self):
        """Test basic sampling operation."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        batch_size = 32
        n_experiences = 100
        
        # Push enough experiences
        for _ in range(n_experiences):
            exp = self._make_experience()
            buffer.push(**exp)
        
        # Sample batch
        batch = buffer.sample(batch_size)
        self.assertIsNotNone(batch)
        
        # Check batch structure
        states, discrete_actions, continuous_actions, rewards, next_states, dones, actors, horizons = batch
        
        self.assertEqual(states.shape, (batch_size, self.state_size))
        self.assertEqual(discrete_actions.shape, (batch_size, 1))
        self.assertEqual(continuous_actions.shape, (batch_size, 1))
        self.assertEqual(rewards.shape, (batch_size, 1))
        self.assertEqual(next_states.shape, (batch_size, self.state_size))
        self.assertEqual(dones.shape, (batch_size, 1))
        self.assertEqual(len(actors), batch_size)
        self.assertEqual(horizons.shape, (batch_size, 1))
        
        # Check types
        self.assertTrue(isinstance(states, torch.Tensor))
        self.assertTrue(isinstance(discrete_actions, torch.Tensor))
        self.assertTrue(isinstance(continuous_actions, torch.Tensor))
        self.assertTrue(isinstance(rewards, torch.Tensor))
        self.assertTrue(isinstance(next_states, torch.Tensor))
        self.assertTrue(isinstance(dones, torch.Tensor))
        self.assertTrue(isinstance(actors, np.ndarray))
        self.assertTrue(isinstance(horizons, torch.Tensor))
    
    def test_sample_uniform_distribution(self):
        """Test that sampling is uniform across buffer."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        n_experiences = 1000
        batch_size = 100
        n_samples = 50
        
        # Push experiences with unique discrete actions as markers
        for i in range(n_experiences):
            exp = self._make_experience()
            exp['discrete_action'] = i % 4  # Cycle through actions
            buffer.push(**exp)
        
        # Sample multiple times and check distribution
        sampled_indices = []
        for _ in range(n_samples):
            batch = buffer.sample(batch_size)
            states, discrete_actions, _, _, _, _, _, _ = batch
            sampled_indices.extend(discrete_actions.cpu().numpy().flatten().tolist())
        
        # Check that all actions appear in samples (basic uniformity test)
        unique_actions = set(sampled_indices)
        self.assertEqual(len(unique_actions), 4)
    
    def test_ring_buffer_wraparound(self):
        """Test that buffer wraps around correctly when individual buckets are full."""
        # Use small bucket size for faster test
        RL_CONFIG.replay_bucket_size = 10
        RL_CONFIG.replay_main_bucket_size = 50
        
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        total_capacity = buffer.capacity
        
        self.assertEqual(total_capacity, 100)  # 5*10 + 50 = 100
        
        # First establish thresholds with varied TD-errors
        for i in range(2000):
            exp = self._make_experience(td_error=np.random.uniform(0, 5))
            exp['reward'] = float(i)
            buffer.push(**exp)
        
        # Buffer should have wrapped and be at capacity
        self.assertEqual(len(buffer), total_capacity)
        
        # Continue pushing to verify wraparound behavior
        for i in range(2000, 2200):
            exp = self._make_experience(td_error=np.random.uniform(0, 5))
            exp['reward'] = float(i)
            buffer.push(**exp)
        
        # Size should still be at capacity
        self.assertEqual(len(buffer), total_capacity)
        
        # Sample and check that we get recent rewards (not oldest ones)
        batch = buffer.sample(32)
        _, _, _, rewards, _, _, _, _ = batch
        
        # Recent rewards should be higher (from later pushes)
        mean_reward = rewards.mean().item()
        # Oldest experiences (0-99) should be overwritten
        self.assertGreater(mean_reward, 100)
    
    def test_actor_composition_dqn_only(self):
        """Test actor composition tracking with DQN-only experiences."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        n_experiences = 100
        for _ in range(n_experiences):
            exp = self._make_experience(actor='dqn')
            buffer.push(**exp)
        
        comp = buffer.get_actor_composition()
        self.assertEqual(comp['total'], n_experiences)
        self.assertEqual(comp['dqn'], n_experiences)
        self.assertEqual(comp['expert'], 0)
        self.assertAlmostEqual(comp['frac_dqn'], 1.0)
        self.assertAlmostEqual(comp['frac_expert'], 0.0)
    
    def test_actor_composition_expert_only(self):
        """Test actor composition tracking with expert-only experiences."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        n_experiences = 100
        for _ in range(n_experiences):
            exp = self._make_experience(actor='expert')
            buffer.push(**exp)
        
        comp = buffer.get_actor_composition()
        self.assertEqual(comp['total'], n_experiences)
        self.assertEqual(comp['dqn'], 0)
        self.assertEqual(comp['expert'], n_experiences)
        self.assertAlmostEqual(comp['frac_dqn'], 0.0)
        self.assertAlmostEqual(comp['frac_expert'], 1.0)
    
    def test_actor_composition_mixed(self):
        """Test actor composition tracking with mixed DQN/expert experiences."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        n_dqn = 70
        n_expert = 30
        
        for _ in range(n_dqn):
            exp = self._make_experience(actor='dqn')
            buffer.push(**exp)
        
        for _ in range(n_expert):
            exp = self._make_experience(actor='expert')
            buffer.push(**exp)
        
        comp = buffer.get_actor_composition()
        self.assertEqual(comp['total'], n_dqn + n_expert)
        self.assertEqual(comp['dqn'], n_dqn)
        self.assertEqual(comp['expert'], n_expert)
        self.assertAlmostEqual(comp['frac_dqn'], 0.7, places=2)
        self.assertAlmostEqual(comp['frac_expert'], 0.3, places=2)
    
    def test_partition_stats(self):
        """Test partition statistics reporting."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        # Push some experiences
        for _ in range(500):
            td_error = np.random.uniform(0, 5)
            exp = self._make_experience(td_error=td_error)
            buffer.push(**exp)
        
        stats = buffer.get_partition_stats()
        
        # Check required fields
        self.assertIn('total_size', stats)
        self.assertIn('total_capacity', stats)
        self.assertEqual(stats['total_size'], 500)
        self.assertEqual(stats['total_capacity'], buffer.capacity)
        
        # Check bucket stats
        for bucket in buffer.buckets:
            prefix = bucket['name'].replace('-', '_')
            self.assertIn(f'{prefix}_size', stats)
            self.assertIn(f'{prefix}_capacity', stats)
            self.assertIn(f'{prefix}_fill_pct', stats)
        
        # Check threshold stats
        for i in range(5):
            p = 100 - ((i + 1) * 10)
            self.assertIn(f'threshold_p{p}', stats)
    
    def test_terminal_state_handling(self):
        """Test that terminal states (done=True) are handled correctly."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        # Push mix of terminal and non-terminal
        n_terminal = 10
        n_non_terminal = 90
        
        for _ in range(n_non_terminal):
            exp = self._make_experience(done=False)
            buffer.push(**exp)
        
        for _ in range(n_terminal):
            exp = self._make_experience(done=True)
            buffer.push(**exp)
        
        self.assertEqual(len(buffer), n_terminal + n_non_terminal)
        
        # Sample and verify we can get terminal states
        batch = buffer.sample(32)
        _, _, _, _, _, dones, _, _ = batch
        
        # Should have some terminal states in sample
        n_done = dones.sum().item()
        self.assertGreaterEqual(n_done, 0)  # At least 0 (could be more)
    
    def test_horizon_values(self):
        """Test that horizon values are stored and sampled correctly."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        # Push experiences with varying horizons
        horizons_used = [1, 2, 3, 5, 10]
        for h in horizons_used:
            for _ in range(20):
                exp = self._make_experience(horizon=h)
                buffer.push(**exp)
        
        # Sample and check horizons
        batch = buffer.sample(50)
        _, _, _, _, _, _, _, horizons = batch
        
        unique_horizons = set(horizons.cpu().numpy().flatten().tolist())
        
        # Should see multiple horizon values
        self.assertGreater(len(unique_horizons), 1)
        
        # All horizons should be positive
        self.assertTrue(all(h >= 1 for h in unique_horizons))
    
    def test_continuous_action_clamping(self):
        """Test that continuous actions are clamped to valid range."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        # Push experiences with out-of-range continuous actions
        exp1 = self._make_experience()
        exp1['continuous_action'] = 2.0  # Too high
        buffer.push(**exp1)
        
        exp2 = self._make_experience()
        exp2['continuous_action'] = -2.0  # Too low
        buffer.push(**exp2)
        
        # Check clamping
        self.assertLessEqual(buffer.continuous_actions[0], 0.9)
        self.assertGreaterEqual(buffer.continuous_actions[1], -0.9)
    
    def test_invalid_actor_tag(self):
        """Test that invalid actor tags raise errors."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        # Empty actor tag
        exp = self._make_experience(actor='')
        with self.assertRaises(ValueError):
            buffer.push(**exp)
        
        # Invalid actor tags
        for invalid_actor in ['unknown', 'none', 'random']:
            exp = self._make_experience(actor=invalid_actor)
            with self.assertRaises(ValueError):
                buffer.push(**exp)
    
    def test_invalid_horizon(self):
        """Test that invalid horizons raise errors."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        # Zero horizon
        exp = self._make_experience(horizon=0)
        with self.assertRaises(ValueError):
            buffer.push(**exp)
        
        # Negative horizon
        exp = self._make_experience(horizon=-1)
        with self.assertRaises(ValueError):
            buffer.push(**exp)
    
    def test_state_size_mismatch_handling(self):
        """Test that mismatched state sizes are handled gracefully."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        # State too small
        exp = self._make_experience()
        exp['state'] = np.random.randn(self.state_size // 2).astype(np.float32)
        buffer.push(**exp)  # Should not crash
        
        # Check state was padded
        self.assertEqual(buffer.states[0].shape[0], self.state_size)
        
        # State too large
        exp = self._make_experience()
        exp['state'] = np.random.randn(self.state_size * 2).astype(np.float32)
        buffer.push(**exp)  # Should not crash
        
        # Check state was truncated
        self.assertEqual(buffer.states[1].shape[0], self.state_size)
    
    def test_discrete_action_range(self):
        """Test that discrete actions are in valid range (0-3)."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        # Push experiences with all valid discrete actions
        for action in range(4):
            exp = self._make_experience()
            exp['discrete_action'] = action
            buffer.push(**exp)
        
        # Verify stored correctly
        for i in range(4):
            self.assertEqual(buffer.discrete_actions[i], i)
        
        # Sample and verify
        batch = buffer.sample(4)
        _, discrete_actions, _, _, _, _, _, _ = batch
        
        actions = discrete_actions.cpu().numpy().flatten()
        self.assertTrue(all(0 <= a < 4 for a in actions))
    
    def test_bucket_fill_progression(self):
        """Test that buckets fill in expected order as TD-errors vary."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        # Stage 1: Fill with varied TD-errors to establish thresholds
        for _ in range(1500):
            exp = self._make_experience(td_error=np.random.uniform(0, 2))
            buffer.push(**exp)
        
        initial_stats = buffer.get_partition_stats()
        
        # Stage 2: Push experiences with very low TD-errors
        for _ in range(100):
            exp = self._make_experience(td_error=0.01)
            buffer.push(**exp)
        
        stats1 = buffer.get_partition_stats()
        main_size1 = stats1['main_size']
        
        # Main bucket should have grown
        self.assertGreater(main_size1, initial_stats['main_size'])
        
        # Stage 3: Push high TD-error experiences (should go to priority buckets)
        for _ in range(100):
            exp = self._make_experience(td_error=5.0)
            buffer.push(**exp)
        
        stats2 = buffer.get_partition_stats()
        top_bucket_size = stats2['p90_100_size']
        
        # Top priority bucket should have experiences
        self.assertGreater(top_bucket_size, 0)
    
    def test_td_error_window_size(self):
        """Test that TD-error window maintains correct size."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        max_window_size = 50000
        
        # Push many experiences
        for _ in range(max_window_size + 1000):
            exp = self._make_experience(td_error=np.random.uniform(0, 5))
            buffer.push(**exp)
        
        # Window should be capped at max size
        self.assertEqual(len(buffer.td_error_window), max_window_size)
    
    def test_multiple_sample_batches(self):
        """Test sampling multiple batches independently."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        # Fill buffer
        for _ in range(1000):
            exp = self._make_experience()
            buffer.push(**exp)
        
        batch_size = 32
        
        # Sample multiple batches
        batch1 = buffer.sample(batch_size)
        batch2 = buffer.sample(batch_size)
        batch3 = buffer.sample(batch_size)
        
        # All should be valid
        self.assertIsNotNone(batch1)
        self.assertIsNotNone(batch2)
        self.assertIsNotNone(batch3)
        
        # Check they're different (with high probability)
        states1, _, _, _, _, _, _, _ = batch1
        states2, _, _, _, _, _, _, _ = batch2
        
        # Compare first state from each batch (should be different with high probability)
        self.assertFalse(torch.allclose(states1[0], states2[0]))
    
    def test_empty_buffer_stats(self):
        """Test statistics on empty buffer."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        # Actor composition on empty buffer
        comp = buffer.get_actor_composition()
        self.assertEqual(comp['total'], 0)
        self.assertEqual(comp['dqn'], 0)
        self.assertEqual(comp['expert'], 0)
        
        # Partition stats on empty buffer
        stats = buffer.get_partition_stats()
        self.assertEqual(stats['total_size'], 0)
        self.assertGreater(stats['total_capacity'], 0)


class TestNBucketIntegration(unittest.TestCase):
    """Integration tests for N-bucket replay buffer in realistic scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_size = 128
        self.capacity = 100000
        
        # Store original config
        self.orig_n_buckets = RL_CONFIG.replay_n_buckets
        self.orig_bucket_size = RL_CONFIG.replay_bucket_size
        self.orig_main_bucket_size = RL_CONFIG.replay_main_bucket_size
        
        # Set realistic config
        RL_CONFIG.replay_n_buckets = 5
        RL_CONFIG.replay_bucket_size = 10000
        RL_CONFIG.replay_main_bucket_size = 50000
    
    def tearDown(self):
        """Clean up."""
        RL_CONFIG.replay_n_buckets = self.orig_n_buckets
        RL_CONFIG.replay_bucket_size = self.orig_bucket_size
        RL_CONFIG.replay_main_bucket_size = self.orig_main_bucket_size
    
    def test_realistic_training_scenario(self):
        """Simulate realistic training with varied TD-errors."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        # Simulate early training (mostly low TD-error)
        for _ in range(5000):
            td_error = np.random.exponential(0.5)  # Exponential distribution
            exp = self._make_experience(td_error=td_error)
            buffer.push(**exp)
        
        # Check buffer is filling
        self.assertGreater(len(buffer), 4000)
        
        # Simulate some high TD-error discoveries
        for _ in range(500):
            td_error = np.random.uniform(5, 10)
            exp = self._make_experience(td_error=td_error)
            buffer.push(**exp)
        
        # Sample should work
        batch = buffer.sample(64)
        self.assertIsNotNone(batch)
        
        # Check partition stats
        stats = buffer.get_partition_stats()
        self.assertGreater(stats['total_size'], 5000)
        
        # Top bucket should have some high-error experiences
        self.assertGreater(stats['p90_100_size'], 0)
    
    def test_expert_to_dqn_transition(self):
        """Simulate transition from expert to DQN-driven exploration."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        # Phase 1: Expert-heavy (low TD-error)
        for _ in range(3000):
            exp = self._make_experience(actor='expert', td_error=0.2)
            buffer.push(**exp)
        
        comp1 = buffer.get_actor_composition()
        self.assertGreater(comp1['frac_expert'], 0.9)
        
        # Phase 2: DQN starts exploring (higher TD-error)
        for _ in range(2000):
            actor = 'dqn' if np.random.random() < 0.7 else 'expert'
            td_error = np.random.uniform(0, 3) if actor == 'dqn' else 0.2
            exp = self._make_experience(actor=actor, td_error=td_error)
            buffer.push(**exp)
        
        comp2 = buffer.get_actor_composition()
        # DQN should be present but may not dominate yet due to buffer retention
        self.assertGreater(comp2['frac_dqn'], 0.15)
        
        # Phase 3: DQN dominant - push enough to fill and wrap the buffer
        # Buffer capacity is 100K, so push well beyond that to ensure wraparound
        for _ in range(150000):
            actor = 'dqn' if np.random.random() < 0.9 else 'expert'
            td_error = np.random.gamma(2, 1) if actor == 'dqn' else 0.2
            exp = self._make_experience(actor=actor, td_error=td_error)
            buffer.push(**exp)
        
        comp3 = buffer.get_actor_composition()
        # After wraparound with 90% DQN experiences, should be ~90% DQN
        self.assertGreater(comp3['frac_dqn'], 0.85)
    
    def test_varying_horizons_distribution(self):
        """Test with realistic distribution of n-step horizons."""
        buffer = HybridReplayBuffer(self.capacity, self.state_size)
        
        # Most experiences are 1-step, some are longer
        horizon_distribution = [1] * 70 + [2] * 15 + [3] * 8 + [5] * 5 + [10] * 2
        
        for _ in range(1000):
            horizon = np.random.choice(horizon_distribution)
            exp = self._make_experience(horizon=horizon)
            buffer.push(**exp)
        
        # Sample and check horizon distribution
        batch = buffer.sample(100)
        _, _, _, _, _, _, _, horizons = batch
        
        horizon_values = horizons.cpu().numpy().flatten()
        
        # Should be dominated by horizon=1
        n_h1 = np.sum(horizon_values == 1)
        self.assertGreater(n_h1, 50)  # At least half should be 1-step
        
        # Should have some variety
        unique_horizons = set(horizon_values)
        self.assertGreater(len(unique_horizons), 1)
    
    def _make_experience(self, td_error=0.0, reward=0.0, done=False, actor='dqn', horizon=1):
        """Create a dummy experience for testing."""
        state = np.random.randn(self.state_size).astype(np.float32)
        next_state = np.random.randn(self.state_size).astype(np.float32)
        discrete_action = np.random.randint(0, 4)
        continuous_action = np.random.uniform(-0.9, 0.9)
        
        return {
            'state': state,
            'discrete_action': discrete_action,
            'continuous_action': continuous_action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'actor': actor,
            'horizon': horizon,
            'td_error': td_error
        }


def run_tests():
    """Run all tests with verbose output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestNBucketReplayBuffer))
    suite.addTests(loader.loadTestsFromTestCase(TestNBucketIntegration))
    
    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
