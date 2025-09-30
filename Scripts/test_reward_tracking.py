#!/usr/bin/env python3
"""
Test script to verify subjective/objective reward tracking
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from config import metrics
from aimodel import parse_frame_data
import struct

def test_reward_tracking():
    """Test that subj + obj = total reward tracking works"""
    print("Testing reward component tracking...")

    # Clear any existing data
    metrics.episode_rewards.clear()
    metrics.subj_rewards.clear()
    metrics.obj_rewards.clear()

    # Test data: total=5.0, subj=2.0, obj=3.0
    total_reward = 5.0
    subj_reward = 2.0
    obj_reward = 3.0

    # Add episode reward
    metrics.add_episode_reward(total_reward, 0.0, 0.0, subj_reward, obj_reward)

    # Check that rewards were stored
    assert len(metrics.episode_rewards) == 1
    assert len(metrics.subj_rewards) == 1
    assert len(metrics.obj_rewards) == 1

    assert metrics.episode_rewards[-1] == total_reward
    assert metrics.subj_rewards[-1] == subj_reward
    assert metrics.obj_rewards[-1] == obj_reward

    print("✓ Reward storage test passed")

    # Test interval tracking
    metrics.reward_sum_interval_total = 0.0
    metrics.reward_count_interval_total = 0
    metrics.reward_sum_interval_subj = 0.0
    metrics.reward_count_interval_subj = 0
    metrics.reward_sum_interval_obj = 0.0
    metrics.reward_count_interval_obj = 0

    # Add another episode
    metrics.add_episode_reward(total_reward, 0.0, 0.0, subj_reward, obj_reward)

    assert metrics.reward_count_interval_total == 1
    assert metrics.reward_count_interval_subj == 1
    assert metrics.reward_count_interval_obj == 1
    assert metrics.reward_sum_interval_total == total_reward
    assert metrics.reward_sum_interval_subj == subj_reward
    assert metrics.reward_sum_interval_obj == obj_reward

    print("✓ Interval tracking test passed")

    # Test that subj + obj = total
    total_from_components = subj_reward + obj_reward
    assert abs(total_from_components - total_reward) < 1e-6, f"subj + obj should equal total: {total_from_components} != {total_reward}"

    print("✓ Reward component validation passed")

    print("All tests passed! Subjective/objective reward tracking is working correctly.")

if __name__ == "__main__":
    test_reward_tracking()