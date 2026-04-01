#!/usr/bin/env python3
"""Robotron AI v3 — Reward shaping module.

Transforms raw objective/subjective rewards from the Lua wire protocol
into a shaped reward signal for PPO training.

Reward components:
  - Survival: +bonus per frame alive
  - Score: log-scaled score delta for dense kill signal
  - Human rescue: large bonus per rescue (scaled by progressive multiplier)
  - Death penalty: strong negative on terminal frame
  - Proximity penalty: gentle penalty for being near enemies
"""

import math
import numpy as np
from .config import CONFIG, TrainConfig


class RewardShaper:
    """Transforms raw rewards into shaped training signal."""

    def __init__(self, cfg: TrainConfig = None):
        self.cfg = cfg or CONFIG.train

    def shape(
        self,
        obj_reward: float,
        subj_reward: float,
        done: bool,
        player_alive: bool,
        score_delta: float = 0.0,
        nearest_enemy_dist: float = 1.0,
        humans_rescued_this_frame: int = 0,
        human_bonus_level: int = 1,
    ) -> float:
        """Compute shaped reward from frame data.

        Args:
            obj_reward: raw objective reward from Lua (score-based)
            subj_reward: raw subjective reward from Lua (heuristic)
            done: True on death/episode end
            player_alive: whether player is currently alive
            score_delta: change in game score this frame
            nearest_enemy_dist: distance to nearest enemy (normalized 0-1)
            humans_rescued_this_frame: how many humans just picked up
            human_bonus_level: progressive rescue bonus level (1-5)

        Returns:
            float: shaped reward, clipped to [-reward_clip, +reward_clip]
        """
        r = 0.0
        cfg = self.cfg

        # Survival bonus
        if player_alive and not done:
            r += cfg.survival_bonus

        # Score-based reward (log-scaled for density)
        if score_delta > 0:
            r += cfg.score_log_scale * math.log1p(score_delta)

        # Human rescue bonus
        if humans_rescued_this_frame > 0:
            # Progressive bonus: 1000, 2000, 3000, 4000, 5000
            bonus_multiplier = min(5, human_bonus_level)
            r += cfg.human_rescue_bonus * bonus_multiplier * humans_rescued_this_frame

        # Proximity penalty (encourages "safety bubble")
        if player_alive and nearest_enemy_dist < 1.0:
            inv_dist = 1.0 / max(nearest_enemy_dist, 0.01)
            r -= cfg.proximity_penalty_scale * inv_dist

        # Death penalty
        if done:
            r -= cfg.death_penalty

        # Clip
        r = max(-cfg.reward_clip, min(cfg.reward_clip, r))
        return r

    def shape_simple(
        self,
        obj_reward: float,
        subj_reward: float,
        done: bool,
    ) -> float:
        """Simplified shaping using just the raw Lua rewards.

        When full entity-level detail isn't available, fall back to
        scaling and combining the Lua-provided objective and subjective.
        """
        cfg = self.cfg

        # Objective reward dominates
        r = obj_reward * 0.03 + subj_reward * 0.001

        # Survival bonus when not dying
        if not done:
            r += cfg.survival_bonus

        # Death penalty
        if done:
            r -= cfg.death_penalty

        return max(-cfg.reward_clip, min(cfg.reward_clip, r))


# Module-level singleton
_shaper: RewardShaper = None

def shape_reward(obj_reward: float, subj_reward: float, done: bool) -> float:
    global _shaper
    if _shaper is None:
        _shaper = RewardShaper()
    return _shaper.shape_simple(obj_reward, subj_reward, done)
