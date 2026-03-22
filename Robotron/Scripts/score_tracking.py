#!/usr/bin/env python3
"""Server-side Robotron score tracking guards."""

from __future__ import annotations

from typing import Any

# Robotron score awards are quantized to 25-point units (for example 25, 100,
# 150, 200, 500, 1000 in the Williams source tables), so off-grid values are
# a strong signal that we are looking at transient RAM rather than gameplay.
SCORE_GRANULARITY = 25
SCORE_CONFIRMATION_FRAMES = 2
MAX_TRACKED_SCORE = 99_999_999
MAX_SCORE_DELTA_PER_FRAME = 100_000
MAX_TRACKED_LASERS = 99

_RESET_SESSION_REASONS = {"player_dead", "wave_inactive"}


def reset_score_tracking_session(client_state: dict[str, Any]) -> None:
    """Reset transient score tracking state but keep the last committed score."""
    client_state["score_tracking_ready"] = False
    client_state["score_tracking_candidate_score"] = None
    client_state["score_tracking_candidate_frames"] = 0


def _seed_score_candidate(client_state: dict[str, Any], score: int) -> None:
    client_state["score_tracking_candidate_score"] = int(score)
    client_state["score_tracking_candidate_frames"] = 1


def _frame_reject_reason(frame: Any) -> str | None:
    if not bool(getattr(frame, "player_alive", False)):
        return "player_dead"

    score = int(getattr(frame, "game_score", 0) or 0)
    if score < 0 or score > MAX_TRACKED_SCORE:
        return "score_out_of_range"
    if (score % SCORE_GRANULARITY) != 0:
        return "score_not_multiple_of_25"

    wave = int(getattr(frame, "level_number", 0) or 0)
    if wave <= 0:
        return "wave_inactive"

    lasers = int(getattr(frame, "num_lasers", 0) or 0)
    if lasers < 0 or lasers > MAX_TRACKED_LASERS:
        return "lasers_out_of_range"

    return None


def advance_score_tracking(client_state: dict[str, Any], frame: Any) -> tuple[int | None, int | None, str | None]:
    """Advance score tracking for one frame.

    Returns:
      accepted_score:
        A score safe to use for live high-score tracking on this frame.
      completed_game_score:
        A completed prior-game total to add to rolling averages exactly once.
      reject_reason:
        Short code for why this frame was ignored, if any.
    """
    reject_reason = _frame_reject_reason(frame)
    if reject_reason is not None:
        if reject_reason in _RESET_SESSION_REASONS:
            reset_score_tracking_session(client_state)
        elif not bool(client_state.get("score_tracking_ready", False)):
            reset_score_tracking_session(client_state)
        return None, None, reject_reason

    score = int(getattr(frame, "game_score", 0) or 0)
    last_score = int(client_state.get("last_alive_game_score", 0) or 0)
    tracking_ready = bool(client_state.get("score_tracking_ready", False))

    if not tracking_ready:
        candidate_score = client_state.get("score_tracking_candidate_score")
        candidate_frames = int(client_state.get("score_tracking_candidate_frames", 0) or 0)
        if candidate_score is None or candidate_frames <= 0:
            _seed_score_candidate(client_state, score)
            return None, None, None

        delta = score - int(candidate_score)
        if delta < 0 or delta > MAX_SCORE_DELTA_PER_FRAME:
            _seed_score_candidate(client_state, score)
            return None, None, "score_jump_too_large" if delta > MAX_SCORE_DELTA_PER_FRAME else None

        candidate_frames += 1
        client_state["score_tracking_candidate_score"] = score
        client_state["score_tracking_candidate_frames"] = candidate_frames
        if candidate_frames < SCORE_CONFIRMATION_FRAMES:
            return None, None, None

        client_state["score_tracking_ready"] = True
        client_state["score_tracking_candidate_score"] = None
        client_state["score_tracking_candidate_frames"] = 0
        completed_score = last_score if last_score > 0 and score < last_score else None
        return score, completed_score, None

    delta = score - last_score
    if delta < 0:
        completed_score = last_score if last_score > 0 else None
        reset_score_tracking_session(client_state)
        client_state["last_alive_game_score"] = score
        _seed_score_candidate(client_state, score)
        return None, completed_score, None
    if delta > MAX_SCORE_DELTA_PER_FRAME:
        return None, None, "score_jump_too_large"

    return score, None, None
