"""Tests for reward anti-hack behavior."""

import numpy as np

from sam.rewards.engine import RewardVectorEngine


def test_curiosity_only_first_visit():
    engine = RewardVectorEngine()
    r1 = engine.compute(
        state_hash="abc",
        prev_hash=None,
        wm_confidence=0.5,
        value_delta=0.0,
        level_before=0,
        level_after=0,
        is_env_step=True,
        is_internal=False,
        game_over=False,
        reset_after_over=False,
        grid_changed=10,
    )
    assert r1.values[0] == 1.0

    r2 = engine.compute(
        state_hash="abc",
        prev_hash="abc",
        wm_confidence=0.5,
        value_delta=0.0,
        level_before=0,
        level_after=0,
        is_env_step=True,
        is_internal=False,
        game_over=False,
        reset_after_over=False,
        grid_changed=0,
    )
    assert r2.values[0] == 0.0


def test_efficiency_penalizes_env_steps():
    engine = RewardVectorEngine()
    r = engine.compute(
        state_hash="x1",
        prev_hash=None,
        wm_confidence=0.5,
        value_delta=0.0,
        level_before=0,
        level_after=0,
        is_env_step=True,
        is_internal=False,
        game_over=False,
        reset_after_over=False,
        grid_changed=5,
    )
    assert r.values[1] == -1.0


def test_efficiency_floor_in_scalar():
    engine = RewardVectorEngine()
    engine.weights = np.zeros(7, dtype=np.float32)
    engine.weights[1] = 0.01
    vec = engine.compute(
        state_hash="h",
        prev_hash=None,
        wm_confidence=0.5,
        value_delta=0.0,
        level_before=0,
        level_after=0,
        is_env_step=True,
        is_internal=False,
        game_over=False,
        reset_after_over=False,
        grid_changed=0,
    )
    scalar = engine.scalar(vec)
    assert scalar <= -0.25
