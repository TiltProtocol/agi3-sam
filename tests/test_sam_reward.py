"""Reward vector anti-hack tests."""

from arcengine import FrameData, GameState

from sam.planner.state_graph import frame_hash
from sam.rewards.vector import RewardVectorEngine, EFFICIENCY_FLOOR


def _frame(grid=None, levels=0, actions=None):
    g = grid or [[0] * 4 for _ in range(4)]
    return FrameData(
        frame=[g],
        levels_completed=levels,
        available_actions=actions or [1, 2, 3, 4],
        state=GameState.NOT_FINISHED,
    )


def test_curiosity_capped_on_revisit():
    engine = RewardVectorEngine()
    a = _frame([[1, 0], [0, 0]])
    b = _frame([[1, 0], [0, 0]])
    r1 = engine.step_env(a, b, 1)
    assert r1.curiosity == 1.0
    r2 = engine.step_env(b, a, 2)
    assert r2.curiosity == 0.0


def test_efficiency_always_negative_on_env_step():
    engine = RewardVectorEngine()
    a = _frame()
    b = _frame()
    rv = engine.step_env(a, b, 1)
    assert rv.efficiency == -1.0


def test_proximity_does_not_rise_on_loop():
    engine = RewardVectorEngine()
    grid = [[1, 0], [0, 0]]
    f = _frame(grid)
    h = frame_hash(f)
    engine.hash_visits[h] = 5
    before = f
    after = _frame(grid)
    rv = engine.step_env(before, after, 1, value_delta=0.5)
    assert rv.proximity == 0.0


def test_efficiency_weight_floor():
    from sam.rewards.vector import RewardVector

    rv = RewardVector(efficiency=-1.0, curiosity=5.0)
    w = __import__("numpy").ones(7)
    w[1] = EFFICIENCY_FLOOR
    score = rv.weighted_sum(w)
    assert score < 5.0
