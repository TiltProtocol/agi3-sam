"""Tests for StateGraph BFS planning."""

from arcengine import FrameData, GameState

from sam.planner.state_graph import StateGraph, frame_hash


def _frame(levels: int, grid_val: int = 1) -> FrameData:
    grid = [[grid_val] * 4 for _ in range(4)]
    return FrameData(
        game_id="test",
        frame=[grid],
        state=GameState.NOT_FINISHED,
        levels_completed=levels,
        available_actions=[1, 2, 3, 4],
    )


def test_bfs_finds_level_up_path():
    graph = StateGraph(max_depth=10)
    s0 = _frame(0, 1)
    s1 = _frame(0, 2)
    win = _frame(1, 3)
    graph.reset_level(s0)
    graph.add_transition(s0, 1, s1)
    graph.add_transition(s1, 2, win)
    plan = graph.plan_to_level_up(s0, target_levels=1)
    assert plan == [1, 2]


def test_frame_hash_stable():
    f = _frame(0)
    assert frame_hash(f) == frame_hash(f)
