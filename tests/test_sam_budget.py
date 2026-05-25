"""Action budget = 5x baseline tests."""

from sam.config import BUDGET_MULTIPLIER, GameMetadata


def test_ls20_level_budgets():
    meta = GameMetadata(
        game_id="ls20-9607627b",
        baseline_actions=[22, 123, 73, 84, 96, 192, 186],
    )
    assert meta.max_actions_for_level(0) == 5 * 22
    assert meta.max_actions_for_level(6) == 5 * 186
    assert meta.max_actions_for_level(0) == 110


def test_budget_multiplier_constant():
    assert BUDGET_MULTIPLIER == 5


def test_agent_max_actions_from_metadata():
    from pathlib import Path

    from sam.config import find_metadata

    root = Path(__file__).resolve().parents[1] / "environment_files"
    meta = find_metadata("ls20-9607627b", root)
    assert meta is not None
    assert meta.max_actions_for_level(0) == 110
