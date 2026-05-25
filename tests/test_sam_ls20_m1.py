"""Regression: SAM completes ls20 level 1 within per-level action budget."""

from __future__ import annotations

from pathlib import Path

from arc_agi import Arcade, OperationMode

from sam.agent import SamAgent
from sam.config import find_metadata


def test_sam_ls20_level1_m1():
    meta = find_metadata("ls20-9607627b", Path("environment_files"))
    assert meta is not None

    agent = SamAgent(
        game_id="ls20-9607627b",
        environments_dir="environment_files",
        run_dir="sam/runs/test",
        checkpoint=None,
        planning_ms=10.0,
    )
    env = agent.arcade.make("ls20-9607627b", save_recording=False)
    summary = agent.play_episode(env)

    assert summary["levels_completed"] >= 1
    assert summary["rhae"]["levels_completed"] >= 1
    first = summary["rhae"]["level_scores"][0]
    assert first["within_budget"] is True
    assert first["actions"] <= meta.max_actions_for_level(0)
