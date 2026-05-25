"""Tests for 5× baseline action budget."""

from pathlib import Path

from sam.config import BUDGET_MULTIPLIER, GameMetadata, find_metadata


def test_budget_multiplier_constant():
    assert BUDGET_MULTIPLIER == 5


def test_ls20_level1_budget():
    root = Path(__file__).resolve().parents[1]
    meta = find_metadata("ls20-9607627b", root / "environment_files")
    assert meta is not None
    assert meta.baseline_for_level(0) == 22
    assert meta.max_actions_for_level(0) == 110


def test_ls20_level7_budget():
    root = Path(__file__).resolve().parents[1]
    meta = find_metadata("ls20-9607627b", root / "environment_files")
    assert meta is not None
    assert meta.baseline_for_level(6) == 186
    assert meta.max_actions_for_level(6) == 930
