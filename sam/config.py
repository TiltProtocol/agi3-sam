"""Configuration loading and action budget helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

BUDGET_MULTIPLIER = 5
DEFAULT_PLANNING_MS = 50.0
MAX_SKILL_MODULES = 8
MAX_TOTAL_PARAMS = 3_000_000


@dataclass
class GameMetadata:
    game_id: str
    title: str = ""
    baseline_actions: list[int] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    local_dir: str = ""

    @classmethod
    def from_json_path(cls, path: Path) -> GameMetadata:
        data = json.loads(path.read_text())
        return cls(
            game_id=data.get("game_id", ""),
            title=data.get("title", ""),
            baseline_actions=list(data.get("baseline_actions", [])),
            tags=list(data.get("tags", [])),
            local_dir=data.get("local_dir", ""),
        )

    def max_actions_for_level(self, level_index: int) -> int:
        if not self.baseline_actions:
            return 80
        idx = min(max(level_index, 0), len(self.baseline_actions) - 1)
        return BUDGET_MULTIPLIER * self.baseline_actions[idx]

    def baseline_for_level(self, level_index: int) -> int:
        if not self.baseline_actions:
            return 0
        idx = min(max(level_index, 0), len(self.baseline_actions) - 1)
        return self.baseline_actions[idx]


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def find_metadata(game_id: str, environments_dir: Path) -> GameMetadata | None:
    """Locate metadata.json for a game id like ls20-9607627b."""
    base = game_id.split("-")[0] if "-" in game_id else game_id
    for meta_path in environments_dir.rglob("metadata.json"):
        data = json.loads(meta_path.read_text())
        gid = data.get("game_id", "")
        if gid == game_id or gid.startswith(base):
            return GameMetadata.from_json_path(meta_path)
    return None
