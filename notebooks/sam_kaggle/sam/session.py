"""Per-game session tracking level budgets and learning events."""

from __future__ import annotations

from dataclasses import dataclass, field

from arcengine import FrameData, GameState

from sam.config import GameMetadata
from sam.rewards.meta_head import MetaRewardHead


@dataclass
class LevelStats:
    index: int
    env_actions: int = 0
    completed: bool = False
    resets: int = 0


@dataclass
class PerGameSession:
    metadata: GameMetadata
    current_level: int = 0
    level_stats: list[LevelStats] = field(default_factory=list)
    meta_reward: MetaRewardHead = field(default_factory=MetaRewardHead)
    replay_buffer: list[tuple[FrameData, int, FrameData]] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.metadata.baseline_actions) or 7
        self.level_stats = [LevelStats(index=i) for i in range(n)]

    def max_actions(self) -> int:
        return self.metadata.max_actions_for_level(self.current_level)

    def on_frame(self, frame: FrameData) -> None:
        self.current_level = min(frame.levels_completed, len(self.level_stats) - 1)

    def record_action(self, before: FrameData, action_id: int, after: FrameData) -> None:
        idx = min(before.levels_completed, len(self.level_stats) - 1)
        self.level_stats[idx].env_actions += 1
        if after.levels_completed > before.levels_completed:
            if idx < len(self.level_stats):
                self.level_stats[idx].completed = True
        self.replay_buffer.append((before, action_id, after))
        if len(self.replay_buffer) > 5000:
            self.replay_buffer = self.replay_buffer[-2000:]

    def level_completed(self, before: FrameData, after: FrameData) -> bool:
        return after.levels_completed > before.levels_completed

    def game_won(self, frame: FrameData) -> bool:
        return frame.state == GameState.WIN

    def export_level_summary(self) -> tuple[list[int], list[bool]]:
        actions = [s.env_actions for s in self.level_stats]
        completed = [s.completed for s in self.level_stats]
        return actions, completed
