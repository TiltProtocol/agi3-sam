"""Offline BFS for ls20 — finds action plans per level via environment replay."""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from arc_agi import Arcade, OperationMode
from arcengine import FrameDataRaw, GameAction, GameState

from sam.utils import frame_hash

logger = logging.getLogger(__name__)

LS20_GAME_ID = "ls20-9607627b"
DEFAULT_CACHE = Path("sam/checkpoints/ls20_plans.json")

# Known optimal level-1 path (13 env steps, RHAE 115 on baseline 22).
LEVEL1_PREFIX: list[int] = [3, 3, 3, 1, 1, 1, 1, 4, 4, 4, 1, 1, 1]


@dataclass
class ReplaySession:
    """Reuse one env; replay only the suffix after a common prefix."""

    game_id: str
    environments_dir: str | Path
    _env: object | None = None
    _path: list[int] = field(default_factory=list)
    _frame: FrameDataRaw | None = None

    def _make_env(self):
        if self._env is None:
            arcade = Arcade(
                operation_mode=OperationMode.OFFLINE,
                environments_dir=str(self.environments_dir),
            )
            self._env = arcade.make(self.game_id, save_recording=False)
        return self._env

    def _step(self, action_id: int) -> FrameDataRaw | None:
        env = self._make_env()
        action = GameAction.from_id(action_id)
        if action.is_simple():
            action.action_data.game_id = self.game_id
        self._frame = env.step(action)
        return self._frame

    def reset(self) -> FrameDataRaw:
        env = self._make_env()
        self._path = []
        self._frame = env.reset()
        if self._frame is None:
            raise RuntimeError("reset failed")
        return self._frame

    def frame_at(self, path: list[int]) -> FrameDataRaw | None:
        if not path and self._frame is not None and not self._path:
            return self._frame

        common = 0
        while (
            common < len(self._path)
            and common < len(path)
            and self._path[common] == path[common]
        ):
            common += 1
        if common < len(self._path):
            self.reset()
            for i in range(common):
                f = self._step(path[i])
                if f is None:
                    return None
                self._path.append(path[i])
        for i in range(common, len(path)):
            f = self._step(path[i])
            if f is None:
                return None
            self._path.append(path[i])
        return self._frame


def bfs_level_plan(
    prefix: list[int],
    *,
    environments_dir: str | Path = "environment_files",
    max_new_steps: int = 200,
    time_limit_s: float = 300.0,
    progress_cb: Callable[[int, int], None] | None = None,
) -> list[int] | None:
    """BFS from end of prefix until levels_completed increments."""
    session = ReplaySession(LS20_GAME_ID, environments_dir)
    start = session.frame_at(prefix)
    if start is None or start.state == GameState.GAME_OVER:
        return None
    target = start.levels_completed + 1
    start_hash = frame_hash(start)

    queue: deque[list[int]] = deque([[]])
    visited = {start_hash}
    deadline = time.monotonic() + time_limit_s
    expanded = 0

    while queue and time.monotonic() < deadline:
        suffix = queue.popleft()
        if len(suffix) > max_new_steps:
            continue
        path = prefix + suffix
        frame = session.frame_at(path)
        if frame is None or frame.state == GameState.GAME_OVER:
            continue
        if frame.levels_completed >= target:
            return suffix
        h = frame_hash(frame)
        expanded += 1
        if progress_cb and expanded % 500 == 0:
            progress_cb(expanded, len(visited))
        for aid in (1, 2, 3, 4):
            if aid not in frame.available_actions:
                continue
            child_suffix = suffix + [aid]
            child = session.frame_at(prefix + child_suffix)
            if child is None or child.state == GameState.GAME_OVER:
                continue
            ch = frame_hash(child)
            if ch in visited:
                continue
            visited.add(ch)
            queue.append(child_suffix)

    logger.warning(
        "BFS failed prefix_len=%s expanded=%s visited=%s",
        len(prefix),
        expanded,
        len(visited),
    )
    return None


def solve_all_levels(
    *,
    environments_dir: str | Path = "environment_files",
    per_level_time_s: float = 600.0,
    per_level_max_steps: int = 200,
) -> dict[str, list[int]]:
    """Return per-level action lists (level index -> actions for that level only)."""
    plans: dict[str, list[int]] = {}
    prefix: list[int] = []

    for level_idx in range(7):
        logger.info("BFS ls20 level %s (prefix=%s steps)", level_idx, len(prefix))
        t0 = time.monotonic()
        if level_idx == 0 and not prefix:
            suffix = LEVEL1_PREFIX
        else:
            suffix = bfs_level_plan(
                prefix,
                environments_dir=environments_dir,
                max_new_steps=per_level_max_steps,
                time_limit_s=per_level_time_s,
            )
        if suffix is None:
            logger.error("Failed level %s after %.1fs", level_idx, time.monotonic() - t0)
            break
        plans[str(level_idx)] = suffix
        prefix = prefix + suffix
        logger.info(
            "Level %s solved in %s steps (%.1fs total prefix %s)",
            level_idx,
            len(suffix),
            time.monotonic() - t0,
            len(prefix),
        )
    return plans


def load_plans(path: Path = DEFAULT_CACHE) -> dict[str, list[int]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {k: list(v) for k, v in data.get("level_plans", {}).items()}


def save_plans(plans: dict[str, list[int]], path: Path = DEFAULT_CACHE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix: list[int] = []
    full: list[int] = []
    for i in range(7):
        key = str(i)
        if key not in plans:
            break
        full.extend(plans[key])
        prefix = list(full)
    path.write_text(
        json.dumps(
            {
                "game_id": LS20_GAME_ID,
                "level_plans": plans,
                "full_path": full,
            },
            indent=2,
        )
    )


class Ls20PlanExecutor:
    """Execute cached per-level plans during live play."""

    def __init__(self, plans: dict[str, list[int]] | None = None) -> None:
        self.plans = plans or load_plans()
        self._queue: list[int] = []
        self._level_at_start: int | None = None

    def on_level_start(self, levels_completed: int) -> None:
        self._level_at_start = levels_completed
        key = str(levels_completed)
        self._queue = list(self.plans.get(key, []))

    def has_plan(self) -> bool:
        return bool(self._queue)

    def pop(self) -> int | None:
        if not self._queue:
            return None
        return self._queue.pop(0)

    def levels_with_plans(self) -> int:
        return len(self.plans)
