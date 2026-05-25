"""Systematic graph exploration with RESET + prefix replay (BFS over action paths)."""

from __future__ import annotations

from collections import deque

from arcengine import GameAction

from sam.planner.state_graph import StateGraph
from sam.utils import frame_hash, raw_to_frame_data


class BranchingExplorer:
    """BFS over action paths: replay a prefix, try the next untried edge, RESET when stuck."""

    def __init__(self, graph: StateGraph) -> None:
        self.graph = graph
        self._tried: dict[str, set[int]] = {}
        self._pending_plan: list[int] = []
        self._root_hash: str | None = None
        self._frontier: deque[list[int]] = deque([[]])
        self._replay: list[int] = []
        self._current_path: list[int] = []

    def reset_level(self, root_hash: str | None = None) -> None:
        self._tried.clear()
        self._pending_plan = []
        self._replay = []
        self._current_path = []
        self._frontier = deque([[]])
        if root_hash:
            self._root_hash = root_hash

    def mark_tried(self, frame_hash_val: str, action_id: int) -> None:
        if action_id == GameAction.RESET.value:
            self._current_path = []
            return
        self._tried.setdefault(frame_hash_val, set()).add(action_id)

    def has_pending(self) -> bool:
        return len(self._pending_plan) > 0

    def pop_pending(self) -> int:
        return self._pending_plan.pop(0)

    def set_plan(self, plan: list[int]) -> None:
        self._pending_plan = list(plan)
        self._replay = []
        self._current_path = []

    def _untried(self, h: str, available: list[int]) -> list[int]:
        tried = set(self._tried.get(h, set()))
        node = self.graph.nodes.get(h)
        if node:
            tried |= set(node.edges.keys())
        return [a for a in available if a != 0 and a not in tried]

    def _available(self, h: str, frame) -> list[int]:
        node = self.graph.nodes.get(h)
        if node and node.frame_snapshot:
            return [a for a in node.frame_snapshot.available_actions if a != 0]
        return [a for a in frame.available_actions if a != 0]

    def _target_path(self) -> list[int]:
        return self._frontier[0] if self._frontier else []

    def _schedule_replay(self, path: list[int]) -> None:
        self._replay = list(path)
        self._current_path = []

    def next_action(self, frame, *, target_levels: int) -> tuple[int, str]:
        fd = raw_to_frame_data(frame)
        h = frame_hash(frame)

        if self._replay:
            aid = self._replay.pop(0)
            self._current_path.append(aid)
            return aid, "replay"

        plan = self.graph.plan_to_level_up(fd, target_levels)
        if plan:
            self._pending_plan = plan[1:]
            self._current_path = []
            return plan[0], "bfs_plan"

        node = self.graph.nodes.get(h)
        if node:
            for aid, edge in node.edges.items():
                if aid in self._available(h, frame) and edge.levels_after >= target_levels:
                    return aid, "graph_win_edge"

        if not self._frontier:
            return GameAction.RESET.value, "exhausted"

        target = self._target_path()

        if self._current_path != target:
            if h == self._root_hash and not self._current_path:
                self._schedule_replay(target)
                if self._replay:
                    aid = self._replay.pop(0)
                    self._current_path.append(aid)
                    return aid, "replay"
            self._schedule_replay(target)
            return GameAction.RESET.value, "resync_reset"

        if len(self._current_path) >= self.graph.max_depth:
            self._frontier.popleft()
            self._current_path = []
            if not self._frontier:
                return GameAction.RESET.value, "depth_exhausted"
            self._schedule_replay(self._target_path())
            return GameAction.RESET.value, "depth_backtrack"

        untried = self._untried(h, self._available(h, frame))
        if untried:
            aid = untried[0]
            for other in untried[1:]:
                alt = target + [other]
                if alt not in self._frontier:
                    self._frontier.append(alt)
            new_path = target + [aid]
            self._frontier[0] = new_path
            self._current_path = new_path
            return aid, "untried_edge"

        self._frontier.popleft()
        self._current_path = []
        if not self._frontier:
            return GameAction.RESET.value, "frontier_empty"
        self._schedule_replay(self._target_path())
        return GameAction.RESET.value, "next_frontier"
