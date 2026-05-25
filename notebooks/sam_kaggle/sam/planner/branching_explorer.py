"""Systematic graph exploration with reset-based backtracking."""

from __future__ import annotations

from collections import deque

from arcengine import GameAction

from sam.planner.state_graph import StateGraph
from sam.utils import frame_hash, raw_to_frame_data


class BranchingExplorer:
    """Online BFS: expand untried actions; backtrack via RESET when dead-end."""

    def __init__(self, graph: StateGraph) -> None:
        self.graph = graph
        self._tried: dict[str, set[int]] = {}
        self._pending_plan: list[int] = []
        self._root_hash: str | None = None

    def reset_level(self, root_hash: str | None = None) -> None:
        self._tried.clear()
        self._pending_plan = []
        if root_hash:
            self._root_hash = root_hash

    def mark_tried(self, frame_hash_val: str, action_id: int) -> None:
        self._tried.setdefault(frame_hash_val, set()).add(action_id)

    def has_pending(self) -> bool:
        return len(self._pending_plan) > 0

    def pop_pending(self) -> int:
        return self._pending_plan.pop(0)

    def set_plan(self, plan: list[int]) -> None:
        self._pending_plan = list(plan)

    def _untried(self, h: str, available: list[int]) -> list[int]:
        tried = self._tried.get(h, set())
        return [a for a in available if a != 0 and a not in tried]

    def _path_to_untried(self, start: str) -> list[int] | None:
        """BFS in graph from start to any node with untried outgoing actions."""
        if start not in self.graph.nodes:
            return None
        queue: deque[tuple[str, list[int]]] = deque([(start, [])])
        visited = {start}
        while queue:
            h, path = queue.popleft()
            if len(path) > 20:
                continue
            node = self.graph.nodes.get(h)
            if not node:
                continue
            avail = (
                list(node.frame_snapshot.available_actions)
                if node.frame_snapshot
                else [1, 2, 3, 4]
            )
            if self._untried(h, avail):
                return path
            for aid, edge in node.edges.items():
                nh = edge.next_hash
                if nh not in visited:
                    visited.add(nh)
                    queue.append((nh, path + [aid]))
        return None

    def next_action(
        self,
        frame,
        *,
        target_levels: int,
    ) -> tuple[int, str]:
        fd = raw_to_frame_data(frame)
        h = frame_hash(frame)
        available = [a for a in frame.available_actions if a != 0]

        node = self.graph.nodes.get(h)
        if node:
            for aid, edge in node.edges.items():
                if aid in available and edge.levels_after >= target_levels:
                    return aid, "graph_win_edge"

        plan = self.graph.plan_to_level_up(fd, target_levels)
        if plan:
            self._pending_plan = plan[1:]
            return plan[0], "bfs_plan"

        untried = self._untried(h, available)
        if untried:
            return untried[0], "untried_edge"

        # Navigate via graph to a state that still has untried actions
        if self._root_hash:
            nav = self._path_to_untried(self._root_hash)
            if nav is not None:
                self._pending_plan = nav
                return GameAction.RESET.value, "nav_reset"

        return GameAction.RESET.value, "backtrack_reset"
