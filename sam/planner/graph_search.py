"""Online graph search: momentum exploration + graph plan execution."""

from __future__ import annotations

from arcengine import FrameDataRaw, GameAction

from sam.planner.state_graph import StateGraph
from sam.utils import frame_hash, raw_to_frame_data


class GraphSearchExplorer:
    """Explore with run-length momentum; replay graph plans when a level-up path exists."""

    ACTION_ORDER = (3, 1, 4, 2)
    MAX_RUN = {3: 3, 1: 4, 4: 3, 2: 3}
    SWITCH_ORDER = {
        1: (4, 3, 2),
        3: (1, 4, 2),
        4: (1, 3, 2),
        2: (1, 3, 4),
    }

    def __init__(self, graph: StateGraph) -> None:
        self.graph = graph
        self._pending_plan: list[int] = []
        self._root_hash: str | None = None
        self._tried: dict[str, set[int]] = {}
        self._path: list[int] = []
        self._path_hashes: list[str] = []
        self._replay: list[int] = []

    def reset_level(self, root_hash: str | None = None) -> None:
        self._pending_plan = []
        self._tried = {}
        self._path = []
        self._path_hashes = []
        self._replay = []
        if root_hash:
            self._root_hash = root_hash

    def mark_tried(self, state_hash: str, action_id: int) -> None:
        if action_id == GameAction.RESET.value:
            self._path_hashes = []
            return
        self._tried.setdefault(state_hash, set()).add(action_id)

    def has_pending(self) -> bool:
        return bool(self._pending_plan)

    def pop_pending(self) -> int:
        return self._pending_plan.pop(0)

    def set_plan(self, plan: list[int]) -> None:
        self._pending_plan = list(plan)
        self._path = []
        self._path_hashes = []
        self._replay = []

    @classmethod
    def _run_len(cls, path: list[int]) -> int:
        if not path:
            return 0
        last = path[-1]
        count = 0
        for aid in reversed(path):
            if aid == last:
                count += 1
            else:
                break
        return count

    @classmethod
    def pick_action(cls, untried: list[int], path: list[int]) -> int:
        if path:
            last = path[-1]
            run = cls._run_len(path)
            cap = cls.MAX_RUN.get(last, 3)
            if run >= cap:
                for aid in cls.SWITCH_ORDER.get(last, cls.ACTION_ORDER):
                    if aid in untried and aid != last:
                        return aid
            elif last in untried:
                return last
        for aid in cls.ACTION_ORDER:
            if aid in untried:
                return aid
        return untried[0]

    def _available(self, frame: FrameDataRaw, state_hash: str) -> list[int]:
        node = self.graph.nodes.get(state_hash)
        if node and node.frame_snapshot:
            return [a for a in node.frame_snapshot.available_actions if a != 0]
        return [a for a in frame.available_actions if a != 0]

    def _untried(self, state_hash: str, available: list[int]) -> list[int]:
        tried = set(self._tried.get(state_hash, set()))
        node = self.graph.nodes.get(state_hash)
        if node:
            tried |= set(node.edges.keys())
        return [a for a in available if a not in tried]

    def _backtrack(self) -> None:
        if self._path:
            self._path.pop()
        if self._path_hashes:
            self._path_hashes.pop()

    def next_action(self, frame: FrameDataRaw, *, target_levels: int) -> tuple[int, str]:
        fd = raw_to_frame_data(frame)
        state_hash = frame_hash(frame)

        plan = self.graph.plan_to_level_up(fd, target_levels)
        if plan and not self._pending_plan:
            self.set_plan(plan)
            return plan[0], "graph_plan"

        if self._replay:
            aid = self._replay.pop(0)
            return aid, "replay"

        if state_hash == self._root_hash and self._path and not self._path_hashes:
            self._replay = list(self._path)
            if self._replay:
                aid = self._replay.pop(0)
                return aid, "replay"

        if state_hash in self._path_hashes:
            self._backtrack()
            self._replay = list(self._path)
            return GameAction.RESET.value, "loop_backtrack"

        self._path_hashes.append(state_hash)
        available = self._available(frame, state_hash)
        untried = self._untried(state_hash, available)
        if not untried:
            self._backtrack()
            self._replay = list(self._path)
            return GameAction.RESET.value, "dead_end_reset"

        aid = self.pick_action(untried, self._path)
        self._path.append(aid)
        return aid, "momentum_expand"
