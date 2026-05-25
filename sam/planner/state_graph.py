"""State graph: hash frames + BFS planning on real transitions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from arcengine import FrameData, GameAction

from sam.utils import frame_hash


@dataclass
class GraphEdge:
    action_id: int
    next_hash: str
    levels_after: int


@dataclass
class GraphNode:
    hash: str
    frame_snapshot: FrameData | None = None
    edges: dict[int, GraphEdge] = field(default_factory=dict)
    parent: str | None = None
    parent_action: int | None = None
    depth: int = 0


class StateGraph:
    """BFS planner over discovered real transitions."""

    def __init__(self, max_depth: int = 15) -> None:
        self.max_depth = max_depth
        self.nodes: dict[str, GraphNode] = {}
        self.root_hash: str | None = None
        self.level_roots: dict[int, str] = {}

    def reset_level(self, frame: FrameData) -> None:
        h = frame_hash(frame)
        self.root_hash = h
        self.level_roots[frame.levels_completed] = h
        if h not in self.nodes:
            self.nodes[h] = GraphNode(hash=h, frame_snapshot=frame, depth=0)

    def add_transition(
        self, before: FrameData, action_id: int, after: FrameData
    ) -> None:
        h_before = frame_hash(before)
        h_after = frame_hash(after)
        if h_before not in self.nodes:
            self.nodes[h_before] = GraphNode(hash=h_before, frame_snapshot=before)
        if h_after not in self.nodes:
            self.nodes[h_after] = GraphNode(
                hash=h_after,
                frame_snapshot=after,
                parent=h_before,
                parent_action=action_id,
                depth=self.nodes[h_before].depth + 1,
            )
        self.nodes[h_before].edges[action_id] = GraphEdge(
            action_id=action_id,
            next_hash=h_after,
            levels_after=after.levels_completed,
        )

    def bfs_plan(
        self,
        start_hash: str,
        goal_fn: Callable[[str], bool],
        available_fn: Callable[[str], list[int]],
    ) -> list[int] | None:
        if start_hash not in self.nodes:
            return None
        queue: deque[tuple[str, list[int]]] = deque([(start_hash, [])])
        visited = {start_hash}
        while queue:
            node_hash, path = queue.popleft()
            if goal_fn(node_hash):
                return path
            if len(path) >= self.max_depth:
                continue
            node = self.nodes.get(node_hash)
            if not node:
                continue
            for action_id, edge in node.edges.items():
                if action_id not in available_fn(node_hash):
                    continue
                nh = edge.next_hash
                if nh in visited:
                    continue
                visited.add(nh)
                queue.append((nh, path + [action_id]))
        return None

    def plan_to_level_up(self, current: FrameData, target_levels: int) -> list[int] | None:
        start = frame_hash(current)

        def goal(h: str) -> bool:
            n = self.nodes.get(h)
            if not n or not n.frame_snapshot:
                return False
            return n.frame_snapshot.levels_completed >= target_levels

        def avail(node_hash: str) -> list[int]:
            node = self.nodes.get(node_hash)
            if node and node.frame_snapshot:
                return list(node.frame_snapshot.available_actions)
            return list(current.available_actions)

        return self.bfs_plan(start, goal, avail)

    def novelty_rate(self, window: int = 20) -> float:
        recent = list(self.nodes.keys())[-window:]
        if not recent:
            return 0.0
        return len(recent) / window
