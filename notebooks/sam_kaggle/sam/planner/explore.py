"""Exploration scheduler: prefer untried actions at current state hash."""

from __future__ import annotations

from arcengine import FrameData

from sam.planner.state_graph import StateGraph, frame_hash


class ExplorationScheduler:
    def __init__(self, graph: StateGraph) -> None:
        self.graph = graph
        self._tried: dict[str, set[int]] = {}
        self._round_robin = 0

    def next_action_id(self, frame: FrameData) -> int:
        h = frame_hash(frame)
        tried = self._tried.setdefault(h, set())
        available = [a for a in frame.available_actions if a != 0]
        if not available:
            return 1
        node = self.graph.nodes.get(h)
        if node:
            for aid, edge in node.edges.items():
                if aid in available and edge.levels_after > frame.levels_completed:
                    return aid
        for aid in available:
            if aid not in tried:
                return aid
        aid = available[self._round_robin % len(available)]
        self._round_robin += 1
        return aid

    def mark_tried(self, frame: FrameData, action_id: int) -> None:
        h = frame_hash(frame)
        self._tried.setdefault(h, set()).add(action_id)
