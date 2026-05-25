"""Internal planner: AttemptSolve → DetectNovelty → Investigate → InternalReplay → LiveReset."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import torch

from arcengine import FrameDataRaw, GameAction, GameState

from sam.core.sam_core import SamCore
from sam.planner.branching_explorer import BranchingExplorer
from sam.planner.explore import ExplorationScheduler
from sam.planner.novelty import detect_novelty
from sam.planner.state_graph import StateGraph
from sam.utils import frame_hash, frame_to_model_input, raw_to_frame_data


class PlannerPhase(Enum):
    ATTEMPT_SOLVE = auto()
    DETECT_NOVELTY = auto()
    INVESTIGATE = auto()
    INTERNAL_REPLAY = auto()
    LIVE_RESET = auto()
    EXECUTE_LIVE = auto()


@dataclass
class PlannerState:
    phase: PlannerPhase = PlannerPhase.ATTEMPT_SOLVE
    plan: list[int] = field(default_factory=list)
    plan_index: int = 0
    investigation_steps: int = 0
    wm_errors: list[float] = field(default_factory=list)
    internal_steps: int = 0


class InternalPlanner:
    def __init__(
        self,
        core: SamCore,
        graph: StateGraph,
        *,
        max_depth: int = 15,
        planning_ms: float = 50.0,
        max_investigation: int = 8,
    ) -> None:
        self.core = core
        self.graph = graph
        self.max_depth = max_depth
        self.planning_ms = planning_ms
        self.max_investigation = max_investigation
        self.state = PlannerState()
        self.visited_hashes: set[str] = set()
        self.explorer = ExplorationScheduler(graph)
        self.brancher = BranchingExplorer(graph)

    def reset_level(self, frame: FrameDataRaw) -> None:
        self.state = PlannerState()
        fd = raw_to_frame_data(frame)
        self.graph.reset_level(fd)
        self.visited_hashes = {frame_hash(frame)}
        self.brancher.reset_level()

    def _goal_levels(self, frame: FrameDataRaw) -> int:
        return frame.levels_completed + 1

    def attempt_solve(self, frame: FrameDataRaw) -> list[int] | None:
        deadline = time.monotonic() + self.planning_ms / 1000.0
        fd = raw_to_frame_data(frame)
        target = self._goal_levels(frame)
        plan = self.graph.plan_to_level_up(fd, target)
        if plan is not None:
            return plan

        # WM-augmented beam: expand unseen edges with world model
        start = frame_hash(frame)
        z = self.core.encode(frame_to_model_input(frame))
        beam: list[tuple[list[int], str, torch.Tensor]] = [( [], start, z)]
        best: list[int] | None = None
        for depth in range(min(self.max_depth, 8)):
            if time.monotonic() > deadline:
                break
            next_beam: list[tuple[list[int], str, torch.Tensor]] = []
            for path, node_hash, z_cur in beam:
                node = self.graph.nodes.get(node_hash)
                avail = list(frame.available_actions)
                if node and node.frame_snapshot:
                    avail = list(node.frame_snapshot.available_actions)
                for aid in avail:
                    if node and aid in node.edges:
                        nh = node.edges[aid].next_hash
                        nnode = self.graph.nodes.get(nh)
                        if nnode and nnode.frame_snapshot:
                            if nnode.frame_snapshot.levels_completed >= target:
                                return path + [aid]
                        next_beam.append((path + [aid], nh, z_cur))
                    else:
                        z_next, conf = self.core.world_model(z_cur, aid)
                        z_next = self.core.skills.apply(z_next)
                        if conf.item() > 0.4 and depth < 5:
                            next_beam.append((path + [aid], f"sim_{aid}_{depth}", z_next))
            beam = next_beam[:32]
        return best

    def choose_action(
        self,
        frame: FrameDataRaw,
        *,
        level_actions_used: int,
        max_actions: int,
        prev_frame: FrameDataRaw | None = None,
    ) -> tuple[int, PlannerPhase, dict]:
        """Return (action_id, phase, debug_info). Uses env steps only when executing."""
        cur_hash = frame_hash(frame)
        info: dict = {"phase": self.state.phase.name}

        if self.state.plan and self.state.plan_index < len(self.state.plan):
            aid = self.state.plan[self.state.plan_index]
            self.state.plan_index += 1
            self.state.phase = PlannerPhase.EXECUTE_LIVE
            info["executing_plan"] = True
            return aid, PlannerPhase.EXECUTE_LIVE, info

        plan = self.attempt_solve(frame)
        if plan:
            self.brancher.set_plan(plan)
            aid = self.brancher.pop_pending()
            self.state.phase = PlannerPhase.ATTEMPT_SOLVE
            info["plan_len"] = len(plan)
            return aid, PlannerPhase.ATTEMPT_SOLVE, info

        self.visited_hashes.add(cur_hash)

        if self.brancher.has_pending():
            aid = self.brancher.pop_pending()
            self.state.phase = PlannerPhase.EXECUTE_LIVE
            info["branch_plan"] = True
            return aid, PlannerPhase.EXECUTE_LIVE, info

        target = self._goal_levels(frame)
        aid, reason = self.brancher.next_action(frame, target_levels=target)
        self.state.phase = PlannerPhase.DETECT_NOVELTY
        info["branch_reason"] = reason
        if aid == GameAction.RESET.value:
            self.state.phase = PlannerPhase.LIVE_RESET
        return aid, self.state.phase, info

    def should_reset(
        self,
        frame: FrameDataRaw,
        level_actions_used: int,
        max_actions: int,
    ) -> bool:
        if frame.state == GameState.GAME_OVER:
            return True
        return False

    @staticmethod
    def reset_action_id() -> int:
        return GameAction.RESET.value
