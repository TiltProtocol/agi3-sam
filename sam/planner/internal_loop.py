"""Internal reasoning loop: AttemptSolve → DetectNovelty → Investigate → InternalReplay → LiveReset."""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable

import torch

from arcengine import FrameData, GameAction, GameState

from sam.planner.novelty import detect_novelty
from sam.planner.state_graph import StateGraph, frame_hash

if TYPE_CHECKING:
    from sam.core.encoder import GridEncoder
    from sam.core.policy import ActionHead
    from sam.core.world_model import WorldModel
    from sam.growth.bank import SkillModuleBank
    from sam.rewards.vector import RewardVectorEngine


class LoopPhase(Enum):
    ATTEMPT_SOLVE = auto()
    DETECT_NOVELTY = auto()
    INVESTIGATE = auto()
    INTERNAL_REPLAY = auto()
    LIVE_RESET = auto()
    EXECUTE_LIVE = auto()


class InternalLoop:
    def __init__(
        self,
        graph: StateGraph,
        encoder: GridEncoder,
        world_model: WorldModel,
        policy: ActionHead,
        skill_bank: SkillModuleBank,
        reward_engine: RewardVectorEngine,
        planning_ms: float = 50.0,
        max_investigate_steps: int = 8,
    ) -> None:
        self.graph = graph
        self.encoder = encoder
        self.world_model = world_model
        self.policy = policy
        self.skill_bank = skill_bank
        self.reward_engine = reward_engine
        self.planning_ms = planning_ms
        self.max_investigate_steps = max_investigate_steps
        self.phase = LoopPhase.ATTEMPT_SOLVE
        self._planned_actions: list[int] = []
        self._plan_index = 0
        self._investigate_budget = 0
        self._wm_errors: list[float] = []
        self.visited_hashes: set[str] = set()

    def reset_level(self, frame: FrameData) -> None:
        self.graph.reset_level(frame)
        self.phase = LoopPhase.ATTEMPT_SOLVE
        self._planned_actions = []
        self._plan_index = 0
        self._investigate_budget = 0
        h = frame_hash(frame)
        self.visited_hashes.add(h)

    def _encode(self, frame: FrameData) -> torch.Tensor:
        with torch.no_grad():
            z = self.encoder(frame.frame)
            z = self.skill_bank.apply(z)
        return z

    def _plan_confidence(self, plan: list[int] | None) -> float:
        if not plan:
            return 0.0
        covered = sum(
            1
            for i in range(len(plan) - 1)
            if self._edge_exists_at_step(plan, i)
        )
        return covered / max(len(plan), 1)

    def _edge_exists_at_step(self, plan: list[int], i: int) -> bool:
        return True

    def attempt_solve(
        self, frame: FrameData, target_levels: int | None = None
    ) -> tuple[list[int] | None, float]:
        deadline = time.perf_counter() + self.planning_ms / 1000.0
        target = (frame.levels_completed + 1) if target_levels is None else target_levels
        plan = self.graph.plan_to_level_up(frame, target)
        if plan and time.perf_counter() < deadline:
            return plan, self._plan_confidence(plan)
        start = frame_hash(frame)
        if start not in self.graph.nodes:
            return None, 0.0
        z = self._encode(frame)
        sim_plan: list[int] = []
        cur_z = z
        depth = 0
        while depth < self.graph.max_depth and time.perf_counter() < deadline:
            best_action = None
            best_score = -1e9
            for aid in frame.available_actions:
                z_next, conf = self.world_model(cur_z, aid)
                score = float(conf.item()) if hasattr(conf, "item") else float(conf)
                if score > best_score:
                    best_score = score
                    best_action = aid
            if best_action is None:
                break
            sim_plan.append(best_action)
            cur_z = z_next
            depth += 1
        return (sim_plan if sim_plan else None), 0.3

    def internal_simulate_plan(
        self, frame: FrameData, plan: list[int]
    ) -> tuple[bool, list[tuple[int, bool]]]:
        z = self._encode(frame)
        steps: list[tuple[int, bool]] = []
        for aid in plan:
            z_next, conf = self.world_model(z, aid)
            err = float(torch.nn.functional.mse_loss(z_next, z).item())
            self._wm_errors.append(err)
            self.reward_engine.step_internal(aid, wm_confidence=float(conf))
            steps.append((aid, err < 0.5))
            z = z_next
        success = len(steps) > 0 and steps[-1][1]
        self.skill_bank.on_wm_plateau(self._wm_errors)
        return success, steps

    def choose_action(
        self,
        frames: list[FrameData],
        latest: FrameData,
        game_id: str,
        fallback: Callable[[list[FrameData], FrameData], GameAction],
    ) -> GameAction:
        h = frame_hash(latest)
        prev = frames[-2] if len(frames) >= 2 else None
        is_novel, _ = detect_novelty(latest, self.visited_hashes, h, prev)
        self.visited_hashes.add(h)
        self.skill_bank.on_novel_state(is_novel)

        if self._plan_index < len(self._planned_actions):
            aid = self._planned_actions[self._plan_index]
            self._plan_index += 1
            self.phase = LoopPhase.EXECUTE_LIVE
            return self.policy.to_game_action(aid, game_id)

        if self.phase == LoopPhase.ATTEMPT_SOLVE:
            plan, conf = self.attempt_solve(latest)
            if plan and conf >= 0.5:
                self._planned_actions = plan
                self._plan_index = 0
                self.phase = LoopPhase.EXECUTE_LIVE
                return self.choose_action(frames, latest, game_id, fallback)
            self.phase = LoopPhase.DETECT_NOVELTY

        if self.phase == LoopPhase.DETECT_NOVELTY:
            if is_novel:
                self.phase = LoopPhase.INVESTIGATE
                self._investigate_budget = self.max_investigate_steps
            else:
                self.phase = LoopPhase.INTERNAL_REPLAY

        if self.phase == LoopPhase.INVESTIGATE and self._investigate_budget > 0:
            self._investigate_budget -= 1
            z = self._encode(latest)
            aid = self.policy.select_action(z, latest.available_actions, temperature=0.8)
            self.phase = LoopPhase.ATTEMPT_SOLVE
            return self.policy.to_game_action(aid, game_id)

        if self.phase == LoopPhase.INTERNAL_REPLAY:
            plan, _ = self.attempt_solve(latest)
            if plan:
                ok, _ = self.internal_simulate_plan(latest, plan)
                if ok:
                    self._planned_actions = plan
                    self._plan_index = 0
                    self.phase = LoopPhase.EXECUTE_LIVE
                    return self.choose_action(frames, latest, game_id, fallback)
            self.phase = LoopPhase.LIVE_RESET

        if self.phase == LoopPhase.LIVE_RESET:
            if latest.state not in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                self.phase = LoopPhase.ATTEMPT_SOLVE
                return GameAction.RESET
            self.phase = LoopPhase.ATTEMPT_SOLVE

        return fallback(frames, latest)
