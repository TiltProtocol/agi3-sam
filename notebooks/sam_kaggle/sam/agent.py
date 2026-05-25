"""SamAgent: offline ARC-AGI-3 agent with graph BFS + internal planning."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from arc_agi import Arcade, OperationMode
from arc_agi.wrapper import EnvironmentWrapper
from arcengine import FrameDataRaw, GameAction, GameState

from sam.audit.logger import AuditLogger
from sam.audit.rhae import compute_rhae_report
from sam.config import DEFAULT_PLANNING_MS, GameMetadata, find_metadata, load_yaml_config
from sam.core.sam_core import SamCore
from sam.planner.internal_planner import InternalPlanner, PlannerPhase
from sam.planner.state_graph import StateGraph
from sam.rewards.engine import RewardVectorEngine
from sam.rewards.meta_head import MetaRewardHead
from sam.utils import frame_hash, frame_to_model_input, raw_to_frame_data

logger = logging.getLogger(__name__)


@dataclass
class SessionStats:
    env_steps: int = 0
    internal_steps: int = 0
    level_actions: list[int] = field(default_factory=list)
    current_level_steps: int = 0
    resets: int = 0
    modules_grown: int = 0


class SamAgent:
    """Small Action Model agent for offline ARC-AGI-3 play."""

    def __init__(
        self,
        game_id: str = "ls20-9607627b",
        environments_dir: str | Path = "environment_files",
        config_path: str | Path | None = None,
        run_dir: str | Path = "sam/runs",
        checkpoint: str | Path | None = None,
        device: str | None = None,
        planning_ms: float = DEFAULT_PLANNING_MS,
    ) -> None:
        self.game_id = game_id
        self.environments_dir = Path(environments_dir)
        cfg_path = config_path or Path("sam/configs/ls20.yaml")
        self.config = load_yaml_config(Path(cfg_path))
        self.planning_ms = float(self.config.get("planning_ms", planning_ms))

        self.metadata = find_metadata(game_id, self.environments_dir)
        if self.metadata is None:
            raise FileNotFoundError(f"No metadata for {game_id} under {self.environments_dir}")

        self.device = torch.device(device or "cpu")
        self.core = SamCore(latent_dim=int(self.config.get("latent_dim", 128))).to(self.device)
        self.graph = StateGraph(max_depth=int(self.config.get("max_depth", 15)))
        self.planner = InternalPlanner(
            self.core, self.graph, planning_ms=self.planning_ms
        )
        self.rewards = RewardVectorEngine()
        self.meta = MetaRewardHead().to(self.device)
        self.stats = SessionStats()
        self.run_dir = Path(run_dir)
        self.audit = AuditLogger(self.run_dir, game_id)
        self.replay_buffer: list[dict[str, Any]] = []
        self._wm_errors: list[float] = []
        self._prev_frame: FrameDataRaw | None = None
        self._prev_z: torch.Tensor | None = None
        self._aux_trace: list[float] = []
        self._competence_trace: list[float] = []

        if checkpoint:
            self.load_checkpoint(checkpoint)

        self.arcade = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(self.environments_dir),
        )

    def load_checkpoint(self, path: str | Path) -> None:
        from sam.train.checkpoint import load_sam_checkpoint

        load_sam_checkpoint(self.core, path, device=self.device)

    def save_checkpoint(self, path: str | Path) -> None:
        from sam.train.checkpoint import save_sam_checkpoint

        save_sam_checkpoint(self.core, path)

    def _max_actions_current_level(self, frame: FrameDataRaw) -> int:
        return self.metadata.max_actions_for_level(frame.levels_completed)

    def _micro_update(
        self,
        z: torch.Tensor,
        z_next: torch.Tensor,
        action_id: int,
        reward_scalar: float,
        *,
        modules_only: bool = False,
    ) -> None:
        lr = 1e-4
        self.core.train()
        z = z.detach().requires_grad_(False)
        z_next = z_next.detach()

        pred, conf = self.core.world_model(z, action_id)
        pred = self.core.skills.apply(pred)
        if z_next.dim() == 1:
            z_next = z_next.unsqueeze(0)
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        wm_loss = F.mse_loss(pred, z_next)
        self._wm_errors.append(float(wm_loss.item()))

        _, value_vec = self.core.value(z)
        target = torch.tensor(reward_scalar, device=self.device, dtype=torch.float32)

        params: list[torch.nn.Parameter] = []
        if not modules_only:
            params.extend(self.core.world_model.parameters())
            params.extend(self.core.value.parameters())
        if self.core.skills.modules:
            for m in self.core.skills.modules:
                params.extend(m.parameters())

        if not params:
            self.core.eval()
            return

        opt = torch.optim.Adam(params, lr=lr)
        opt.zero_grad()
        (wm_loss + F.mse_loss(value_vec.mean(), target)).backward()
        opt.step()
        self.core.eval()

    def _record_transition(
        self,
        before: FrameDataRaw,
        action_id: int,
        after: FrameDataRaw,
        *,
        phase: PlannerPhase,
        internal: bool,
    ) -> None:
        before_fd = raw_to_frame_data(before)
        after_fd = raw_to_frame_data(after)
        self.graph.add_transition(before_fd, action_id, after_fd)

        h_before = frame_hash(before)
        h_after = frame_hash(after)
        z = self.core.encode(frame_to_model_input(before))
        z_next = self.core.encode(frame_to_model_input(after))
        _, conf = self.core.world_model(z, action_id)
        _, val = self.core.value(z)
        _, val_next = self.core.value(z_next)

        grid_changed = 0
        if before.frame and after.frame:
            from sam.planner.novelty import grid_diff_mask

            g0 = before.frame[0].tolist() if hasattr(before.frame[0], "tolist") else before.frame[0]
            g1 = after.frame[0].tolist() if hasattr(after.frame[0], "tolist") else after.frame[0]
            grid_changed = grid_diff_mask(g0, g1)

        rv = self.rewards.compute(
            state_hash=h_after,
            prev_hash=h_before,
            wm_confidence=float(conf.item()),
            value_delta=float(val_next.mean().item() - val.mean().item()),
            level_before=before.levels_completed,
            level_after=after.levels_completed,
            is_env_step=not internal,
            is_internal=internal,
            game_over=after.state == GameState.GAME_OVER,
            reset_after_over=False,
            grid_changed=grid_changed,
            action_id=action_id,
        )
        aux = self.meta.propose(z_next)
        rv.aux = aux
        scalar = self.rewards.scalar(rv, aux)
        self._aux_trace.append(sum(aux.values()))
        self._competence_trace.append(float(rv.values[5]))

        self.audit.log(
            self.audit.make_record(
                level=before.levels_completed,
                action_id=action_id,
                state_hash=h_after,
                reward_vector=rv.as_dict(),
                reward_scalar=scalar,
                phase=phase.name,
                module_count=len(self.core.skills.modules),
                levels_completed=after.levels_completed,
                state=after.state.name if hasattr(after.state, "name") else str(after.state),
                internal=internal,
            )
        )

        if not internal:
            self.replay_buffer.append(
                {
                    "z": z.cpu(),
                    "z_next": z_next.cpu(),
                    "action": action_id,
                    "reward": scalar,
                }
            )
            self._micro_update(z, z_next, action_id, scalar)

        growth = self.core.skills.on_novel_state(False)
        if growth:
            self.stats.modules_grown += 1
        plateau = self.core.skills.on_wm_plateau(self._wm_errors)
        if plateau:
            self.stats.modules_grown += 1

        self._prev_frame = after
        self._prev_z = z_next

    def _on_level_complete(self, frame: FrameDataRaw) -> None:
        self.stats.level_actions.append(self.stats.current_level_steps)
        self.stats.current_level_steps = 0
        evt = self.core.skills.on_level_success()
        if evt and len(self.core.skills.modules) < 8:
            self.stats.modules_grown += 1
        self._meso_update()
        self.meta.record_episode(self._aux_trace, self._competence_trace)
        self._aux_trace = []
        self._competence_trace = []
        self.planner.reset_level(frame)
        self.rewards.reset_level()

    def _meso_update(self) -> None:
        if len(self.replay_buffer) < 10:
            return
        batch = self.replay_buffer[-100:]
        for item in batch:
            z = item["z"].to(self.device)
            z_next = item["z_next"].to(self.device)
            self._micro_update(z, z_next, int(item["action"]), float(item["reward"]))

    def _mega_update(self) -> None:
        self._meso_update()

    def _action_from_id(self, action_id: int) -> GameAction:
        action = GameAction.from_id(action_id)
        if action.is_simple():
            action.action_data.game_id = self.game_id
        elif action.is_complex():
            action.set_data({"game_id": self.game_id, "x": 32, "y": 32})
        return action

    def play_episode(self, env: EnvironmentWrapper | None = None) -> dict[str, Any]:
        if env is None:
            env = self.arcade.make(self.game_id, save_recording=False)
        if env is None:
            raise RuntimeError(f"Failed to create environment for {self.game_id}")

        frame = env.reset()
        if frame is None:
            raise RuntimeError("Reset failed")

        self.stats = SessionStats()
        self.planner.reset_level(frame)
        self.rewards.reset_level()
        self.planner.brancher.reset_level(frame_hash(frame))
        self._prev_frame = frame
        self._prev_z = self.core.encode(frame_to_model_input(frame))

        win = False
        while frame.state not in (GameState.WIN, GameState.GAME_OVER):
            max_actions = self._max_actions_current_level(frame)
            if self.stats.current_level_steps >= max_actions:
                logger.warning("Level budget exhausted at level %s", frame.levels_completed)
                break

            if self.planner.should_reset(frame, self.stats.current_level_steps, max_actions):
                reset_id = InternalPlanner.reset_action_id()
                after = env.step(self._action_from_id(reset_id))
                if after is None:
                    break
                self.stats.resets += 1
                self.stats.current_level_steps += 1
                self.stats.env_steps += 1
                self._record_transition(frame, reset_id, after, phase=PlannerPhase.LIVE_RESET, internal=False)
                frame = after
                self.planner.reset_level(frame)
                self.planner.brancher.reset_level(frame_hash(frame))
                continue

            action_id, phase, pinfo = self.planner.choose_action(
                frame,
                level_actions_used=self.stats.current_level_steps,
                max_actions=max_actions,
                prev_frame=self._prev_frame,
            )

            action = self._action_from_id(action_id)
            after = env.step(action)
            if after is None:
                break

            self.stats.env_steps += 1
            self.stats.current_level_steps += 1
            self._record_transition(frame, action_id, after, phase=phase, internal=False)
            self.planner.brancher.mark_tried(frame_hash(frame), action_id)

            if after.levels_completed > frame.levels_completed:
                self._on_level_complete(after)

            frame = after

        if frame.state == GameState.WIN:
            win = True
            if self.stats.current_level_steps > 0:
                self.stats.level_actions.append(self.stats.current_level_steps)
            self.core.skills.on_game_win()
            self._mega_update()

        report = compute_rhae_report(self.metadata, self.stats.level_actions)
        summary = {
            "game_id": self.game_id,
            "win": win,
            "state": frame.state.name if hasattr(frame.state, "name") else str(frame.state),
            "levels_completed": frame.levels_completed,
            "env_steps": self.stats.env_steps,
            "resets": self.stats.resets,
            "modules": len(self.core.skills.modules),
            "total_params": self.core.total_params(),
            "rhae": report,
            "audit_path": str(self.audit.path),
        }
        summary_path = self.run_dir / f"{self.game_id.replace('/', '_')}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        return summary

    def run(self) -> dict[str, Any]:
        env = self.arcade.make(self.game_id, save_recording=False)
        return self.play_episode(env)
