#!/usr/bin/env python3
"""SAM pretrain: self-supervised next-latent + inverse dynamics on public envs."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_agi import Arcade, OperationMode
from arcengine import GameAction

from sam.config import load_yaml_config
from sam.core.sam_core import SamCore
from sam.train.checkpoint import save_sam_checkpoint
from sam.utils import frame_to_model_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sam.pretrain")


def collect_transitions(
    game_id: str,
    env_dir: Path,
    max_steps: int = 200,
    seed: int = 0,
) -> list[tuple[list, int, list]]:
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(env_dir))
    env = arc.make(game_id, seed=seed)
    if env is None:
        return []
    frame = env.reset()
    if frame is None:
        return []
    out: list[tuple[list, int, list]] = []
    for _ in range(max_steps):
        avail = [a for a in frame.available_actions if a != 6] or frame.available_actions
        aid = random.choice(avail)
        action = GameAction.from_id(aid)
        if action.is_complex():
            action.set_data({"game_id": game_id, "x": 32, "y": 32})
        nxt = env.step(action)
        if nxt is None:
            break
        out.append((frame_to_model_input(frame), aid, frame_to_model_input(nxt)))
        frame = nxt
    return out


def pretrain(
    config_path: Path,
    output: Path,
    *,
    max_steps_per_game: int = 150,
    device: str = "cpu",
) -> dict:
    cfg = load_yaml_config(config_path)
    env_dir = Path(cfg.get("environments_dir", "environment_files"))
    games = cfg.get("games", ["ls20-9607627b"])
    latent_dim = int(cfg.get("latent_dim", 128))
    lr = float(cfg.get("learning_rate", 1e-4))
    epochs_ss = int(cfg.get("epochs_self_supervised", 2))

    core = SamCore(latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(core.parameters(), lr=lr)

    all_transitions: list[tuple[list, int, list]] = []
    for gid in games:
        tr = collect_transitions(gid, env_dir, max_steps=max_steps_per_game)
        logger.info("Collected %s transitions from %s", len(tr), gid)
        all_transitions.extend(tr)

    if not all_transitions:
        logger.warning("No transitions collected; saving untrained checkpoint")
        save_sam_checkpoint(core, output)
        return {"transitions": 0, "loss": None}

    core.train()
    last_loss = 0.0
    for epoch in range(epochs_ss):
        random.shuffle(all_transitions)
        for before, aid, after in all_transitions:
            z = core.encode(before)
            z_next = core.encode(after)
            if z.dim() == 1:
                z = z.unsqueeze(0)
            if z_next.dim() == 1:
                z_next = z_next.unsqueeze(0)
            pred, _ = core.world_model(z.squeeze(0), aid)
            loss = F.mse_loss(pred.unsqueeze(0), z_next.detach())
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
        logger.info("Epoch %s SS loss=%.4f", epoch + 1, last_loss)

    core.eval()
    save_sam_checkpoint(core, output)
    report = {
        "transitions": len(all_transitions),
        "loss": last_loss,
        "params": core.total_params(),
        "checkpoint": str(output),
        "games": len(games),
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="sam/configs/global.yaml")
    parser.add_argument("--output", default="sam/checkpoints/global_sam.pt")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = pretrain(
        Path(args.config),
        Path(args.output),
        max_steps_per_game=args.max_steps,
        device=args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
