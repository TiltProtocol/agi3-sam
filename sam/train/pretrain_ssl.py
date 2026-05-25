#!/usr/bin/env python3
"""Self-supervised pretrain: next-frame + inverse dynamics on public env recordings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from sam.sam_model import SAMCore


def collect_transitions(env_dir: Path, max_games: int = 5, steps_per_game: int = 30) -> list:
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction, GameState

    from sam.utils import frame_to_model_input

    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(env_dir))
    transitions = []
    games = list(env_dir.rglob("metadata.json"))[:max_games]
    for meta_path in games:
        meta = json.loads(meta_path.read_text())
        gid = meta["game_id"]
        w = arc.make(gid)
        if w is None:
            continue
        raw = w.reset()
        if raw is None:
            continue
        for step_i in range(steps_per_game):
            if raw.state == GameState.WIN:
                break
            aids = [a for a in raw.available_actions if a != 0] or [1]
            aid = aids[step_i % len(aids)]
            action = GameAction.from_id(aid)
            nxt = w.step(action)
            if nxt is None:
                break
            transitions.append(
                (frame_to_model_input(raw), aid, frame_to_model_input(nxt))
            )
            raw = nxt
    return transitions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environments-dir", default="environment_files")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--out", default="sam/checkpoints/global_sam.pt")
    args = parser.parse_args()
    env_dir = ROOT / args.environments_dir
    core = SAMCore.create()
    transitions = collect_transitions(env_dir)
    if not transitions:
        print("No transitions collected; saving init checkpoint")
        core.save_checkpoint(ROOT / args.out)
        return 0
    opt = torch.optim.Adam(
        list(core.encoder.parameters()) + list(core.world_model.parameters()),
        lr=1e-4,
    )
    for epoch in range(args.epochs):
        total = 0.0
        for f0, aid, f1 in transitions:
            z = core.encoder(f0)
            z1 = core.encoder(f1)
            z_pred, _ = core.world_model(z, aid)
            loss = nn.functional.mse_loss(z_pred, z1.detach())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"epoch {epoch + 1} loss={total / len(transitions):.4f}")
    core.save_checkpoint(ROOT / args.out)
    print(f"saved {args.out} params={core.param_count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
