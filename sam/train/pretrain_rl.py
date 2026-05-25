#!/usr/bin/env python3
"""RL fine-tune with reward vector on recorded transitions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sam.train.pretrain_ssl import collect_transitions
from sam.rewards.vector import RewardVectorEngine
from sam.sam_model import SAMCore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environments-dir", default="environment_files")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--checkpoint", default="sam/checkpoints/global_sam.pt")
    args = parser.parse_args()
    core = SAMCore.create()
    ckpt = ROOT / args.checkpoint
    core.load_checkpoint(ckpt)
    transitions = collect_transitions(ROOT / args.environments_dir, max_games=10)
    engine = RewardVectorEngine()
    for epoch in range(args.epochs):
        for f0, aid, f1 in transitions:
            from arcengine import FrameData

            before = FrameData(frame=f0, available_actions=[aid])
            after = FrameData(frame=f1, available_actions=[aid])
            rv = engine.step_env(before, after, aid)
            import torch

            z = core.encoder(f0)
            z2 = core.encoder(f1)
            core.micro_update(z, z2, aid, torch.tensor(rv.as_array()[:7]))
        print(f"rl epoch {epoch + 1} done")
    core.save_checkpoint(ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
