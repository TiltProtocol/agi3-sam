#!/usr/bin/env python3
"""Minimal smoke train: SSL + RL one epoch, export global_sam.pt."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    pretrain = ROOT / "sam/train/pretrain.py"
    subprocess.check_call(
        [
            sys.executable,
            str(pretrain),
            "--config",
            "sam/configs/global.yaml",
            "--output",
            "sam/checkpoints/global_sam.pt",
            "--max-steps",
            "40",
        ],
        cwd=str(ROOT),
    )
    out = ROOT / "sam/checkpoints/global_sam.pt"
    print(f"smoke train complete: {out.exists()}")
    return 0 if out.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
