#!/usr/bin/env python3
"""
Kaggle offline submission helper for SAM.

Install wheels from data/kaggle/arc_agi_3_wheels/ (when present) and run
inference-only SAM with no external API calls.

Usage (Kaggle notebook cell):
    !pip install -q /kaggle/input/arc-agi-3-wheels/*.whl torch pyyaml
    !python kaggle/run_offline.py --game ls20-9607627b
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def install_wheels(wheel_dir: Path) -> None:
    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
        print(f"No wheels in {wheel_dir}; assuming arc-agi already installed")
        return
    for whl in wheels:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", str(whl)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle offline SAM runner")
    parser.add_argument("--game", default="ls20-9607627b")
    parser.add_argument(
        "--wheel-dir",
        default=os.environ.get("ARC_WHEEL_DIR", "data/kaggle/arc_agi_3_wheels"),
    )
    parser.add_argument(
        "--env-dir",
        default=os.environ.get("ARC_ENV_DIR", "environment_files"),
    )
    parser.add_argument(
        "--checkpoint",
        default="sam/checkpoints/global_sam.pt",
    )
    parser.add_argument("--run-dir", default="/kaggle/working/sam_runs")
    args = parser.parse_args()

    os.environ["OPERATION_MODE"] = "offline"

    wheel_dir = Path(args.wheel_dir)
    if wheel_dir.exists():
        install_wheels(wheel_dir)

    from sam.agent import SamAgent

    agent = SamAgent(
        game_id=args.game,
        environments_dir=args.env_dir,
        checkpoint=args.checkpoint if Path(args.checkpoint).exists() else None,
        run_dir=args.run_dir,
        planning_ms=30.0,
    )
    result = agent.run()
    out = Path(args.run_dir) / "kaggle_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
