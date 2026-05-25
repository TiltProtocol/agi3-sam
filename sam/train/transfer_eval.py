#!/usr/bin/env python3
"""Transfer evaluation: load global checkpoint, run on held-out games."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam.agent import SamAgent
from sam.config import load_yaml_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sam.transfer_eval")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="sam/configs/global.yaml")
    parser.add_argument("--checkpoint", default="sam/checkpoints/global_sam.pt")
    parser.add_argument("--run-dir", default="sam/runs/transfer")
    args = parser.parse_args()

    cfg = load_yaml_config(Path(args.config))
    games = cfg.get("transfer_eval_games", ["ft09-0d8bbf25"])
    env_dir = cfg.get("environments_dir", "environment_files")

    results = []
    levels_won = 0
    for gid in games:
        logger.info("Evaluating %s", gid)
        agent = SamAgent(
            game_id=gid,
            environments_dir=env_dir,
            checkpoint=args.checkpoint,
            run_dir=args.run_dir,
            planning_ms=50.0,
        )
        summary = agent.run()
        results.append(summary)
        if summary.get("levels_completed", 0) >= 1:
            levels_won += 1

    report = {
        "games_evaluated": len(games),
        "games_with_level": levels_won,
        "m4_pass": levels_won >= 1,
        "results": results,
    }
    out = Path(args.run_dir) / "transfer_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
