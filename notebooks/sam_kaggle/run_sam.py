#!/usr/bin/env python3
"""Offline SAM runner for ARC-AGI-3."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam.agent import SamAgent
from sam.config import find_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("run_sam")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM offline on ARC-AGI-3")
    parser.add_argument("--game", default="ls20-9607627b", help="Game id")
    parser.add_argument(
        "--env-dir",
        default="environment_files",
        help="Local environments directory",
    )
    parser.add_argument(
        "--config",
        default="sam/configs/ls20.yaml",
        help="YAML config path",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional global_sam.pt checkpoint",
    )
    parser.add_argument(
        "--run-dir",
        default="sam/runs",
        help="Audit output directory",
    )
    parser.add_argument(
        "--planning-ms",
        type=float,
        default=None,
        help="Planning time budget per action (ms)",
    )
    args = parser.parse_args()

    meta = find_metadata(args.game, Path(args.env_dir))
    if meta is None:
        logger.error("Metadata not found for %s", args.game)
        sys.exit(1)

    logger.info("Starting SAM on %s (%s)", meta.title or args.game, meta.game_id)
    logger.info("Baselines: %s", meta.baseline_actions)

    agent = SamAgent(
        game_id=args.game,
        environments_dir=args.env_dir,
        config_path=args.config,
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        planning_ms=args.planning_ms or 50.0,
    )

    logger.info("Core params: %s (budget %s)", agent.core.total_params(), 3_000_000)
    result = agent.run()

    print(json.dumps(result, indent=2))
    rhae = result.get("rhae", {})
    logger.info(
        "Finished: win=%s levels=%s env_steps=%s RHAE=%s",
        result.get("win"),
        result.get("levels_completed"),
        result.get("env_steps"),
        rhae.get("total_score"),
    )
    logger.info("Audit log: %s", result.get("audit_path"))


if __name__ == "__main__":
    main()
