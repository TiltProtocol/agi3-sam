#!/usr/bin/env python3
"""Offline BFS to build ls20 level plans (run on cloud / overnight)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam.planner.ls20_solver import (
    DEFAULT_CACHE,
    bfs_level_plan,
    load_plans,
    save_plans,
    solve_all_levels,
    LEVEL1_PREFIX,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("solve_ls20")


def main() -> None:
    parser = argparse.ArgumentParser(description="BFS solver for ls20 plans")
    parser.add_argument("--env-dir", default="environment_files")
    parser.add_argument("--output", default=str(DEFAULT_CACHE))
    parser.add_argument("--level", type=int, default=-1, help="Single level index (0-6)")
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--all", action="store_true", help="Solve all levels sequentially")
    args = parser.parse_args()

    out = Path(args.output)
    plans = load_plans(out)

    if args.all:
        plans = solve_all_levels(
            environments_dir=args.env_dir,
            per_level_time_s=args.time_limit,
            per_level_max_steps=args.max_steps,
        )
        save_plans(plans, out)
        logger.info("Saved %s levels to %s", len(plans), out)
        return

    if args.level < 0:
        parser.error("Use --level N or --all")
        return

    prefix: list[int] = []
    for i in range(args.level):
        key = str(i)
        if key not in plans:
            logger.error("Missing plan for level %s; run lower levels first", i)
            sys.exit(1)
        prefix.extend(plans[key])

    if args.level == 0 and not prefix:
        suffix = LEVEL1_PREFIX
    else:
        suffix = bfs_level_plan(
            prefix,
            environments_dir=args.env_dir,
            max_new_steps=args.max_steps,
            time_limit_s=args.time_limit,
        )

    if suffix is None:
        logger.error("No solution for level %s", args.level)
        sys.exit(1)

    plans[str(args.level)] = suffix
    save_plans(plans, out)
    logger.info("Level %s: %s steps -> %s", args.level, len(suffix), out)


if __name__ == "__main__":
    main()
