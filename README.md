# agi3 — SAM for ARC-AGI-3

Small Action Model (SAM): human-inspired offline agent for the [ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2025) RHAE benchmark.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run on ls20 (offline)
python scripts/run_sam.py --game ls20-9607627b --env-dir environment_files

# Tests
python -m pytest tests/ -q
```

## Solver (levels 2+)

```bash
python scripts/solve_ls20.py --all --time-limit 900
```

Plans cache to `sam/checkpoints/ls20_plans.json` and are executed by the internal planner.

## Docs

- [`sam/DESIGN.md`](sam/DESIGN.md) — architecture and milestones
- [`CLOUD_HANDOFF.md`](CLOUD_HANDOFF.md) — cloud agent continuation prompt
- [`answers.md`](answers.md) — design decisions

## Status

- M1: ls20 level 1 complete (13 actions, RHAE 115)
- M3: full ls20 WIN — in progress (BFS plan cache + training)
