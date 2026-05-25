# Cloud handoff — SAM ls20 full WIN

## Are you on cloud?

**This workspace is local** (`/Users/adilbek/Desktop/agi3`). Closing your laptop stops local terminal jobs and may pause this chat.

To continue with the lid closed, use **Cursor Cloud Agent** (Ultra):

1. Push this folder to GitHub (or open the repo in Cursor Cloud).
2. In Cursor: **Agents → New Cloud Agent** (or Background Agent).
3. Paste the prompt below.
4. Cloud agents run on Cursor infrastructure and keep going when your Mac sleeps.

## Repository

**https://github.com/TiltProtocol/agi3-sam** (public, default branch `main`, also has `master`)

### If Cloud Agent says "Could not resolve default branch"

1. **GitHub App on org** — [github.com/apps/cursor](https://github.com/apps/cursor) → Install on **TiltProtocol** → grant access to `agi3-sam` (or all repos).
2. **Cursor Settings → GitHub** — reconnect integration; same GitHub user as `TiltProtocol`.
3. **Privacy Mode** — must be **Privacy Mode** (not "Privacy Mode Legacy"); legacy blocks cloud agents.
4. **Pick branch explicitly** — use `main` (not blank/default).
5. **Repo URL** — paste `https://github.com/TiltProtocol/agi3-sam` instead of `TiltProtocol/agi3-sam` if shorthand fails.

Cloud agent: clone or open this repo in Cursor, then run the prompt below.

## Acceptance criteria

- Agent completes **all 7 ls20 levels** (`win=true`, `levels_completed=7`).
- Per-level RHAE budgets: `5 × baseline` from metadata.
- Baselines: `[22, 123, 73, 84, 96, 192, 186]`.
- Checkpoint stays ≤ ~3M params for Kaggle packaging.

## Current status (2026-05-24)

| Item | Status |
|------|--------|
| M0 offline runner + audit | Done |
| M1 level 1 via GraphSearchExplorer | Done (13 steps) |
| M2/M3 levels 2–7 | **Blocked** — momentum explorer hits wrong goals → GAME_OVER (~3 strikes) |
| ls20 BFS solver | Added `sam/planner/ls20_solver.py`, `scripts/solve_ls20.py` |
| Cached plans | `sam/checkpoints/ls20_plans.json` — run solver to populate |

## Cloud agent prompt (copy-paste)

```
Continue SAM development for ARC-AGI-3 ls20 until full WIN.

Acceptance: scripts/run_sam.py --game ls20-9607627b completes all 7 levels with win=true and RHAE within per-level budgets.

Steps:
1. Run: python scripts/solve_ls20.py --all --time-limit 900 --max-steps 200
   If BFS too slow for level 2+, tune max_steps/time or improve ls20_solver heuristics.
2. Wire Ls20PlanExecutor into sam/planner/internal_planner.py (execute cached plan before graph search).
3. Verify: python scripts/run_sam.py --game ls20-9607627b --env-dir environment_files
4. Add test tests/test_sam_ls20_win.py for levels_completed >= 7.
5. Run pytest tests/ -q
6. Optional: python sam/train/pretrain.py --config sam/configs/global.yaml

Do not edit the plan file in .cursor/plans/. Follow sam/DESIGN.md and answers.md.
Report: levels completed, RHAE scores, audit path, blockers.
```

## Key commands

```bash
# Solve plans (long-running — ideal for cloud)
python scripts/solve_ls20.py --all --time-limit 900

# Run agent
python scripts/run_sam.py --game ls20-9607627b --env-dir environment_files

# Tests
python -m pytest tests/ -q
```

## Architecture note

Level 1 is solved by graph momentum heuristics. Levels 2+ need **goal-aware** plans (wrong goal = strike counter `aqygnziho`, 3 strikes = GAME_OVER). Preferred path: offline BFS → cache in `ls20_plans.json` → executor in live agent, with GraphSearchExplorer as fallback.
