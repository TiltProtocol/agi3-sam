# Cloud handoff — SAM ls20 full WIN

## Are you on cloud?

**This workspace is local** (`/Users/adilbek/Desktop/agi3`). Closing your laptop stops local terminal jobs and may pause this chat.

To continue with the lid closed, use **Cursor Cloud Agent** (Ultra):

1. Push this folder to GitHub (or open the repo in Cursor Cloud).
2. In Cursor: **Agents → New Cloud Agent** (or Background Agent).
3. Paste the prompt below.
4. Cloud agents run on Cursor infrastructure and keep going when your Mac sleeps.

## Repository

## Repository mirrors

Use the repo that matches **the GitHub account connected in Cursor Settings → GitHub**:

| Account | URL | Branch |
|---------|-----|--------|
| TiltProtocol | https://github.com/TiltProtocol/agi3-sam | `main` |
| rontoTech | https://github.com/rontoTech/agi3-sam | `main` |

Both are public and identical. If one fails branch resolution, try the other.

### If Cloud Agent still says "Could not resolve default branch"

This is a **known Cursor backend bug** (GitHub App token). Repo-side settings are correct.

**Fix A — match GitHub account**

1. Cursor **Settings → GitHub** → note which account is connected (TiltProtocol vs rontoTech).
2. Cloud Agent → use the **matching** repo URL above.
3. Set branch explicitly to **`main`** (do not leave blank).

**Fix B — Privacy Mode**

Settings → Privacy → must be **Privacy Mode**, not **Privacy Mode (Legacy)**.

**Fix C — GitHub Actions (works with laptop closed, no Cursor Cloud Agent)**

1. Open https://github.com/TiltProtocol/agi3-sam/actions (or rontoTech mirror).
2. Run workflow **SAM ls20 solver** → **Run workflow**.
3. Download artifact `sam-ls20-results` (`ls20_plans.json`, run summary).

**Fix D — Cursor support**

Email support with Request ID from the failed run URL on [cursor.com/agents](https://cursor.com/agents).

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
