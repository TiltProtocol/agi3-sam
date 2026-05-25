# SAM — Small Action Model for ARC-AGI-3

Human-inspired offline agent architecture for the ARC-AGI-3 RHAE benchmark.

## Philosophy

SAM mimics how a human solves ARC puzzles:

1. **Try what you know** — replay successful plans from memory (StateGraph BFS).
2. **Notice what's new** — curiosity-driven investigation when states are novel.
3. **Think before acting** — internal world-model rollouts (free internal steps).
4. **Reset and retry** — level RESET keeps learned weights; only env steps count toward RHAE.

Decisions trace to [`answers.md`](../answers.md).

## Architecture

| Module | Human analogy | Role |
|--------|---------------|------|
| GridEncoder | Visual perception | 64×64 color grid → latent `z` |
| GameAdapter | Game-specific intuition | LoRA on encoder/WM (~50–100K params/game) |
| StateGraph | Episodic memory | Hash states; BFS on real transitions |
| WorldModel | Mental simulation | Predict `z'` from `(z, action)` |
| InternalPlanner | Working memory loop | AttemptSolve → Investigate → InternalReplay → LiveReset |
| ActionHead | Motor output | Masked policy over `available_actions` |
| RewardVectorEngine | Intrinsic motivation | 7-dim vector every step |
| SkillModuleBank | Skill learning | Literal growth (max 8 modules/game) |
| MetaRewardHead | Parent→child beliefs | Gated auxiliary reward dims |

## Internal loop (per level)

```
AttemptSolve (graph BFS + WM beam, 0 env steps)
  → success: ExecuteLive
  → fail: DetectNovelty
    → Investigate (minimal env steps, curiosity bias)
      → InternalReplay (simulate from graph root)
        → success: ExecuteLive
        → fail: LiveReset (keep weights)
```

Planning budget: **50ms CPU** per env action (configurable).

## Reward vector (7 fixed dims)

| Dim | Signal | Anti-hack |
|-----|--------|-----------|
| curiosity | +1 first hash visit | Per-hash cap |
| efficiency | −1 env / −0.05 internal | Weight floor 0.25 |
| intuition | Δ world-model confidence | Clipped |
| proximity | Δ value + level events | No gain on loops |
| savvy | Outcome entropy split | Penalize no-ops |
| competence | +5 level complete | Env-only |
| recovery | +0.5 after GAME_OVER reset | Small |

**MetaRewardHead** proposes up to 2 aux dims; committed only if Pearson r > 0.3 with competence over ≥20 episodes.

## Learning schedule

| Event | Update |
|-------|--------|
| Real transition | Micro-step WM + ValueHead + modules (lr 1e-4) |
| Internal transition | Replay buffer only (efficiency ×0.05) |
| Novelty understood | Grow module + 10 module-only steps |
| Level success | Grow module + meso-update (100 replay steps) |
| Game WIN | Mega-update + checkpoint export |

## RHAE & budgets

- Per-level max actions = **5 × baseline_actions[level]** from `metadata.json`
- Score = `min((baseline/actions)² × 100, 115)` per level
- ls20 baselines: `[22, 123, 73, 84, 96, 192, 186]`

## Param budget

Global core target ~1.3M params; growth cap **≤3M total** for Kaggle CPU inference.

## Milestones

| ID | Criterion |
|----|-----------|
| M0 | Offline run + JSONL audit |
| M1 | ls20 level 1 complete |
| M2 | ls20 levels 1–3 |
| M3 | ls20 full WIN |
| M4 | ≥1 level on 3 transfer games |
| M5 | Kaggle offline submit |

## Open tradeoffs (ask human)

When investigation must choose between **novelty** vs **efficiency** on level 1 with tight budget — default: explore unvisited graph edges first, then curiosity policy.

## Files

- `agent.py` — SamAgent orchestrator
- `scripts/run_sam.py` — offline runner
- `train/pretrain.py` — 25-env self-supervised pretrain
- `configs/ls20.yaml` — prototype config
