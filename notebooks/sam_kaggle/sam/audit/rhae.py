"""RHAE score estimation vs metadata baselines."""

from __future__ import annotations

from dataclasses import dataclass

from sam.config import GameMetadata


@dataclass
class LevelScore:
    level_index: int
    actions: int
    baseline: int
    score: float
    max_actions: int


def rhae_level_score(actions: int, baseline: int) -> float:
    if actions <= 0 or baseline <= 0:
        return 0.0
    raw = (baseline / actions) ** 2 * 100.0
    return min(raw, 115.0)


def compute_rhae_report(
    metadata: GameMetadata,
    level_actions: list[int],
) -> dict:
    scores: list[LevelScore] = []
    total = 0.0
    for i, actions in enumerate(level_actions):
        baseline = metadata.baseline_for_level(i)
        max_a = metadata.max_actions_for_level(i)
        sc = rhae_level_score(actions, baseline)
        scores.append(
            LevelScore(
                level_index=i,
                actions=actions,
                baseline=baseline,
                score=sc,
                max_actions=max_a,
            )
        )
        total += sc
    n_levels = len(metadata.baseline_actions) or max(len(level_actions), 1)
    return {
        "game_id": metadata.game_id,
        "level_scores": [
            {
                "level": s.level_index,
                "actions": s.actions,
                "baseline": s.baseline,
                "score": round(s.score, 2),
                "max_actions": s.max_actions,
                "within_budget": s.actions <= s.max_actions,
            }
            for s in scores
        ],
        "total_score": round(total, 2),
        "max_possible": round(115.0 * n_levels, 2),
        "levels_completed": len(level_actions),
    }
