"""JSONL audit logger for SAM runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class AuditRecord:
    timestamp: str
    game_id: str
    level: int
    env_step: int
    action_id: int
    state_hash: str
    reward_vector: dict[str, float]
    reward_scalar: float
    phase: str
    module_count: int
    levels_completed: int
    state: str
    internal: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class AuditLogger:
    def __init__(self, run_dir: Path, game_id: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.game_id = game_id
        self.path = self.run_dir / f"{game_id.replace('/', '_')}.jsonl"
        self.records: list[AuditRecord] = []
        self.env_steps = 0

    def log(self, record: AuditRecord) -> None:
        self.records.append(record)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")

    def make_record(
        self,
        *,
        level: int,
        action_id: int,
        state_hash: str,
        reward_vector: dict[str, float],
        reward_scalar: float,
        phase: str,
        module_count: int,
        levels_completed: int,
        state: str,
        internal: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> AuditRecord:
        if not internal:
            self.env_steps += 1
        return AuditRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            game_id=self.game_id,
            level=level,
            env_step=self.env_steps,
            action_id=action_id,
            state_hash=state_hash,
            reward_vector=reward_vector,
            reward_scalar=reward_scalar,
            phase=phase,
            module_count=module_count,
            levels_completed=levels_completed,
            state=state,
            internal=internal,
            extra=extra or {},
        )
