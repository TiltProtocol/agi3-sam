"""Shared helpers for frame conversion and hashing."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from arcengine import FrameData, FrameDataRaw


def grid_from_frame(frame: FrameDataRaw | FrameData) -> list[list[int]]:
    if not frame.frame:
        return [[0] * 64 for _ in range(64)]
    layer = frame.frame[0]
    if hasattr(layer, "tolist"):
        return layer.tolist()
    return layer


def frame_to_model_input(frame: FrameDataRaw | FrameData) -> list[list[list[int]]]:
    return [grid_from_frame(frame)]


def raw_to_frame_data(frame: FrameDataRaw) -> FrameData:
    return FrameData(
        game_id=frame.game_id,
        frame=frame_to_model_input(frame),
        state=frame.state,
        levels_completed=frame.levels_completed,
        win_levels=frame.win_levels,
        action_input=frame.action_input,
        guid=frame.guid,
        full_reset=getattr(frame, "full_reset", False),
        available_actions=list(frame.available_actions),
    )


def frame_hash(frame: FrameDataRaw | FrameData) -> str:
    payload = {
        "levels": frame.levels_completed,
        "actions": sorted(frame.available_actions),
        "grid": grid_from_frame(frame),
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def count_params(module: Any) -> int:
    import torch.nn as nn

    if isinstance(module, nn.Module):
        return sum(p.numel() for p in module.parameters())
    return 0
