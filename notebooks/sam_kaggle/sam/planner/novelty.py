"""Cheap novelty detection on grid diffs."""

from __future__ import annotations

from arcengine import FrameData


def grid_diff_mask(a: list[list[int]], b: list[list[int]]) -> int:
    """Count changed cells between two grids."""
    if not a or not b:
        return 0
    h = min(len(a), len(b))
    w = min(len(a[0]), len(b[0])) if a and b else 0
    changed = 0
    for y in range(h):
        for x in range(w):
            if a[y][x] != b[y][x]:
                changed += 1
    return changed


def connected_component_estimate(grid: list[list[int]]) -> int:
    """Rough count of distinct non-zero color blobs (4-connectivity)."""
    if not grid:
        return 0
    h, w = len(grid), len(grid[0])
    seen = set()
    components = 0

    def neighbors(y: int, x: int) -> list[tuple[int, int]]:
        out = []
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                out.append((ny, nx))
        return out

    for y in range(h):
        for x in range(w):
            if grid[y][x] == 0 or (y, x) in seen:
                continue
            components += 1
            stack = [(y, x)]
            color = grid[y][x]
            while stack:
                cy, cx = stack.pop()
                if (cy, cx) in seen:
                    continue
                if grid[cy][cx] != color:
                    continue
                seen.add((cy, cx))
                stack.extend(neighbors(cy, cx))
    return components


def detect_novelty(
    frame: FrameData,
    visited_hashes: set[str],
    current_hash: str,
    prev_frame: FrameData | None,
) -> tuple[bool, dict]:
    """Return (is_novel, features dict)."""
    is_new_hash = current_hash not in visited_hashes
    diff_cells = 0
    new_components = 0
    if prev_frame and frame.frame and prev_frame.frame:
        diff_cells = grid_diff_mask(prev_frame.frame[0], frame.frame[0])
        new_components = max(
            0,
            connected_component_estimate(frame.frame[0])
            - connected_component_estimate(prev_frame.frame[0]),
        )
    is_novel = is_new_hash or diff_cells > 50 or new_components > 0
    return is_novel, {
        "new_hash": is_new_hash,
        "diff_cells": diff_cells,
        "new_components": new_components,
    }
