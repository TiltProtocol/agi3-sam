#!/usr/bin/env python3
"""Download all public ARC-AGI-3 demo environments to environment_files/."""

from __future__ import annotations

import argparse
from pathlib import Path

from arc_agi import Arcade, OperationMode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="environment_files",
        help="Directory for downloaded game metadata and source files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arc = Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(output_dir))
    environments = arc.get_environments()
    print(f"Found {len(environments)} public environments")

    failed: list[str] = []
    for env in environments:
        wrapper = arc.make(env.game_id)
        if wrapper is None:
            failed.append(env.game_id)
            print(f"FAIL {env.game_id}")
        else:
            print(f"ok   {env.game_id}")

    downloaded = list(output_dir.rglob("metadata.json"))
    print(f"\nSaved {len(downloaded)} environments to {output_dir.resolve()}")

    if failed:
        print(f"Failed to download {len(failed)} environment(s): {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
