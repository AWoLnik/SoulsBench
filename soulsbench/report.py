"""Compare score cards across runs.

python -m soulsbench.report runs/            # every scorecard.json under runs/
python -m soulsbench.report runs/A runs/B    # specific run directories
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from soulsbench.metrics import ScoreCard, comparison_markdown


def load_cards(paths: list[Path]) -> list[ScoreCard]:
    files: list[Path] = []
    for p in paths:
        files += [p] if p.is_file() else sorted(p.rglob("scorecard.json"))
    cards = []
    for f in files:
        data = json.loads(f.read_text())
        data["sla"].setdefault("realtime", False)  # cards from before the flag existed were paused
        if data["sla"]["realtime"] and "min_action_interval_ms" not in data["sla"]:
            # 2026-10-04..06 "realtime" applied each answer at the next 250 ms tick boundary
            data["adapter"] += " (tick-lagged rt)"
        cards.append(ScoreCard.model_validate(data))
    return cards


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m soulsbench.report",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, help="also write the table to this Markdown file")
    args = parser.parse_args(argv)
    cards = load_cards(args.paths)
    if not cards:
        print("no scorecard.json found", file=sys.stderr)
        return 1
    table = comparison_markdown(cards)
    print(table)
    if args.out:
        args.out.write_text(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
