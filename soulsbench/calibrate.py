"""Derive per-attack hit windows for the telegraph from logged live runs.

    python -m soulsbench.calibrate runs/<run_dir> [more run dirs] [--min-samples 3]

A tick where the player lost HP while the boss stayed in the same attack animation brackets the
hit between the previous and current animation time. The union of those brackets per attack is its
observed hitbox window. Resolution is one control period (250 ms), so windows are conservative. The
bridge loads the result at start-up; until then telegraphs report ``timing_source: "none"``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from soulsbench.metrics import percentile
from soulsbench.windows_bridge.state import ATTACK_WINDOWS_SCHEMA, DEFAULT_ATTACK_WINDOWS


def hit_brackets(step_files: list[Path]) -> dict[str, list[tuple[float, float]]]:
    brackets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for path in step_files:
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("damage_taken", 0) <= 0 or rec.get("target_kind") != "attack":
                continue
            anim = rec["target_anim"]
            phase = rec.get("state", {}).get("target", {}).get("boss_phase", 1)
            key = f"p2:{anim}" if phase == 2 else anim
            t_pre, t_post = rec["target_anim_time_pre"], rec["target_anim_time"]
            # Only trust ticks where the same attack instance spans the whole tick.
            if rec["target_anim_pre"] == anim and t_post > t_pre:
                brackets[key].append((t_pre, t_post))
    return brackets


def windows_from(
    brackets: dict[str, list[tuple[float, float]]], min_samples: int, trim: float = 10.0
) -> dict[str, dict]:
    """One window per attack: [p_trim of bracket starts, p_(100-trim) of bracket ends].

    Plain min/max let a few misattributed hits (a previous swing landing as the next attack
    starts) stretch windows across the whole animation; trimming keeps them honest.
    """
    out = {}
    for anim, spans in sorted(brackets.items()):
        if len(spans) >= min_samples:
            lo = percentile([a for a, _ in spans], trim)
            hi = percentile([b for _, b in spans], 100 - trim)
            out[anim] = {"windows": [[round(lo, 3), round(max(hi, lo), 3)]], "samples": len(spans)}
    return out


def hit_times(step_files: list[Path], max_bracket_s: float = 0.3) -> dict[str, list[float]]:
    """Per attack, when hits landed (seconds into the animation), from logs fine enough to say:
    true-realtime logs bracket each hit within one action (~150 ms)."""
    times: dict[str, list[float]] = defaultdict(list)
    for anim, t_pre, t_post in _hit_brackets(step_files):
        if t_post - t_pre <= max_bracket_s:
            times[anim].append((t_pre + t_post) / 2)
    return times


def _hit_brackets(step_files: list[Path]):
    for path in step_files:
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("damage_taken", 0) <= 0 or rec.get("target_kind") != "attack":
                continue
            anim = rec["target_anim"]
            phase = rec.get("state", {}).get("target", {}).get("boss_phase", 1)
            key = f"p2:{anim}" if phase == 2 else anim
            t_pre, t_post = rec["target_anim_time_pre"], rec["target_anim_time"]
            if rec["target_anim_pre"] == anim and t_post > t_pre:
                yield key, t_pre, t_post


def windows_from_hit_times(
    times: dict[str, list[float]], min_samples: int, gap_s: float = 0.35, margin_s: float = 0.08
) -> dict[str, dict]:
    """One window per distinct hit of an attack: cluster the hit times (a gap over ``gap_s``
    starts a new hit) and keep clusters with ``min_samples`` hits, padded by ``margin_s``
    (about half the ~150 ms bracket). Multi-hit attacks get several windows."""
    out = {}
    for anim, ts in sorted(times.items()):
        ts = sorted(ts)
        clusters, cur = [], [ts[0]] if ts else []
        for t in ts[1:]:
            if t - cur[-1] > gap_s:
                clusters.append(cur)
                cur = []
            cur.append(t)
        if cur:
            clusters.append(cur)
        wins = [
            [round(max(0.0, c[0] - margin_s), 3), round(c[-1] + margin_s, 3)]
            for c in clusters
            if len(c) >= min_samples
        ]
        if wins:
            out[anim] = {"windows": wins, "samples": len(ts), "source": "realtime hit times"}
    return out


def followup_chances(step_files: list[Path], min_samples: int) -> dict[str, float]:
    """Per attack: the fraction of times the boss went straight from it into another attack."""
    chained: dict[str, list[bool]] = defaultdict(list)
    for path in step_files:
        prev: dict[str, str] = {}  # episode -> boss anim on the previous tick
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            ep, anim = rec["episode_id"], rec["target_anim"]
            last = prev.get(ep)
            if last is not None and anim != last and last.startswith(ATTACK_PREFIXES):
                chained[last].append(anim.startswith(ATTACK_PREFIXES))
            prev[ep] = anim
    return {
        a: round(sum(v) / len(v), 2) for a, v in sorted(chained.items()) if len(v) >= min_samples
    }


ATTACK_PREFIXES = ("Attack", "ThrowAtk")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m soulsbench.calibrate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--boss", default="iudex")
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DEFAULT_ATTACK_WINDOWS)
    parser.add_argument(
        "--hit-times",
        action="store_true",
        help="calibrate from precise hit times (true-realtime logs); attacks without enough "
        "hits keep their current windows",
    )
    args = parser.parse_args(argv)
    files = [f for r in args.runs for f in ([r] if r.is_file() else sorted(r.rglob("steps.jsonl")))]
    data = (
        json.loads(args.out.read_text()) if args.out.exists() else {"schema": ATTACK_WINDOWS_SCHEMA}
    )
    if args.hit_times:
        windows = data.get("bosses", {}).get(args.boss, {}) | windows_from_hit_times(
            hit_times(files), args.min_samples
        )
    else:
        windows = windows_from(hit_brackets(files), args.min_samples)
    data.setdefault("bosses", {})[args.boss] = windows
    if not args.hit_times:  # hit-time runs are a few recent logs; follow-ups want all of them
        data.setdefault("followups", {})[args.boss] = followup_chances(files, args.min_samples)
    data["note"] = (
        f"Calibrated from {len(files)} steps.jsonl file(s) with python -m soulsbench.calibrate."
    )
    args.out.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{len(windows)} attack(s) calibrated -> {args.out}")
    for anim, w in windows.items():
        print(f"  {anim}: {w['windows']} ({w['samples']} hits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
