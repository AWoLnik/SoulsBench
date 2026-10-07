"""Run the benchmark from WSL against the live game.

    python -m soulsbench.run --backend souls_gym --scenario gundyr_p1 --adapter expert --seeds 10
    python -m soulsbench.run --backend souls_gym --scenario gundyr_p1 --adapter expert,idle,random --seeds 10

Each adapter gets its own run directory under --out with config.json, episodes.jsonl, steps.jsonl,
scorecard.json and scorecard.md. With several adapters, a comparison.md is written as well.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

from soulsbench import __version__
from soulsbench.adapters import make_adapter
from soulsbench.backends import BACKENDS, make_backend
from soulsbench.backends.souls_gym import BridgeUnavailable
from soulsbench.harness import DecisionGuard, run_episode
from soulsbench.metrics import (
    MIN_SEEDS_FOR_REPORT,
    EpisodeMetrics,
    ScoreCard,
    build_scorecard,
    comparison_markdown,
    scorecard_markdown,
)
from soulsbench.scenarios import SCENARIOS, get_scenario
from soulsbench.schema import DEFAULT_SLA, SLA

logger = logging.getLogger("soulsbench.run")


class RunHalted(RuntimeError):
    """Too many seeds in a row aborted: the bridge or game is broken, stop instead of burning seeds."""


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m soulsbench.run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--backend", choices=BACKENDS, default="souls_gym")
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default="gundyr_p1")
    p.add_argument(
        "--adapter",
        required=True,
        help="idle | random | expert | http | http:<name> | openai (comma-separate to run several)",
    )
    p.add_argument(
        "--seeds",
        type=int,
        default=MIN_SEEDS_FOR_REPORT,
        help=f"episodes per adapter, seeds seed-start..seed-start+N-1 (report needs N>={MIN_SEEDS_FOR_REPORT})",
    )
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument(
        "--retries",
        type=int,
        default=1,
        help="re-run a seed whose episode aborted on a bridge error",
    )
    p.add_argument(
        "--max-aborts",
        type=int,
        default=3,
        help="halt the whole run after this many consecutive seeds abort (after retries)",
    )
    p.add_argument("--bridge-url", help="default: $SOULSBENCH_BRIDGE_URL, else auto-detect")
    p.add_argument("--http-base-url", help="base URL for http/openai adapters (overrides env)")
    p.add_argument("--decide-budget-ms", type=float, default=DEFAULT_SLA.decide_budget_ms)
    p.add_argument(
        "--control-period-ms",
        type=float,
        default=DEFAULT_SLA.control_period_ms,
        help="paused mode: game time per decision",
    )
    p.add_argument(
        "--min-action-interval-ms",
        type=float,
        default=DEFAULT_SLA.min_action_interval_ms,
        help="realtime: minimum time between actions (caps instant scripted baselines)",
    )
    p.add_argument(
        "--paused",
        action="store_true",
        help="pause the game while the model decides (non-default; the benchmark is realtime)",
    )
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument(
        "--log-states", action="store_true", help="also write the full CombatState per tick"
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")


def run_adapter(env, spec: str, args: argparse.Namespace, sla: SLA, stamp: str) -> ScoreCard:
    scenario = get_scenario(args.scenario)
    adapter = make_adapter(spec, http_base_url=args.http_base_url)
    adapter.preflight()
    if sla.realtime and hasattr(adapter, "set_action_lag"):
        # Realtime: an answer lands as soon as it arrives, so the lag is the model's latency,
        # at most the decide budget.
        adapter.set_action_lag(sla.decide_budget_ms / 1000.0)
    run_id = f"{stamp}_{scenario.name}_{_slug(adapter.name)}"
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    config = {
        "run_id": run_id,
        "soulsbench_version": __version__,
        "backend": env.name,
        "bridge_url": getattr(env, "url", None),
        "bridge_via": getattr(env, "via", None),
        "bridge_health": env.health().model_dump(mode="json"),
        "scenario": scenario.model_dump(),
        "adapter": adapter.name,
        "adapter_spec": spec,
        "sla": sla.model_dump(),
        "seeds": seeds,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"\n== {adapter.name} on {scenario.name}: {len(seeds)} seed(s) -> {run_dir}")

    guard = DecisionGuard(sla)
    episodes: list[EpisodeMetrics] = []
    halted: str | None = None
    consecutive_aborts = 0
    try:
        with (
            (run_dir / "steps.jsonl").open("w") as steps,
            (run_dir / "episodes.jsonl").open("w") as eps,
        ):
            for seed in seeds:
                for attempt in range(args.retries + 1):
                    m = run_episode(
                        env,
                        adapter,
                        guard,
                        run_id=run_id,
                        scenario=scenario,
                        seed=seed,
                        attempt=attempt,
                        step_log=steps,
                        log_states=args.log_states,
                    )
                    episodes.append(m)
                    eps.write(m.model_dump_json() + "\n")
                    eps.flush()
                    steps.flush()
                    print(
                        f"  seed {seed:>3}{f' (retry {attempt})' if attempt else ''}: {m.outcome:<7} "
                        f"boss dmg {m.boss_hp_dealt:>4}/{m.boss_hp_pool}  survived {m.survival_s:6.1f}s  "
                        f"hp {m.player_hp_remaining:>3}/{m.player_hp_max}  estus {m.estus_used}  "
                        f"p50 {m.latency_ms_p50:6.2f}ms  timeouts {m.timeouts}"
                        + (f"  error: {m.error}" if m.error else "")
                    )
                    if m.outcome != "aborted":
                        break
                consecutive_aborts = consecutive_aborts + 1 if m.outcome == "aborted" else 0
                if consecutive_aborts >= args.max_aborts:
                    halted = (
                        f"{consecutive_aborts} seeds in a row aborted (last error: {m.error}). "
                        "Stopping; check the game and bridge.log, then re-run."
                    )
                    break
    finally:
        guard.close()
        adapter.close()

    card = build_scorecard(
        episodes,
        run_id=run_id,
        backend=env.name,
        scenario=scenario.name,
        adapter=adapter.name,
        sla=sla,
        seeds=seeds,
    )
    (run_dir / "scorecard.json").write_text(card.model_dump_json(indent=2))
    (run_dir / "scorecard.md").write_text(scorecard_markdown(card))
    print(scorecard_markdown(card))
    if halted:
        raise RunHalted(halted)
    return card


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per step otherwise
    sys.stdout.reconfigure(line_buffering=True)  # progress lines show up promptly when piped
    sla = SLA(
        decide_budget_ms=args.decide_budget_ms,
        control_period_ms=args.control_period_ms,
        realtime=not args.paused,
        min_action_interval_ms=args.min_action_interval_ms,
    )
    if sla != DEFAULT_SLA:
        logger.warning(
            "Non-standard SLA %s: results are not comparable to the default benchmark.", sla
        )
    specs = [s for s in args.adapter.split(",") if s.strip()]
    for spec in specs:  # fail fast on config errors before touching the game
        adapter = make_adapter(spec, http_base_url=args.http_base_url)
        try:
            adapter.preflight()
        except (ValueError, NotImplementedError) as e:
            print(f"adapter {spec!r}: {e}", file=sys.stderr)
            return 2
        finally:
            adapter.close()

    try:
        env = make_backend(args.backend, bridge_url=args.bridge_url)
    except BridgeUnavailable as e:
        print(e, file=sys.stderr)
        return 2
    health = env.health()
    if not health.ready:
        print(
            f"Bridge at {getattr(env, 'url', '?')} is up but not ready: {health.detail.get('status')}",
            file=sys.stderr,
        )
        return 2
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    cards = []
    try:
        for spec in specs:
            cards.append(run_adapter(env, spec, args, sla, stamp))
    except RunHalted as e:
        print(f"\nRUN HALTED: {e}", file=sys.stderr)
        return 3
    finally:
        env.close()
    if len(cards) > 1:
        table = comparison_markdown(cards)
        (args.out / f"{stamp}_{args.scenario}_comparison.md").write_text(table)
        print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
