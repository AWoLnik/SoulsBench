"""The full benchmark: every adapter on the same seeds, interleaved, resumable.

    python -m soulsbench.bench --name full50 --adapters openai,jev,pplx,clef,kev,laya --seeds 50
    python -m soulsbench.bench --name full50 ...same args...   # resumes where it stopped

Seeds run in rounds: seed 0 for every adapter, then seed 1, and so on, with the adapter order
rotated each round. API conditions, game drift and time of night then hit every adapter alike, and
a stop at any point leaves a balanced comparison. Everything lives in ``runs/bench_<name>/``:

* ``bench.json``                    the plan (adapters, seeds, scenario, SLA); a resume must match it
* ``<adapter>/episodes.jsonl``      one line per attempt (the source of truth for resuming)
* ``<adapter>/steps.jsonl``         per-decision log (with full states unless ``--no-log-states``)
* ``<adapter>/scorecard.{json,md}`` and ``comparison.md``, rewritten after every episode

A (adapter, seed) pair counts as done once it has a non-aborted episode. Aborted attempts are
retried (``--retries``); after ``--max-aborts`` aborted pairs in a row the bench stops, and running
the same command again resumes it.

``--record-seeds 0-2`` records those seeds of every adapter with OBS (see
``soulsbench.windows_bridge.recorder``; the obs-websocket password goes in ``./obs_password.txt``).
``--record-best`` records every episode but keeps only each adapter's best one (a win beats a loss;
then most HP left and fastest kill, or most boss damage and longest survival), deleting the rest as
it goes; ``best_videos.json`` remembers the keepers across resumes. Videos land in
``%USERPROFILE%\\Videos\\SoulsBench\\<name>\\<adapter>_seed000_<outcome>.mp4``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from soulsbench import __version__
from soulsbench.adapters import make_adapter
from soulsbench.backends import make_backend
from soulsbench.backends.souls_gym import BridgeUnavailable
from soulsbench.harness import DecisionGuard, run_episode
from soulsbench.metrics import (
    EpisodeMetrics,
    build_scorecard,
    comparison_markdown,
    scorecard_markdown,
)
from soulsbench.scenarios import get_scenario
from soulsbench.schema import DEFAULT_SLA, SLA

logger = logging.getLogger("soulsbench.bench")

OBS_PASSWORD_FILE = Path("obs_password.txt")
# After the first episode of an adapter, more adapter errors than this means it is not reachable
# (a local server down, a bad key): stop instead of benchmarking an idle bot for hours.
MAX_ERROR_RATE = 0.5
# A fight where nearly every decision timed out measures a stalled server, not the model (seen:
# two Kev servers fighting over the GPU). It is recorded as aborted, so it is excluded and retried.
MAX_TIMEOUT_RATE = 0.8


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")


def parse_seeds(spec: str) -> set[int]:
    """'0-2,7' -> {0, 1, 2, 7}."""
    out: set[int] = set()
    for part in filter(None, (p.strip() for p in spec.split(","))):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m soulsbench.bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--name", required=True, help="bench name: runs/bench_<name>/")
    p.add_argument("--adapters", required=True, help="comma list of adapter specs")
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--scenario", default="gundyr_full")
    p.add_argument("--retries", type=int, default=1, help="extra attempts for an aborted episode")
    p.add_argument("--max-aborts", type=int, default=3, help="stop after this many in a row")
    p.add_argument("--record-seeds", default="", help="seeds to record with OBS, e.g. 0-2")
    p.add_argument(
        "--record-best", action="store_true", help="record all, keep each adapter's best video"
    )
    p.add_argument("--no-log-states", action="store_true", help="smaller steps.jsonl")
    p.add_argument(
        "--skip",
        default="",
        help="adapters to leave out of this session (e.g. an API that is down); their seeds stay "
        "queued and a later run without --skip fills them in",
    )
    p.add_argument("--bridge-url", help="default: $SOULSBENCH_BRIDGE_URL, else auto-detect")
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def load_episodes(path: Path) -> list[EpisodeMetrics]:
    if not path.exists():
        return []
    return [EpisodeMetrics.model_validate_json(line) for line in path.read_text().splitlines()]


class Bench:
    def __init__(self, args: argparse.Namespace, sla: SLA) -> None:
        self.args, self.sla = args, sla
        self.scenario = get_scenario(args.scenario)
        self.seeds = list(range(args.seed_start, args.seed_start + args.seeds))
        self.specs = [s.strip() for s in args.adapters.split(",") if s.strip()]
        self.root = args.out / f"bench_{slug(args.name)}"
        self.record_seeds = parse_seeds(args.record_seeds)
        self.record_best = args.record_best
        self.skip = {s.strip() for s in args.skip.split(",") if s.strip()}
        self.best_file = self.root / "best_videos.json"
        self.best: dict[str, dict] = (
            json.loads(self.best_file.read_text()) if self.best_file.exists() else {}
        )
        self.obs_password = (
            OBS_PASSWORD_FILE.read_text().strip() if OBS_PASSWORD_FILE.exists() else ""
        )

    # --- plan -------------------------------------------------------------------------------------

    def plan(self) -> dict:
        return {
            "name": self.args.name,
            "adapters": self.specs,
            "seeds": self.seeds,
            "scenario": self.scenario.name,
            "sla": self.sla.model_dump(),
        }

    def check_plan(self) -> None:
        path = self.root / "bench.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if {k: saved.get(k) for k in self.plan()} != self.plan():
                raise SystemExit(
                    f"{path} was started with a different plan; use the same arguments to resume "
                    "or a new --name."
                )
            return
        self.root.mkdir(parents=True, exist_ok=True)
        info = self.plan() | {
            "soulsbench_version": __version__,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "note": "Same tuned System One prompt and state for every model (tuned on Jev runs).",
        }
        path.write_text(json.dumps(info, indent=2) + "\n")

    # --- run --------------------------------------------------------------------------------------

    def run(self, env) -> int:
        self.check_plan()
        adapters = {}
        for spec in self.specs:
            a = make_adapter(spec)
            a.preflight()
            if self.sla.realtime and hasattr(a, "set_action_lag"):
                a.set_action_lag(self.sla.decide_budget_ms / 1000.0)
            adapters[spec] = a
        dirs = {spec: self.root / slug(a.name) for spec, a in adapters.items()}
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        episodes = {spec: load_episodes(dirs[spec] / "episodes.jsonl") for spec in self.specs}
        done = {
            (spec, e.seed) for spec, eps in episodes.items() for e in eps if e.outcome != "aborted"
        }
        todo = [
            (seed, spec)
            for i, seed in enumerate(self.seeds)
            for spec in self.specs[i % len(self.specs) :] + self.specs[: i % len(self.specs)]
            if (spec, seed) not in done and spec not in self.skip
        ]
        total = len(self.seeds) * len(self.specs)
        print(
            f"bench {self.root}: {len(done)}/{total} done, {len(todo)} to go"
            + (f" (skipping {', '.join(sorted(self.skip))} this session)" if self.skip else "")
            + (f"; recording seeds {sorted(self.record_seeds)}" if self.record_seeds else "")
            + ("; recording every episode, keeping each adapter's best" if self.record_best else "")
        )
        guard = DecisionGuard(self.sla)
        consecutive_aborts, t0, n_run = 0, time.monotonic(), 0
        try:
            for seed, spec in todo:
                adapter, d = adapters[spec], dirs[spec]
                attempts = sum(1 for e in episodes[spec] if e.seed == seed)
                for attempt in range(attempts, attempts + self.args.retries + 1):
                    m = self._episode(env, adapter, guard, d, seed, attempt)
                    unresponsive = m.decisions and m.timeouts / m.decisions > MAX_TIMEOUT_RATE
                    if unresponsive and m.outcome != "aborted":
                        m = m.model_copy(
                            update={
                                "outcome": "aborted",
                                "win": False,
                                "error": f"adapter unresponsive: {m.timeouts}/{m.decisions} "
                                "decisions timed out",
                            }
                        )
                    episodes[spec].append(m)
                    with (d / "episodes.jsonl").open("a") as f:
                        f.write(m.model_dump_json() + "\n")
                    if m.outcome != "aborted" or unresponsive:
                        break
                n_run += 1
                left = len(todo) - n_run
                eta_h = (time.monotonic() - t0) / n_run * left / 3600
                print(
                    f"[{len(done) + n_run}/{total}] seed {seed:>3} {adapter.name:<12} {m.outcome:<7} "
                    f"dmg {m.boss_hp_dealt:>4}/{m.boss_hp_pool}  {m.survival_s:6.1f}s  "
                    f"p50 {m.latency_ms_p50:6.1f}ms  timeouts {m.timeouts:>3}  "
                    f"${m.cost_usd or 0:.4f}  ETA {eta_h:.1f} h"
                    + (f"  error: {m.error}" if m.error else "")
                )
                self._write_reports(adapters, dirs, episodes)
                if len(episodes[spec]) == 1 and m.decisions:
                    rate = m.adapter_errors / m.decisions
                    if rate > MAX_ERROR_RATE:
                        print(
                            f"STOPPED: {adapter.name} failed {rate:.0%} of its calls in its first "
                            f"episode (error: {m.error or 'see steps.jsonl'}). Is it reachable?",
                            file=sys.stderr,
                        )
                        return 3
                if unresponsive:
                    print(
                        f"STOPPED: {adapter.name} timed out on {m.timeouts}/{m.decisions} "
                        "decisions; the episode was recorded as aborted. Check its server, then "
                        "re-run the same command to resume.",
                        file=sys.stderr,
                    )
                    return 3
                consecutive_aborts = consecutive_aborts + 1 if m.outcome == "aborted" else 0
                if consecutive_aborts >= self.args.max_aborts:
                    print(
                        f"STOPPED: {consecutive_aborts} episodes in a row aborted (last: "
                        f"{m.error}). Fix the game/bridge and re-run the same command to resume.",
                        file=sys.stderr,
                    )
                    return 3
        finally:
            guard.close()
            for a in adapters.values():
                a.close()
        print(f"bench complete: {self.root / 'comparison.md'}")
        return 0

    def _episode(self, env, adapter, guard, d: Path, seed: int, attempt: int) -> EpisodeMetrics:
        record = (self.record_best or seed in self.record_seeds) and hasattr(env, "record_start")
        saved: dict[str, str | None] = {"path": None}
        name = f"{slug(adapter.name)}_seed{seed:03d}" + (f"_try{attempt}" if attempt else "")

        def on_start() -> None:
            if not record:
                return
            try:
                label = f"{adapter.name} | seed {seed}"
                env.record_start(name, label, slug(self.args.name), self.obs_password)
            except Exception as e:  # noqa: BLE001 - no video is not a reason to lose the fight
                logger.warning("recording did not start (%s); fighting on without video", e)

        def on_end(outcome: str) -> None:
            if record:
                time.sleep(3.0)  # keep the death or the kill on tape
                saved["path"] = env.record_stop(outcome)

        m = self._play(env, adapter, guard, d, seed, attempt, on_start, on_end)
        if saved["path"]:
            self._keep_video(adapter.name, m, saved["path"])
        return m

    def _keep_video(self, adapter: str, m: EpisodeMetrics, path: str) -> None:
        if not self.record_best or m.seed in self.record_seeds:
            print(f"    video: {path}")
            return
        best = self.best.get(adapter)
        if m.outcome != "aborted" and (best is None or video_rank(m) > tuple(best["rank"])):
            if best is not None:
                delete_video(best["path"])
            self.best[adapter] = {"path": path, "seed": m.seed, "rank": list(video_rank(m))}
            self.best_file.write_text(json.dumps(self.best, indent=2) + "\n")
            print(f"    new best video for {adapter}: {path}")
        else:
            delete_video(path)

    def _play(self, env, adapter, guard, d, seed, attempt, on_start, on_end) -> EpisodeMetrics:
        with (d / "steps.jsonl").open("a") as steps:
            return run_episode(
                env,
                adapter,
                guard,
                run_id=f"bench_{slug(self.args.name)}_{slug(adapter.name)}",
                scenario=self.scenario,
                seed=seed,
                attempt=attempt,
                step_log=steps,
                log_states=not self.args.no_log_states,
                on_start=on_start,
                on_end=on_end,
            )

    def _write_reports(self, adapters, dirs, episodes) -> None:
        cards = []
        for spec, eps in episodes.items():
            if not eps:
                continue
            card = build_scorecard(
                eps,
                run_id=f"bench_{slug(self.args.name)}_{slug(adapters[spec].name)}",
                backend="souls_gym",
                scenario=self.scenario.name,
                adapter=adapters[spec].name,
                sla=self.sla,
                seeds=sorted({e.seed for e in eps}),
            )
            (dirs[spec] / "scorecard.json").write_text(card.model_dump_json(indent=2))
            (dirs[spec] / "scorecard.md").write_text(scorecard_markdown(card))
            cards.append(card)
        cards.sort(key=lambda c: (-(c.win_rate if c.win_rate == c.win_rate else -1), c.adapter))
        (self.root / "comparison.md").write_text(comparison_markdown(cards))


def video_rank(m: EpisodeMetrics) -> tuple:
    """Bigger is better: any win over any loss; wins by HP left then speed, losses by boss
    damage then survival."""
    if m.win:
        return (1, m.player_hp_remaining, -m.survival_s)
    return (0, m.boss_hp_dealt, m.survival_s)


def delete_video(windows_path: str) -> None:
    """Delete a recording the bench made (only ever under Videos\\SoulsBench)."""
    import subprocess

    try:
        local = subprocess.run(
            ["wslpath", "-u", windows_path], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        local = windows_path  # not WSL (tests, or a Windows-side harness)
    path = Path(local)
    if "SoulsBench" not in path.parts or "Videos" not in path.parts:
        raise ValueError(f"refusing to delete {windows_path}: not a SoulsBench recording")
    path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    sys.stdout.reconfigure(line_buffering=True)
    bench = Bench(args, DEFAULT_SLA)
    if (bench.record_seeds or bench.record_best) and not bench.obs_password:
        print(f"note: no {OBS_PASSWORD_FILE}; connecting to OBS without a password")
    try:
        env = make_backend("souls_gym", bridge_url=args.bridge_url)
    except BridgeUnavailable as e:
        print(e, file=sys.stderr)
        return 2
    health = env.health()
    if not health.ready:
        print(f"Bridge is up but not ready: {health.detail.get('status')}", file=sys.stderr)
        return 2
    try:
        return bench.run(env)
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
