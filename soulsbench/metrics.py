"""Episode metrics, score cards, and their on-disk formats (JSONL + JSON + Markdown)."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from soulsbench.schema import SCHEMA_VERSION, SLA

MIN_SEEDS_FOR_REPORT = 10

EpisodeOutcome = Literal["win", "death", "timeout", "aborted"]

RNG_NOTE = (
    "DS3's RNG is not controllable: seeds fix the adapter's RNG (and the start pose in *_randstart "
    "scenarios), not Gundyr's AI. Same seed + same adapter can still play out differently; compare "
    "means over N>=10 seeds, not single episodes."
)


def percentile(values: Sequence[float], q: float) -> float:
    """Linearly interpolated percentile (numpy's default method). NaN for empty input."""
    if not values:
        return math.nan
    xs = sorted(values)
    k = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


class EpisodeMetrics(BaseModel):
    run_id: str
    episode_id: str
    adapter: str
    scenario: str
    seed: int
    attempt: int = 0
    outcome: EpisodeOutcome
    win: bool
    boss_hp_dealt: int
    boss_hp_pool: int
    boss_hp_dealt_frac: float
    survival_s: float
    player_hp_remaining: int
    player_hp_max: int
    player_hp_remaining_frac: float
    estus_used: int
    ticks: int
    decisions: int
    timeouts: int
    over_budget: int
    parse_errors: int
    adapter_errors: int
    illegal: int
    latency_ms_p50: float
    latency_ms_p90: float = math.nan
    latency_ms_p95: float
    latency_ms_p99: float = math.nan
    latency_ms_mean: float = math.nan
    latency_ms_max: float
    api_calls: int = 0  # completed adapter calls, warm-up and late answers included
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None  # at published prices; 0 for local models, None if unknown
    step_rtt_ms_p50: float
    step_rtt_ms_p95: float
    wall_time_s: float
    error: str | None = None
    latencies_ms: list[float] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _nan_from_null(cls, data):
        """JSON has no NaN: an aborted episode's latencies are written as null. Read them back
        as NaN so episodes.jsonl round-trips (the bench resumes from it)."""
        if isinstance(data, dict):
            floats = [n for n, f in cls.model_fields.items() if f.annotation in (float, "float")]
            return {k: (math.nan if v is None and k in floats else v) for k, v in data.items()}
        return data

    @property
    def fallbacks(self) -> int:
        return (
            self.timeouts
            + self.over_budget
            + self.parse_errors
            + self.adapter_errors
            + self.illegal
        )


class Summary(BaseModel):
    n: int
    mean: float
    std: float
    min: float
    max: float

    @classmethod
    def of(cls, values: Iterable[float]) -> Summary:
        xs = [float(v) for v in values]
        if not xs:
            return cls(n=0, mean=math.nan, std=math.nan, min=math.nan, max=math.nan)
        std = statistics.stdev(xs) if len(xs) > 1 else 0.0
        return cls(n=len(xs), mean=statistics.fmean(xs), std=std, min=min(xs), max=max(xs))


class ScoreCard(BaseModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    created_at: str
    backend: str
    scenario: str
    adapter: str
    sla: SLA
    seeds: list[int]
    n_episodes: int
    n_valid: int
    n_aborted: int
    meets_min_seeds: bool
    win_rate: float
    boss_hp_dealt: Summary
    boss_hp_dealt_frac: Summary
    survival_s: Summary
    player_hp_remaining: Summary
    player_hp_remaining_frac: Summary
    estus_used: Summary
    decisions: int
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p90: float = math.nan
    latency_ms_p99: float = math.nan
    latency_ms_mean: float = math.nan
    latency_ms_max: float = math.nan
    latency_histogram: dict[str, int] = Field(default_factory=dict)  # "lo-hi ms" -> decisions
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None  # all attempts, aborted included: the calls were made
    cost_per_episode_usd: float | None = None
    cost_per_win_usd: float | None = None
    timeout_rate: float
    over_budget_rate: float
    parse_error_rate: float
    adapter_error_rate: float
    illegal_rate: float
    fallback_rate: float
    notes: list[str]


LATENCY_BINS_MS = (0, 50, 100, 150, 200, 250, 300, 500)


def latency_histogram(latencies: Sequence[float]) -> dict[str, int]:
    bins = {}
    for lo, hi in zip(LATENCY_BINS_MS, LATENCY_BINS_MS[1:], strict=False):
        bins[f"{lo}-{hi}"] = sum(lo <= x < hi for x in latencies)
    bins[f"{LATENCY_BINS_MS[-1]}+"] = sum(x >= LATENCY_BINS_MS[-1] for x in latencies)
    return bins


def _rate(count: int, total: int) -> float:
    return count / total if total else math.nan


def build_scorecard(
    episodes: Sequence[EpisodeMetrics],
    *,
    run_id: str,
    backend: str,
    scenario: str,
    adapter: str,
    sla: SLA,
    seeds: Sequence[int],
) -> ScoreCard:
    valid = [e for e in episodes if e.outcome != "aborted"]
    n_aborted = len(episodes) - len(valid)
    decisions = sum(e.decisions for e in valid)
    latencies = [x for e in valid for x in e.latencies_ms]
    costs = [e.cost_usd for e in episodes]
    cost = None if any(c is None for c in costs) else sum(costs)
    wins = sum(e.win for e in valid)
    notes = [RNG_NOTE]
    if not sla.realtime:
        notes.append(
            "PAUSED mode: the game waited for every decision, so latency cost no game time. Not "
            "comparable with realtime (default) results."
        )
    if len(valid) < MIN_SEEDS_FOR_REPORT:
        notes.append(
            f"Only {len(valid)} valid episode(s): below the N>={MIN_SEEDS_FOR_REPORT} reporting "
            "minimum. Treat these numbers as a smoke test, not a result."
        )
    if n_aborted:
        notes.append(f"{n_aborted} episode attempt(s) aborted by bridge/game errors (excluded).")
    return ScoreCard(
        run_id=run_id,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        backend=backend,
        scenario=scenario,
        adapter=adapter,
        sla=sla,
        seeds=list(seeds),
        n_episodes=len(episodes),
        n_valid=len(valid),
        n_aborted=n_aborted,
        meets_min_seeds=len(valid) >= MIN_SEEDS_FOR_REPORT,
        win_rate=statistics.fmean(e.win for e in valid) if valid else math.nan,
        boss_hp_dealt=Summary.of(e.boss_hp_dealt for e in valid),
        boss_hp_dealt_frac=Summary.of(e.boss_hp_dealt_frac for e in valid),
        survival_s=Summary.of(e.survival_s for e in valid),
        player_hp_remaining=Summary.of(e.player_hp_remaining for e in valid),
        player_hp_remaining_frac=Summary.of(e.player_hp_remaining_frac for e in valid),
        estus_used=Summary.of(e.estus_used for e in valid),
        decisions=decisions,
        latency_ms_p50=percentile(latencies, 50),
        latency_ms_p95=percentile(latencies, 95),
        latency_ms_p90=percentile(latencies, 90),
        latency_ms_p99=percentile(latencies, 99),
        latency_ms_mean=statistics.fmean(latencies) if latencies else math.nan,
        latency_ms_max=max(latencies, default=math.nan),
        latency_histogram=latency_histogram(latencies),
        api_calls=sum(e.api_calls for e in episodes),
        input_tokens=sum(e.input_tokens for e in episodes),
        output_tokens=sum(e.output_tokens for e in episodes),
        cost_usd=cost,
        cost_per_episode_usd=cost / len(valid) if cost is not None and valid else None,
        cost_per_win_usd=cost / wins if cost is not None and wins else None,
        timeout_rate=_rate(sum(e.timeouts for e in valid), decisions),
        over_budget_rate=_rate(sum(e.over_budget for e in valid), decisions),
        parse_error_rate=_rate(sum(e.parse_errors for e in valid), decisions),
        adapter_error_rate=_rate(sum(e.adapter_errors for e in valid), decisions),
        illegal_rate=_rate(sum(e.illegal for e in valid), decisions),
        fallback_rate=_rate(sum(e.fallbacks for e in valid), decisions),
        notes=notes,
    )


def _pm(s: Summary, fmt: str = ".1f") -> str:
    if s.n == 0:
        return "n/a"
    return f"{s.mean:{fmt}} ± {s.std:{fmt}}"


def _pct(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{100 * x:.1f}%"


def _usd(x: float | None) -> str:
    return "n/a" if x is None else f"${x:.4f}"


def _ms(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.1f}"


def scorecard_markdown(card: ScoreCard) -> str:
    rows = [
        ("episodes (valid / total)", f"{card.n_valid} / {card.n_episodes}"),
        ("win rate", _pct(card.win_rate)),
        (
            "boss HP dealt",
            f"{_pm(card.boss_hp_dealt)}  ({_pct(card.boss_hp_dealt_frac.mean)} of pool)",
        ),
        ("survival (game s)", _pm(card.survival_s)),
        (
            "player HP remaining",
            f"{_pm(card.player_hp_remaining)}  ({_pct(card.player_hp_remaining_frac.mean)})",
        ),
        ("estus used", _pm(card.estus_used, ".2f")),
        (
            "decide latency p50 / p90 / p95 / p99 / max (ms)",
            " / ".join(
                _ms(x)
                for x in (
                    card.latency_ms_p50,
                    card.latency_ms_p90,
                    card.latency_ms_p95,
                    card.latency_ms_p99,
                    card.latency_ms_max,
                )
            )
            + f"  (mean {_ms(card.latency_ms_mean)})",
        ),
        (
            "latency histogram (ms: decisions)",
            ", ".join(f"{k}: {v}" for k, v in card.latency_histogram.items()) or "n/a",
        ),
        (
            "API calls / input tokens / output tokens",
            f"{card.api_calls} / {card.input_tokens} / {card.output_tokens}",
        ),
        (
            "cost total / per fight / per win (USD)",
            " / ".join(
                _usd(x) for x in (card.cost_usd, card.cost_per_episode_usd, card.cost_per_win_usd)
            ),
        ),
        ("timeout rate", _pct(card.timeout_rate)),
        (
            "over-budget / parse-fail / error rate",
            f"{_pct(card.over_budget_rate)} / {_pct(card.parse_error_rate)} / {_pct(card.adapter_error_rate)}",
        ),
        ("illegal-action rate", _pct(card.illegal_rate)),
        ("any fallback to idle", _pct(card.fallback_rate)),
    ]
    lines = [
        f"# SoulsBench score card: {card.adapter} on {card.scenario}",
        "",
        f"run `{card.run_id}` · backend `{card.backend}` · {card.created_at} · "
        f"SLA {card.sla.decide_budget_ms:g} ms decide / "
        + (
            f"realtime, answers applied on arrival (>= {card.sla.min_action_interval_ms:g} ms apart)"
            if card.sla.realtime
            else f"{card.sla.control_period_ms:g} ms tick / PAUSED while deciding"
        )
        + " · "
        f"seeds {card.seeds[0] if card.seeds else '-'}..{card.seeds[-1] if card.seeds else '-'}",
        "",
        "| metric | value (mean ± std) |",
        "|---|---|",
        *(f"| {k} | {v} |" for k, v in rows),
        "",
        *(f"> {n}" for n in card.notes),
        "",
    ]
    return "\n".join(lines)


def comparison_markdown(cards: Sequence[ScoreCard]) -> str:
    header = (
        "| adapter | scenario | N | win | boss HP dealt | survival s | HP left | p50 ms | p95 ms "
        "| p99 ms | timeout | $ total | $ / fight |"
    )
    lines = [header, "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cards:
        lines.append(
            f"| {c.adapter}{'' if c.sla.realtime else ' (paused)'} | {c.scenario} | {c.n_valid} | "
            f"{_pct(c.win_rate)} | "
            f"{_pm(c.boss_hp_dealt, '.0f')} | {_pm(c.survival_s)} | "
            f"{_pm(c.player_hp_remaining, '.0f')} | {_ms(c.latency_ms_p50)} | "
            f"{_ms(c.latency_ms_p95)} | {_ms(c.latency_ms_p99)} | {_pct(c.timeout_rate)} | "
            f"{_usd(c.cost_usd)} | {_usd(c.cost_per_episode_usd)} |"
        )
    if any(not c.meets_min_seeds for c in cards):
        lines += ["", f"> Rows with N < {MIN_SEEDS_FOR_REPORT} are smoke tests, not results."]
    lines += ["", f"> {RNG_NOTE}", ""]
    return "\n".join(lines)
