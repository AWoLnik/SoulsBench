import math

import pytest

from soulsbench.metrics import (
    EpisodeMetrics,
    build_scorecard,
    comparison_markdown,
    percentile,
    scorecard_markdown,
)
from soulsbench.schema import DEFAULT_SLA


def test_percentile_matches_numpy_linear():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)
    assert percentile([7], 95) == 7
    assert math.isnan(percentile([], 50))


def _ep(seed, outcome="death", dealt=100, timeouts=0, decisions=10, latencies=None, survival=30.0):
    return EpisodeMetrics(
        run_id="r",
        episode_id=f"r/seed{seed}",
        adapter="x",
        scenario="gundyr_p1",
        seed=seed,
        outcome=outcome,
        win=outcome == "win",
        boss_hp_dealt=dealt,
        boss_hp_pool=1037,
        boss_hp_dealt_frac=dealt / 1037,
        survival_s=survival,
        player_hp_remaining=0,
        player_hp_max=454,
        player_hp_remaining_frac=0.0,
        estus_used=0,
        ticks=decisions,
        decisions=decisions,
        timeouts=timeouts,
        over_budget=0,
        parse_errors=0,
        adapter_errors=0,
        illegal=1,
        latency_ms_p50=1.0,
        latency_ms_p95=2.0,
        latency_ms_max=3.0,
        step_rtt_ms_p50=1.0,
        step_rtt_ms_p95=2.0,
        wall_time_s=1.0,
        latencies_ms=latencies or [1.0] * decisions,
    )


def test_scorecard_pools_and_excludes_aborted():
    eps = [
        _ep(0, "win", dealt=1037, timeouts=2, latencies=[1.0] * 10),
        _ep(1, "death", dealt=200, latencies=[3.0] * 10),
        _ep(2, "aborted", dealt=0, timeouts=10, latencies=[100.0] * 10),
    ]
    card = build_scorecard(
        eps,
        run_id="r",
        backend="souls_gym",
        scenario="gundyr_p1",
        adapter="x",
        sla=DEFAULT_SLA,
        seeds=[0, 1, 2],
    )
    assert (card.n_episodes, card.n_valid, card.n_aborted) == (3, 2, 1)
    assert card.win_rate == 0.5
    assert card.boss_hp_dealt.mean == pytest.approx(618.5)
    assert card.timeout_rate == pytest.approx(2 / 20)
    assert card.illegal_rate == pytest.approx(2 / 20)
    assert card.fallback_rate == pytest.approx(4 / 20)
    assert card.latency_ms_p50 == pytest.approx(2.0)
    assert not card.meets_min_seeds
    assert any("below the N>=10" in n for n in card.notes)
    assert any("aborted" in n for n in card.notes)
    assert any("RNG" in n for n in card.notes)


def test_scorecard_meets_minimum_with_ten_valid():
    card = build_scorecard(
        [_ep(s) for s in range(10)],
        run_id="r",
        backend="souls_gym",
        scenario="gundyr_p1",
        adapter="x",
        sla=DEFAULT_SLA,
        seeds=list(range(10)),
    )
    assert card.meets_min_seeds
    assert not any("below the N>=10" in n for n in card.notes)


def test_markdown_renders():
    card = build_scorecard(
        [_ep(0, "win", dealt=1037)],
        run_id="r",
        backend="souls_gym",
        scenario="gundyr_p1",
        adapter="expert",
        sla=DEFAULT_SLA,
        seeds=[0],
    )
    md = scorecard_markdown(card)
    assert "win rate | 100.0%" in md and "smoke test" in md
    table = comparison_markdown([card, card])
    assert table.count("| expert |") == 2


def test_aborted_episode_round_trips_through_json():
    ep = _ep(0, "aborted", dealt=0, decisions=0, latencies=[]).model_copy(
        update={"latency_ms_p50": math.nan, "latency_ms_max": math.nan, "step_rtt_ms_p50": math.nan}
    )
    back = EpisodeMetrics.model_validate_json(ep.model_dump_json())
    assert math.isnan(back.latency_ms_p50) and back.outcome == "aborted"
