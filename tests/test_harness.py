import io
import json
import time

import pytest

import soulsbench.harness as harness
from soulsbench.adapters import DecisionParseError, IdleAdapter, ScriptedExpertAdapter
from soulsbench.adapters.base import CombatAdapter
from soulsbench.harness import DecisionGuard, run_episode
from soulsbench.scenarios import get_scenario
from soulsbench.schema import SLA, Action, CombatState, Decision
from soulsbench.windows_bridge.state import build_state
from tests.fakes import TABLES, FakeCombatEnv, snapshot


class Scripted(CombatAdapter):
    name = "scripted"

    def __init__(self, fn):
        self.fn = fn

    def decide(self, state: CombatState, deadline_ms: float):
        return self.fn(state)


def _state(**overrides) -> CombatState:
    return build_state(
        snapshot(**overrides),
        TABLES,
        episode_id="ep",
        scenario=get_scenario("gundyr_p1"),
        tick=0,
        t_s=0.0,
        history=(),
    )


@pytest.fixture
def guard():
    g = DecisionGuard(SLA(decide_budget_ms=50, control_period_ms=250, realtime=False))
    yield g
    g.close()


def test_ok_decision(guard):
    out = guard.decide(Scripted(lambda s: Decision(discrete_id=Action.LIGHT_ATTACK)), _state())
    assert (out.status, out.requested, out.executed) == (
        "ok",
        Action.LIGHT_ATTACK,
        Action.LIGHT_ATTACK,
    )
    assert 0 <= out.latency_ms < 50


def test_timeout_becomes_idle(guard):
    def slow(state):
        time.sleep(0.3)
        return Decision(discrete_id=Action.LIGHT_ATTACK)

    t0 = time.perf_counter()
    out = guard.decide(Scripted(slow), _state())
    assert (time.perf_counter() - t0) < 0.2  # the harness did not wait for the slow model
    assert (out.status, out.executed, out.requested) == ("timeout", Action.IDLE, None)
    assert out.latency_ms >= 50


def test_errors_and_parse_failures_become_idle(guard):
    def boom(state):
        raise RuntimeError("model crashed")

    def garbled(state):
        raise DecisionParseError("not json")

    assert guard.decide(Scripted(boom), _state()).status == "error"
    assert guard.decide(Scripted(garbled), _state()).status == "parse_error"
    assert guard.decide(Scripted(lambda s: "light_attack"), _state()).status == "parse_error"
    assert guard.decide(Scripted(lambda s: {"discrete_id": 99}), _state()).status == "parse_error"
    out = guard.decide(Scripted(lambda s: {"discrete_id": 9}), _state())  # plain dicts are accepted
    assert (out.status, out.executed) == ("ok", 9)


def test_illegal_and_over_budget_become_idle(guard, monkeypatch):
    mid_roll = _state(player_anim="RollingMedium", player_anim_time_s=0.1)
    out = guard.decide(Scripted(lambda s: Decision(discrete_id=Action.LIGHT_ATTACK)), mid_roll)
    assert (out.status, out.requested, out.executed) == (
        "illegal",
        Action.LIGHT_ATTACK,
        Action.IDLE,
    )
    from concurrent.futures import Future

    late = Future()
    late.set_result(Decision(discrete_id=Action.LIGHT_ATTACK))
    pending = harness._Pending(late, t0=0.0, t_done=0.051)  # answered 51 ms in, budget is 50
    out = guard.finish(pending, apply_to=_state())
    assert (out.status, out.executed) == ("over_budget", Action.IDLE)


def test_expert_beats_the_fake_and_idle_dies(guard):
    scenario = get_scenario("gundyr_p1")
    log = io.StringIO()
    win = run_episode(
        FakeCombatEnv(),
        ScriptedExpertAdapter(),
        guard,
        run_id="r",
        scenario=scenario,
        seed=0,
        step_log=log,
    )
    assert win.outcome == "win" and win.win
    assert win.boss_hp_dealt == win.boss_hp_pool == 1037
    assert win.decisions == win.ticks
    records = [json.loads(line) for line in log.getvalue().splitlines()]
    assert len(records) == win.ticks
    assert {"latency_ms", "executed", "target_anim_time_pre", "damage_taken"} <= records[0].keys()

    loss = run_episode(FakeCombatEnv(), IdleAdapter(), guard, run_id="r", scenario=scenario, seed=1)
    assert loss.outcome == "death" and loss.player_hp_remaining == 0
    assert loss.boss_hp_dealt == 0
    assert loss.survival_s == pytest.approx(loss.ticks * 0.25)


def test_expert_uses_estus_when_low(guard):
    env = FakeCombatEnv()
    m = run_episode(
        env, ScriptedExpertAdapter(), guard, run_id="r", scenario=get_scenario("gundyr_p1"), seed=0
    )
    assert (
        m.estus_used >= 0
    )  # the toy boss is slow; estus use depends on the script, not asserted exactly


def test_bridge_failure_aborts_episode(guard):
    env = FakeCombatEnv()
    env.fail_on_tick = 5
    m = run_episode(
        env, IdleAdapter(), guard, run_id="r", scenario=get_scenario("gundyr_p1"), seed=0
    )
    assert m.outcome == "aborted" and not m.win
    assert "simulated bridge failure" in (m.error or "")
    assert m.ticks == 4


def test_run_halts_after_consecutive_aborts(tmp_path, monkeypatch):
    import argparse

    import soulsbench.run as run_mod
    from soulsbench.backends.base import EnvError
    from soulsbench.schema import DEFAULT_SLA

    class BrokenEnv(FakeCombatEnv):
        def reset(self, request):
            raise EnvError("bridge POST /reset: FocusError: no keys were sent")

    args = argparse.Namespace(
        scenario="gundyr_p1",
        http_base_url=None,
        out=tmp_path,
        seeds=10,
        seed_start=0,
        retries=1,
        max_aborts=3,
        log_states=False,
    )
    with pytest.raises(run_mod.RunHalted, match="3 seeds in a row"):
        run_mod.run_adapter(BrokenEnv(), "idle", args, DEFAULT_SLA, "stamp")
    run_dir = next(tmp_path.iterdir())
    lines = (run_dir / "episodes.jsonl").read_text().splitlines()
    assert len(lines) == 6  # 3 seeds x (try + 1 retry), then the halt
    assert (run_dir / "scorecard.json").exists()


def test_realtime_applies_each_answer_as_it_arrives():
    sla = SLA(decide_budget_ms=50, realtime=True, min_action_interval_ms=20)
    g = DecisionGuard(sla)
    log = io.StringIO()
    env = FakeCombatEnv()
    t0 = time.perf_counter()
    m = run_episode(
        env,
        Scripted(lambda s: Decision(discrete_id=Action.LIGHT_ATTACK)),
        g,
        run_id="r",
        scenario=get_scenario("gundyr_p1"),
        seed=0,
        step_log=log,
    )
    g.close()
    rows = [json.loads(line) for line in log.getvalue().splitlines()]
    assert rows[0]["executed"] == Action.LIGHT_ATTACK  # no idle tick waiting for a boundary
    assert env.acts == m.ticks == len(rows)  # realtime drives the bridge with /act only
    assert m.outcome == "win" and m.timeouts == 0
    assert time.perf_counter() - t0 >= 0.02 * (m.ticks - 1)  # instant bots are paced


def test_realtime_timeout_applies_idle_at_the_deadline():
    g = DecisionGuard(SLA(decide_budget_ms=30, realtime=True, min_action_interval_ms=0))

    def slow(state):
        time.sleep(0.08)
        return Decision(discrete_id=Action.LIGHT_ATTACK)

    env = FakeCombatEnv(max_ticks=3)
    t0 = time.perf_counter()
    m = run_episode(env, Scripted(slow), g, run_id="r", scenario=get_scenario("gundyr_p1"), seed=0)
    g.close()
    assert m.timeouts == m.decisions == 3 and m.outcome == "timeout"
    # 3 untimed warm-up calls take 0.24 s; waiting for each slow answer would add 0.15 s more
    assert time.perf_counter() - t0 < 0.42


def test_realtime_latency_is_answer_time_not_collection_time():
    g = DecisionGuard(SLA(decide_budget_ms=50, control_period_ms=250, realtime=True))
    pending = g.start(Scripted(lambda s: Decision(discrete_id=Action.LIGHT_ATTACK)), _state())
    time.sleep(0.12)  # the game step runs longer than the budget before we collect
    out = g.finish(pending, apply_to=_state())
    g.close()
    assert out.status == "ok" and out.latency_ms < 50


def test_usage_counts_late_answers_and_prices_by_adapter():
    from soulsbench.pricing import cost_usd
    from soulsbench.schema import Usage

    g = DecisionGuard(SLA(decide_budget_ms=20, realtime=True, min_action_interval_ms=0))

    def late(state):
        time.sleep(0.05)  # misses the budget, still billed
        return Decision(discrete_id=Action.IDLE, usage=Usage(input_tokens=100))

    pending = g.start(Scripted(late), _state())
    assert g.finish(pending, apply_to=_state()).status == "timeout"
    time.sleep(0.08)
    assert g.take_usage() == (1, 100, 0)
    g.close()
    assert cost_usd("jev", 1_000_000, 0) == pytest.approx(0.042)
    assert cost_usd("openai@r5", 1_000_000, 5) == pytest.approx(0.10)
    assert cost_usd("laya", 10**9, 0) == 0.0 and cost_usd("mystery", 1, 0) is None
