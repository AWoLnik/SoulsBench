"""The episode loop: reset -> decide(state, deadline) -> step, enforced against the SLA.

Timeouts, parse failures, adapter errors, over-budget answers and illegal actions all become idle.
The live game is paused (SoulsGym speedhack) between steps, so model thinking time never costs game
time; it is measured in wall-clock and reported as latency.
"""

from __future__ import annotations

import json
import logging
import statistics
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Literal, TextIO

from pydantic import ValidationError

from soulsbench.adapters.base import CombatAdapter, DecisionParseError
from soulsbench.backends.base import CombatEnv, EnvError
from soulsbench.metrics import EpisodeMetrics, percentile
from soulsbench.pricing import cost_usd
from soulsbench.scenarios import Scenario
from soulsbench.schema import SLA, Action, CombatState, Decision, ResetRequest, StepResponse

logger = logging.getLogger(__name__)

DecisionStatus = Literal["ok", "timeout", "over_budget", "parse_error", "error", "illegal"]


@dataclass(frozen=True)
class DecisionOutcome:
    requested: int | None
    executed: int
    status: DecisionStatus
    latency_ms: float
    error: str | None = None


@dataclass
class _Pending:
    future: Future
    t0: float
    t_done: float | None = None  # when the adapter actually answered (not when it was collected)


class DecisionGuard:
    """Calls ``adapter.decide`` on a worker thread and holds it to the decide budget."""

    def __init__(self, sla: SLA, workers: int = 4) -> None:
        self.sla = sla
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="decide")
        self._usage_lock = threading.Lock()
        self._calls = self._tokens_in = self._tokens_out = 0

    def record_usage(self, result: object) -> None:
        """Count one completed adapter call. Late answers (timeouts) are billed too."""
        usage = getattr(result, "usage", None)
        with self._usage_lock:
            self._calls += 1
            if usage is not None:
                self._tokens_in += usage.input_tokens
                self._tokens_out += usage.output_tokens

    def take_usage(self) -> tuple[int, int, int]:
        """(calls, input tokens, output tokens) since the last take."""
        with self._usage_lock:
            out = (self._calls, self._tokens_in, self._tokens_out)
            self._calls = self._tokens_in = self._tokens_out = 0
        return out

    def _on_done(self, pending: _Pending, future: Future) -> None:
        pending.t_done = time.perf_counter()
        if not future.cancelled() and future.exception() is None:
            self.record_usage(future.result())

    def start(self, adapter: CombatAdapter, state: CombatState) -> _Pending:
        """Begin deciding on ``state``; the budget clock starts now."""
        t0 = time.perf_counter()
        pending = _Pending(self._pool.submit(adapter.decide, state, self.sla.decide_budget_ms), t0)
        pending.future.add_done_callback(lambda f: self._on_done(pending, f))
        return pending

    def finish(self, pending: _Pending, apply_to: CombatState) -> DecisionOutcome:
        """Collect a started decision, within what is left of its budget. Legality is checked
        against ``apply_to``, the state the action will actually be executed in."""
        budget_ms = self.sla.decide_budget_ms
        t0, future = pending.t0, pending.future
        try:
            raw = future.result(timeout=max(0.0, budget_ms / 1000.0 - (time.perf_counter() - t0)))
        except FutureTimeout:
            future.cancel()
            return DecisionOutcome(None, Action.IDLE, "timeout", _since_ms(t0))
        except TimeoutError:  # the adapter gave up on its own deadline (e.g. HTTP read timeout)
            return DecisionOutcome(None, Action.IDLE, "timeout", self.sla.decide_budget_ms)
        except DecisionParseError as e:
            return DecisionOutcome(None, Action.IDLE, "parse_error", _since_ms(t0), str(e))
        except Exception as e:  # noqa: BLE001 - any adapter failure is an idle tick, not a crash
            return DecisionOutcome(
                None, Action.IDLE, "error", _since_ms(t0), f"{type(e).__name__}: {e}"
            )
        t_done = pending.t_done or time.perf_counter()  # callback may lag result() by a hair
        latency_ms = (t_done - t0) * 1000.0
        try:
            decision = raw if isinstance(raw, Decision) else Decision.model_validate(raw)
        except ValidationError as e:
            return DecisionOutcome(
                None, Action.IDLE, "parse_error", latency_ms, str(e.errors()[:1])
            )
        requested = decision.discrete_id
        if latency_ms > budget_ms:
            return DecisionOutcome(requested, Action.IDLE, "over_budget", latency_ms)
        if not apply_to.is_legal(requested):
            return DecisionOutcome(requested, Action.IDLE, "illegal", latency_ms)
        return DecisionOutcome(requested, requested, "ok", latency_ms)

    def decide(self, adapter: CombatAdapter, state: CombatState) -> DecisionOutcome:
        return self.finish(self.start(adapter, state), state)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


WARMUP_CALLS = 3


def _warm_up(adapter: CombatAdapter, state: CombatState, guard: DecisionGuard) -> None:
    """Untimed calls on the reset state while the game is still paused, so one-time costs (GPU
    kernel compiles for new input sizes, cold connections) never land inside an episode. Same for
    every adapter; answers and errors are discarded (their tokens are still counted as cost)."""
    for _ in range(WARMUP_CALLS):
        try:
            guard.record_usage(adapter.decide(state, guard.sla.decide_budget_ms))
        except Exception:  # noqa: BLE001 - warm-up must never affect the episode
            pass


def _since_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def episode_id_for(run_id: str, seed: int, attempt: int) -> str:
    return f"{run_id}/seed{seed}" + (f"/try{attempt}" if attempt else "")


def run_episode(
    env: CombatEnv,
    adapter: CombatAdapter,
    guard: DecisionGuard,
    *,
    run_id: str,
    scenario: Scenario,
    seed: int,
    attempt: int = 0,
    step_log: TextIO | None = None,
    log_states: bool = False,
    on_start: Callable[[], None] | None = None,
    on_end: Callable[[str], None] | None = None,
) -> EpisodeMetrics:
    """Play one episode. ``on_start`` runs right before the first action (the game is about to
    run), ``on_end`` with the outcome once it is over (e.g. to start/stop a recording)."""
    sla = guard.sla
    episode_id = episode_id_for(run_id, seed, attempt)
    t_wall0 = time.perf_counter()
    adapter.reset(episode_id, seed=seed)

    latencies: list[float] = []  # only ticks with a real choice (>1 legal action), see below
    decisions = 0
    rtts: list[float] = []
    counts = {"timeout": 0, "over_budget": 0, "parse_error": 0, "error": 0, "illegal": 0}
    estus_used = 0
    outcome: str = "aborted"
    error: str | None = None
    first: CombatState | None = None
    last: CombatState | None = None

    def _tally(decision: DecisionOutcome, basis: CombatState) -> None:
        nonlocal decisions
        decisions += 1
        # A tick where only idle is legal is no decision; an adapter may answer it without
        # thinking, which would flatter its latency. Measure latency on real choices only.
        if sum(basis.legal_actions_mask) > 1:
            latencies.append(decision.latency_ms)
        if decision.status != "ok":
            counts[decision.status] += 1

    try:
        request = ResetRequest(
            episode_id=episode_id,
            scenario=scenario.name,
            seed=seed,
            control_period_ms=sla.control_period_ms,
            realtime=sla.realtime,
        )
        first = last = env.reset(request).state
        guard.take_usage()  # drop calls that finished after the previous episode ended
        _warm_up(adapter, first, guard)  # the game is still paused after reset

        def _record(state: CombatState, applied: DecisionOutcome, act) -> StepResponse:
            nonlocal estus_used
            t0 = time.perf_counter()
            resp = act(applied.executed)
            rtts.append(_since_ms(t0))
            if state.player.estus is not None and resp.state.player.estus is not None:
                estus_used += max(0, state.player.estus - resp.state.player.estus)
            if step_log is not None:
                _write_step(step_log, state, applied, resp, rtts[-1], seed, log_states)
            return resp

        if on_start is not None:
            on_start()
        if sla.realtime:
            # True realtime: the game runs continuously (the first /act starts it). Each answer
            # is applied the moment it arrives (a timeout applies idle at the deadline), and the
            # next decision starts on the state the bridge read right after applying it.
            pending = guard.start(adapter, last)
            t_act = 0.0
            while True:
                state = last
                applied = guard.finish(pending, apply_to=state)
                _tally(applied, state)
                wait_s = sla.min_action_interval_ms / 1000.0 - (time.perf_counter() - t_act)
                if wait_s > 0:
                    time.sleep(wait_s)  # only instant scripted baselines ever wait here
                t_act = time.perf_counter()
                resp = _record(state, applied, env.act)
                last = resp.state
                if resp.terminated or resp.truncated:
                    outcome = resp.outcome
                    break
                pending = guard.start(adapter, last)
        else:
            # Paused: decide while the game is paused, then play one control period.
            while True:
                state = last
                applied = guard.decide(adapter, state)
                _tally(applied, state)
                resp = _record(state, applied, env.step)
                last = resp.state
                if resp.terminated or resp.truncated:
                    outcome = resp.outcome
                    break
    except EnvError as e:
        error = str(e)
        logger.warning("episode %s aborted: %s", episode_id, e)
    if on_end is not None:
        try:
            on_end(outcome)
        except Exception as e:  # noqa: BLE001 - a recording hiccup must not lose the episode
            logger.warning("on_end hook failed: %s", e)
    calls, tokens_in, tokens_out = guard.take_usage()

    boss_pool = first.target.hp_max if first else 0
    boss_dealt = max(0, boss_pool - last.target.hp) if last else 0
    hp_max = first.player.hp_max if first else 0
    hp_left = last.player.hp if last else 0
    return EpisodeMetrics(
        run_id=run_id,
        episode_id=episode_id,
        adapter=adapter.name,
        scenario=scenario.name,
        seed=seed,
        attempt=attempt,
        outcome=outcome,
        win=outcome == "win",
        boss_hp_dealt=boss_dealt,
        boss_hp_pool=boss_pool,
        boss_hp_dealt_frac=boss_dealt / boss_pool if boss_pool else 0.0,
        survival_s=last.t_s if last else 0.0,
        player_hp_remaining=hp_left,
        player_hp_max=hp_max,
        player_hp_remaining_frac=hp_left / hp_max if hp_max else 0.0,
        estus_used=estus_used,
        ticks=last.tick if last else 0,
        decisions=decisions,
        timeouts=counts["timeout"],
        over_budget=counts["over_budget"],
        parse_errors=counts["parse_error"],
        adapter_errors=counts["error"],
        illegal=counts["illegal"],
        latency_ms_p50=percentile(latencies, 50),
        latency_ms_p90=percentile(latencies, 90),
        latency_ms_p95=percentile(latencies, 95),
        latency_ms_p99=percentile(latencies, 99),
        latency_ms_mean=statistics.fmean(latencies) if latencies else float("nan"),
        latency_ms_max=max(latencies, default=float("nan")),
        api_calls=calls,
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        cost_usd=cost_usd(adapter.name, tokens_in, tokens_out),
        step_rtt_ms_p50=percentile(rtts, 50),
        step_rtt_ms_p95=percentile(rtts, 95),
        wall_time_s=time.perf_counter() - t_wall0,
        error=error,
        latencies_ms=[round(x, 3) for x in latencies],
    )


def _write_step(
    out: TextIO,
    state: CombatState,
    decision: DecisionOutcome,
    resp: StepResponse,
    rtt_ms: float,
    seed: int,
    log_states: bool,
) -> None:
    post = resp.state
    record = {
        "episode_id": state.episode_id,
        "seed": seed,
        "tick": post.tick,
        "t_s": round(post.t_s, 4),
        "requested": decision.requested,
        "executed": decision.executed,
        "status": decision.status,
        "latency_ms": round(decision.latency_ms, 3),
        "step_rtt_ms": round(rtt_ms, 3),
        "game_dt_s": round(resp.info.game_dt_s, 4),
        "bridge_illegal": resp.info.illegal,
        "player_hp": post.player.hp,
        "target_hp": post.target.hp,
        "estus": post.player.estus,
        "damage_dealt": resp.info.damage_dealt,
        "damage_taken": resp.info.damage_taken,
        "healed": resp.info.healed,
        "player_anim": post.player.anim,
        "distance": round(post.target.telegraph.distance, 3),
        "target_anim_pre": state.target.telegraph.anim,
        "target_anim_time_pre": round(state.target.telegraph.anim_time_s, 4),
        "target_anim": post.target.telegraph.anim,
        "target_anim_time": round(post.target.telegraph.anim_time_s, 4),
        "target_kind": post.target.telegraph.kind,
        "outcome": resp.outcome,
    }
    if log_states:
        record["state"] = state.model_dump(mode="json")
    out.write(json.dumps(record) + "\n")
