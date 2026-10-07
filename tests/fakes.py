"""Unit-test doubles. NOT a benchmark backend: the CLI only runs against the live game.

``FakeCombatEnv`` is a tiny deterministic stand-in that exists to exercise the schema, the state
builder, the harness and the wire protocol without Dark Souls III.
"""

from __future__ import annotations

from collections import deque

from soulsbench.scenarios import get_scenario
from soulsbench.schema import (
    HISTORY_LEN,
    Action,
    BridgeHealth,
    HistoryEntry,
    ResetRequest,
    ResetResponse,
    StepInfo,
    StepRequest,
    StepResponse,
)
from soulsbench.windows_bridge.state import AnimTables, RawSnapshot, build_state

TABLES = AnimTables(
    player={
        "Idle": (0, (0.0, 0.0, 0.0)),
        "Move": (1, (0.0, 0.0, 0.0)),
        "RollingMedium": (2, (0.69, 0.73, 0.75)),
        "AttackRightLight1": (3, (0.63, 0.8, 1.03)),
    },
    boss={
        "Attack3000": (0, "attacks"),
        "IdleBattle": (1, "movement"),
        "WalkFrontBattle": (2, "movement"),
        "SABreak": (3, "misc"),
    },
)


def snapshot(**overrides) -> RawSnapshot:
    base = dict(
        player_hp=454,
        player_hp_max=454,
        player_sp=95,
        player_sp_max=95,
        player_anim="Idle",
        player_anim_time_s=1.0,
        player_pos=(0.0, 0.0, 0.0),
        player_heading=3.14159265,  # facing +y
        locked_on=True,
        boss_hp=1037,
        boss_hp_max=1037,
        boss_phase=1,
        boss_anim="IdleBattle",
        boss_anim_time_s=0.5,
        boss_pos=(0.0, 5.0, 0.0),
        boss_heading=0.0,
        estus=3,
        estus_max=3,
    )
    base.update(overrides)
    return RawSnapshot(**base)


class FakeCombatEnv:
    """Gundyr-shaped toy: light attacks hit for 100, the boss hits for 60 every 4th tick."""

    name = "fake"

    def __init__(self, max_ticks: int = 400) -> None:
        self.max_ticks = max_ticks
        self.closed = False
        self.fail_on_tick: int | None = None
        self._snap = snapshot()
        self._tick = 0
        self._history: deque[HistoryEntry] = deque(maxlen=HISTORY_LEN)
        self._episode_id = ""
        self._scenario = get_scenario("gundyr_p1")
        self._state = None

    def health(self) -> BridgeHealth:
        return BridgeHealth(ok=True, ready=True, bridge_version="test", backend=self.name)

    def _build(self):
        self._state = build_state(
            self._snap,
            TABLES,
            episode_id=self._episode_id,
            scenario=self._scenario,
            tick=self._tick,
            t_s=self._tick * 0.25,
            history=self._history,
        )
        return self._state

    def reset(self, request: ResetRequest) -> ResetResponse:
        self._episode_id = request.episode_id
        self._scenario = get_scenario(request.scenario)
        self._snap = snapshot(boss_pos=(0.0, 1.5, 0.0))
        self._tick = 0
        self._history.clear()
        return ResetResponse(state=self._build())

    def step(self, discrete_id: int) -> StepResponse:
        from soulsbench.backends.base import EnvError

        if self.fail_on_tick is not None and self._tick + 1 >= self.fail_on_tick:
            raise EnvError("simulated bridge failure")
        prev = self._state
        executed = discrete_id if prev.legal_actions_mask[discrete_id] else Action.IDLE
        s = self._snap
        self._tick += 1
        dealt = 100 if executed == Action.LIGHT_ATTACK else 0
        s.boss_hp = max(0, s.boss_hp - dealt)
        taken = 60 if self._tick % 4 == 0 else 0
        healed = 0
        if executed == Action.USE_ESTUS and s.estus:
            s.estus -= 1
            healed = min(250, s.player_hp_max - s.player_hp)
        s.player_hp = max(0, min(s.player_hp_max, s.player_hp - taken + healed))
        s.player_anim = "AttackRightLight1" if executed == Action.LIGHT_ATTACK else "Idle"
        s.player_anim_time_s = 0.25 if executed == Action.LIGHT_ATTACK else 1.0
        if self._tick % 4 == 3:
            s.boss_anim, s.boss_anim_time_s = "Attack3000", 0.2
        else:
            s.boss_anim, s.boss_anim_time_s = "IdleBattle", 0.5
        self._history.append(
            HistoryEntry(
                tick=self._tick,
                t_s=self._tick * 0.25,
                action=int(executed),
                player_hp=s.player_hp,
                target_hp=s.boss_hp,
                player_anim=s.player_anim,
                target_anim=s.boss_anim,
            )
        )
        state = self._build()
        won, died = s.boss_hp == 0, s.player_hp == 0
        truncated = not (won or died) and self._tick >= self.max_ticks
        outcome = "win" if won else "death" if died else "timeout" if truncated else "running"
        return StepResponse(
            state=state,
            terminated=won or died,
            truncated=truncated,
            outcome=outcome,
            info=StepInfo(
                requested_action=discrete_id,
                executed_action=int(executed),
                illegal=executed != discrete_id,
                game_dt_s=0.25,
                damage_dealt=dealt,
                damage_taken=taken,
                healed=healed,
                step_wall_ms=0.1,
            ),
        )

    def act(self, discrete_id: int) -> StepResponse:
        self.acts = getattr(self, "acts", 0) + 1
        return self.step(discrete_id)  # the toy advances one fixed tick per action

    def close(self) -> None:
        self.closed = True


class FakeBackend:
    """Bridge backend double for the HTTP server: same contract as LiveBridgeEnv."""

    backend = "fake"

    def __init__(self) -> None:
        self.env = FakeCombatEnv()
        self.closed = 0
        self._running = False

    def health(self) -> BridgeHealth:
        return BridgeHealth(ok=True, ready=True, bridge_version="test", backend=self.backend)

    def reset(self, req: ResetRequest) -> ResetResponse:
        self._running = True
        return self.env.reset(req)

    def step(self, req: StepRequest) -> StepResponse:
        from soulsbench.windows_bridge.errors import BridgeStateError

        if not self._running:
            raise BridgeStateError("no running episode: call /reset first")
        resp = self.env.step(req.discrete_id)
        if resp.terminated or resp.truncated:
            self._running = False
        return resp

    def act(self, req: StepRequest) -> StepResponse:
        return self.step(req)

    def record_start(self, req) -> None:
        self.recordings = getattr(self, "recordings", []) + [req.name]

    def record_stop(self, req) -> str | None:
        return f"C:/videos/{self.recordings[-1]}_{req.suffix}.mkv"

    def close(self) -> None:
        self.closed += 1
        self._running = False
