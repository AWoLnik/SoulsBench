"""SoulsBench's SoulsGym subclass, driven through SoulsGym's *real* step() code against a mocked game.

No Dark Souls III here: this pins down the HP/estus/termination bookkeeping and the action-to-key
mapping, which are the parts that would otherwise only fail live. Needs soulsgym importable:

    uv pip install --no-deps "soulsgym @ https://github.com/amacati/SoulsGym/archive/29c94d89ac8a47846c619019296d0ba55c6e80d5.zip"
    uv pip install "gymnasium>=1.0" numpy psutil pyyaml "iced-x86>=1.21" "pymem>=1.10"
"""

from __future__ import annotations

import importlib
import sys
import time
import types

import numpy as np
import pytest

for _name in ("cv2", "pixel_forge"):  # screen capture only; never used by these tests
    try:
        importlib.import_module(_name)
    except ImportError:
        _stub = types.ModuleType(_name)
        _stub.Capture = _stub.Window = object
        sys.modules[_name] = _stub

pytest.importorskip("soulsgym")
pytestmark = pytest.mark.soulsgym

from gymnasium import spaces  # noqa: E402
from soulsgym.core import static as sg_static  # noqa: E402
from soulsgym.exception import InvalidPlayerStateError  # noqa: E402

from soulsbench.schema import Action, ResetRequest, StepRequest  # noqa: E402
from soulsbench.windows_bridge.errors import BridgeStateError, UnsafeSaveError  # noqa: E402
from soulsbench.windows_bridge.live_env import (  # noqa: E402
    KNIGHT_BASE_STATS,
    LiveBridgeEnv,
    SoulsBenchIudexEnv,
    _tables,
    check_character,
    install_extra_bindings,
)
from soulsbench.windows_bridge.state import DEFAULT_ATTACK_WINDOWS  # noqa: E402

install_extra_bindings()


class FakeGame:
    """Just enough of soulsgym's DarkSoulsIII for SoulsEnv.step()."""

    def __init__(self) -> None:
        self.data = types.SimpleNamespace(
            player_animations=sg_static.player_animations["DarkSoulsIII"],
            critical_player_animations=sg_static.critical_player_animations["DarkSoulsIII"],
            boss_animations=sg_static.boss_animations["DarkSoulsIII"],
        )
        self.player_hp = self.player_max_hp = 454
        self.player_sp = self.player_max_sp = 95
        self.iudex_hp = self.iudex_max_hp = 1037
        self.player_animation = "Idle"
        self.iudex_animation = "IdleBattle"
        self.player_pose = np.array([132.2, 575.1, -68.1, -2.77])
        self.iudex_pose = np.array([141.3, 591.9, -68.8, 0.6])
        self.camera_pose = np.zeros(6)
        self.lock_on = True
        self.player_stats = KNIGHT_BASE_STATS
        self.is_ingame = True
        self.iudex_flags = True
        self.game_speed = 0.0
        self.on_pause = None  # applies "what happened in the game" at the end of a step
        self._time = 0

    @property
    def time(self) -> int:
        if self.game_speed > 0:
            self._time += 10  # 10 ms of game time per read while running
        return self._time

    @staticmethod
    def timed(tend: int, tstart: int) -> float:
        return (tend - tstart) / 1000 if tend >= tstart else tend / 1000

    def pause(self) -> None:
        self.game_speed = 0.0
        if self.on_pause:
            self.on_pause(self)

    def resume(self) -> None:
        self.game_speed = 1.0

    def reset_boss_hp(self, boss_id: str) -> None:
        self.iudex_hp = self.iudex_max_hp

    def reset_player_hp(self) -> None:
        self.player_hp = self.player_max_hp


class FakeInput:
    def __init__(self) -> None:
        self.queued: list[str] = []
        self.updates: list[list[str]] = []
        self.resets = 0

    def add_actions(self, actions: list[str]) -> None:
        self.queued.extend(actions)

    def add_action(self, action: str) -> None:
        self.queued.append(action)

    def update_input(self) -> None:
        self.updates.append(list(self.queued))
        self.queued.clear()

    def reset(self) -> None:
        self.resets += 1
        self.queued.clear()


class FakeExtra:
    def __init__(self, estus=(3, 3)) -> None:
        self.estus_value = estus
        self.invalidated = 0

    def estus(self):
        return self.estus_value

    def invalidate(self) -> None:
        self.invalidated += 1

    def read_all(self) -> dict:
        count, cap = self.estus_value
        return {
            "estus": count,
            "estus_max": cap,
            "player_iframes": False,
            "player_parry_window": False,
        }


def make_env(game: FakeGame | None = None) -> SoulsBenchIudexEnv:
    env = SoulsBenchIudexEnv.__new__(SoulsBenchIudexEnv)
    env.game = game or FakeGame()
    env._game_input = FakeInput()
    env._game_window = types.SimpleNamespace(focus=lambda: None, focused=True, hung=lambda: False)
    env._game_speed = 1.0
    env._skip_steps = False
    env.terminated = False
    env._lock_on_timer = 0
    env._random_player_pose = False
    env._last_hard_reset = time.time()
    env.phase = 1
    env.step_size = 0.25
    env.action_space = spaces.Discrete(20)
    env.extra = FakeExtra()
    env.estus_spent = False
    env.pin_boss_hp = True
    env.force_reload = False
    env._game_state = env.game_state()
    return env


def after_step(**values):
    def apply(game: FakeGame) -> None:
        for k, v in values.items():
            setattr(game, k, v)

    return apply


@pytest.mark.parametrize(
    ("action", "keys"),
    [
        (Action.IDLE, []),
        (Action.MOVE_BACK, ["backward"]),
        (Action.ROLL_LEFT, ["left", "roll"]),
        (Action.BACKSTEP, ["roll"]),
        (Action.LIGHT_ATTACK, ["lightattack"]),
        (Action.HEAVY_ATTACK, ["heavyattack"]),
        (Action.BLOCK, ["block"]),
        (Action.PARRY, ["parry"]),
        (Action.USE_ESTUS, ["useitem"]),
        (Action.TOGGLE_LOCK, ["lock_on"]),
    ],
)
def test_actions_press_the_right_soulsgym_keys(action, keys):
    env = make_env()
    env.step(int(action))
    assert env._game_input.updates[0] == keys
    for key in keys:  # every key name must exist in the SoulsGym binding table
        assert key in sg_static.keybindings["DarkSoulsIII"]


def test_keys_are_only_sent_after_refocusing():
    env = make_env()
    calls = []

    def focus():
        calls.append("focus")
        env._game_window.focused = True

    env._game_window = types.SimpleNamespace(focus=focus, focused=False, hung=lambda: False)
    env.step(int(Action.LIGHT_ATTACK))
    assert calls == ["focus"] and env._game_input.resets == 1
    assert env._game_input.updates[0] == ["lightattack"]


def test_step_runs_one_control_period_and_pauses():
    env = make_env()
    t0 = env.game._time
    env.step(int(Action.IDLE))
    assert env.game.game_speed == 0.0
    assert 0.24 <= (env.game._time - t0) / 1000 <= 0.28


def test_boss_damage_is_tracked_while_his_hp_is_pinned():
    env = make_env()
    env.game.on_pause = after_step(iudex_hp=937)
    env.step(int(Action.LIGHT_ATTACK))
    assert env._game_state.boss_hp == 937
    assert env.game.iudex_hp == 1037  # refilled: Gundyr never reaches his phase-2 threshold
    env.game.on_pause = after_step(iudex_hp=987)
    env.step(int(Action.LIGHT_ATTACK))
    assert env._game_state.boss_hp == 887


def test_player_hp_is_real_so_estus_heals():
    env = make_env()
    env.game.on_pause = after_step(player_hp=300)
    env.step(int(Action.IDLE))
    assert env._game_state.player_hp == 300 and env.game.player_hp == 300  # not refilled
    env.game.on_pause = after_step(player_hp=454)
    env.step(int(Action.USE_ESTUS))
    assert env._game_state.player_hp == 454


@pytest.mark.parametrize("floor_hp", [1, 0])
def test_death_at_the_no_death_floor(floor_hp):
    env = make_env()
    env.game.on_pause = after_step(player_hp=floor_hp)
    env.step(int(Action.IDLE))
    assert env.terminated and env._game_state.player_hp == 0
    assert (
        env.game.player_hp == env.game.player_max_hp
    )  # refilled so the game never processes a death


def test_win_when_the_pool_is_spent():
    env = make_env()
    env.game.on_pause = after_step(iudex_hp=537)  # 500 damage
    env.step(int(Action.LIGHT_ATTACK))
    assert not env.terminated and env._game_state.boss_hp == 537
    env.game.on_pause = after_step(iudex_hp=400)  # 637 more: pool spent
    env.step(int(Action.LIGHT_ATTACK))
    assert env.terminated and env._game_state.boss_hp == 0


def test_leaving_the_arena_is_an_error():
    env = make_env()
    env.game.on_pause = after_step(player_pose=np.array([0.0, 0.0, -68.0, 0.0]))
    with pytest.raises(InvalidPlayerStateError):
        env.step(int(Action.IDLE))


def test_soulsgym_window_is_replaced_by_a_focus_only_window():
    import soulsgym.envs.soulsenv as sg_soulsenv
    import soulsgym.games.game as sg_game

    from soulsbench.windows_bridge.live_env import FocusOnlyWindow, install_focus_only_window

    install_focus_only_window()
    assert sg_game.GameWindow is FocusOnlyWindow
    assert sg_soulsenv.GameWindow is FocusOnlyWindow
    with pytest.raises(RuntimeError, match="never captures"):
        FocusOnlyWindow.img.fget(None)


def test_character_check():
    check_character(types.SimpleNamespace(player_stats=KNIGHT_BASE_STATS), allow_any=False)
    veteran = types.SimpleNamespace(player_stats=(80, 40, 10, 30, 15, 40, 40, 9, 9, 7))
    with pytest.raises(UnsafeSaveError, match="fresh SL9 Knight"):
        check_character(veteran, allow_any=False)
    check_character(veteran, allow_any=True)


def test_unsafe_save_is_refused_before_soulsgym_writes_anything():
    env = make_env()
    env.game.player_stats = (80, 40, 10, 30, 15, 40, 40, 9, 9, 7)
    env._allow_any_save = False
    with pytest.raises(UnsafeSaveError):
        env._set_game_properties()
    assert not hasattr(env.game, "last_bonfire")


@pytest.mark.parametrize(
    ("estus", "spent", "reload"),
    [
        ((3, 3), False, False),
        ((1, 3), False, True),
        ((None, None), False, False),
        ((3, 3), True, True),
    ],
)
def test_spent_estus_triggers_a_bonfire_reload(estus, spent, reload):
    env = make_env()
    env.extra = FakeExtra(estus)
    env.estus_spent = spent  # the last charge never shows in memory, so the bridge also tells it
    assert env._reload_required() is reload


# --- LiveBridgeEnv: the bridge backend on top of the env ------------------------------------------


def make_bridge(monkeypatch) -> LiveBridgeEnv:
    bridge = LiveBridgeEnv.__new__(LiveBridgeEnv)
    bridge.game_speed = 1.0
    bridge.allow_any_save = False
    bridge.tables = _tables(DEFAULT_ATTACK_WINDOWS)
    bridge.status = "ready"
    bridge.detail = {}
    bridge._probe = None
    bridge._ep = None
    env = make_env()

    def soulsgym_reset(seed=None, options=None):  # stands in for SoulsGym's arena reset
        env.game.player_hp, env.game.iudex_hp = env.game.player_max_hp, env.game.iudex_max_hp
        env.game.player_animation = "Idle"
        env.terminated = False
        env._game_state = env.game_state()
        return env.obs, env.info

    monkeypatch.setattr(env, "reset", soulsgym_reset)
    bridge._env = env
    return bridge


def test_bridge_episode_flow(monkeypatch):
    bridge = make_bridge(monkeypatch)
    with pytest.raises(BridgeStateError):
        bridge.step(StepRequest(discrete_id=0))
    reset = bridge.reset(
        ResetRequest(episode_id="e", scenario="gundyr_p1", seed=3, control_period_ms=250)
    )
    state = reset.state
    assert state.tick == 0 and state.target.hp == state.target.hp_max == 1037
    assert state.player.estus == 3 and state.player.locked_on
    assert reset.info["estus"] == 3

    bridge._env.game.on_pause = after_step(iudex_hp=937, player_hp=400)
    resp = bridge.step(StepRequest(discrete_id=int(Action.LIGHT_ATTACK)))
    assert resp.outcome == "running" and not resp.info.illegal
    assert (resp.info.damage_dealt, resp.info.damage_taken) == (100, 54)
    assert resp.state.tick == 1 and 0.25 <= resp.state.t_s <= 0.28  # mock clock: 10 ms/read
    assert resp.state.history[-1].action == Action.LIGHT_ATTACK

    # Mid-roll the same request is illegal and executes as idle.
    bridge._ep.state = bridge._ep.state.model_copy(
        update={"legal_actions_mask": [True] + [False] * 15}
    )
    resp = bridge.step(StepRequest(discrete_id=int(Action.LIGHT_ATTACK)))
    assert resp.info.illegal and resp.info.executed_action == Action.IDLE
    assert bridge._env._game_input.updates[-1] == []


def test_bridge_death_and_timeout(monkeypatch):
    bridge = make_bridge(monkeypatch)
    bridge.reset(ResetRequest(episode_id="e", scenario="gundyr_p1", seed=0))
    bridge._env.game.on_pause = after_step(player_hp=1)
    resp = bridge.step(StepRequest(discrete_id=0))
    assert (resp.terminated, resp.outcome, resp.state.player.hp) == (True, "death", 0)
    with pytest.raises(BridgeStateError):
        bridge.step(StepRequest(discrete_id=0))

    bridge.reset(ResetRequest(episode_id="e2", scenario="gundyr_p1", seed=1))
    bridge._ep.scenario = bridge._ep.scenario.model_copy(update={"max_time_s": 0.6})
    bridge._env.game.on_pause = None
    outcomes = [bridge.step(StepRequest(discrete_id=0)).outcome for _ in range(3)]
    assert outcomes == ["running", "running", "timeout"]
    assert bridge._env._game_input.resets >= 1  # keys released at episode end


def test_reset_wakes_a_paused_game_and_close_leaves_it_running_peacefully(monkeypatch):
    bridge = make_bridge(monkeypatch)
    game = bridge._env.game
    game.game_speed, game.allow_attacks = 0.0, True  # paused since the last session
    speeds_at_reset = []
    original = bridge._env.reset

    def spy_reset(*a, **k):
        speeds_at_reset.append(game.game_speed)
        return original(*a, **k)

    monkeypatch.setattr(bridge._env, "reset", spy_reset)
    bridge.reset(ResetRequest(episode_id="e", scenario="gundyr_p1", seed=0))
    assert speeds_at_reset == [
        1.0
    ]  # resumed (so Windows sees it responding) before SoulsGym's reset
    bridge.close()
    assert game.game_speed == 1.0 and game.allow_attacks is False


def test_full_fight_uses_real_boss_hp_and_kills_him_on_screen(monkeypatch):
    bridge = make_bridge(monkeypatch)
    game = bridge._env.game
    bridge.reset(ResetRequest(episode_id="e", scenario="gundyr_full", seed=0))
    assert bridge._env.pin_boss_hp is False and game.allow_deaths is False

    game.on_pause = after_step(iudex_hp=900)
    resp = bridge.step(StepRequest(discrete_id=int(Action.LIGHT_ATTACK)))
    assert resp.state.target.hp == 900 and game.iudex_hp == 900  # not refilled

    game.on_pause = after_step(iudex_hp=500, iudex_animation="Attack1500")
    resp = bridge.step(StepRequest(discrete_id=0))
    assert resp.state.target.boss_phase == 2 and bridge._env.force_reload

    game.on_pause = after_step(iudex_hp=1)  # the no-death floor
    resp = bridge.step(StepRequest(discrete_id=0))
    assert (resp.outcome, resp.state.target.hp) == ("win", 0)
    assert game.allow_deaths is True and game.iudex_hp == 0 and game.game_speed == 1.0


def test_phase1_scenario_still_pins_the_boss(monkeypatch):
    bridge = make_bridge(monkeypatch)
    bridge.reset(ResetRequest(episode_id="e", scenario="gundyr_p1", seed=0))
    bridge._env.game.on_pause = after_step(iudex_hp=900)
    resp = bridge.step(StepRequest(discrete_id=int(Action.LIGHT_ATTACK)))
    assert resp.state.target.hp == 900 and bridge._env.game.iudex_hp == 1037


def test_force_reload_after_transform_or_kill():
    env = make_env()
    env.force_reload = True
    assert env._reload_required() is True
