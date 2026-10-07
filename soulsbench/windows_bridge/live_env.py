"""Live Dark Souls III through SoulsGym's Iudex Gundyr environment. Windows only (imports soulsgym).

``SoulsBenchIudexEnv`` is SoulsGym's ``IudexEnv`` with these changes:

* **Real player HP.** SoulsGym refills the player every step and tracks damage internally, which
  would make estus a no-op. Here the player keeps real game HP, so estus heals. The game's own
  death is still disabled (SoulsGym's no-death flag); HP <= 1 counts as death and ends the episode.
* **Boss pinned to phase 1.** Gundyr's HP is refilled every step, exactly like SoulsGym, so he never
  transforms; damage is tracked against his HP pool.
* **SoulsBench's 16 actions** instead of SoulsGym's 20, mapped onto SoulsGym's key bindings plus
  one extra binding for block.
* **No screen capture, verified focus.** SoulsGym's capturing window class (which also demands an
  800x450 window) is swapped for :class:`FocusOnlyWindow`. Keys are only sent while DS3 is the
  foreground window.

Reset is SoulsGym's arena reset (teleport player and boss to their start poses, refill HP/SP,
re-acquire lock-on), not a save state; the game has none. When estus was spent, reset first reloads
at the Cemetery of Ash bonfire, which is what refills the flask in the game itself. Estus is counted
by :class:`~soulsbench.windows_bridge.state.EstusTracker` because memory never shows the last drink.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
import soulsgym  # noqa: F401  registers the gymnasium envs, imports platform deps early
from soulsgym.core import static as sg_static
from soulsgym.core.game_input import GameInput
from soulsgym.core.game_window import GameWindow
from soulsgym.envs.darksouls3.iudex import IudexEnv
from soulsgym.envs.utils import RetriesExceededError
from soulsgym.exception import ResetNeeded
from soulsgym.games import DarkSoulsIII

from soulsbench import __version__
from soulsbench.scenarios import Scenario, get_scenario
from soulsbench.schema import (
    HISTORY_LEN,
    Action,
    BridgeHealth,
    CombatState,
    HistoryEntry,
    RecordStartRequest,
    RecordStopRequest,
    ResetRequest,
    ResetResponse,
    StepInfo,
    StepRequest,
    StepResponse,
)
from soulsbench.windows_bridge.errors import (
    BridgeNotReady,
    BridgeRefused,
    BridgeStateError,
    FocusError,
    UnsafeSaveError,
    WindowModeError,
)
from soulsbench.windows_bridge.memory import ExtraMemory
from soulsbench.windows_bridge.recorder import ObsRecorder
from soulsbench.windows_bridge.state import (
    DEFAULT_ATTACK_WINDOWS,
    AnimTables,
    EstusTracker,
    RawSnapshot,
    build_state,
    legal_mask,
    load_attack_windows,
    load_followups,
)

logger = logging.getLogger(__name__)

GAME_ID = "DarkSoulsIII"
# Soul level, vigor, attunement, endurance, vitality, strength, dexterity, intelligence, faith, luck.
KNIGHT_BASE_STATS = (9, 12, 10, 11, 15, 13, 12, 9, 9, 7)
DEATH_HP = 1  # the no-death flag floors HP at 1; treat that as dead
TRANSFORM_ANIM = (
    "Attack1500"  # Gundyr's phase-2 transformation (SoulsGym's phase setup waits on it)
)
# SoulsGym's step loop stops 10 ms before step_size ("processing time" offset); measured live, a
# 250 ms step_size runs 240-242 ms of game time. Adding it back makes the control period honest.
SOULSGYM_EARLY_STOP_S = 0.01

# SoulsBench adds one key binding on top of SoulsGym's (README key table): block = Attack (left hand).
EXTRA_BINDINGS = {"block": ("u", 0x55)}
TAP_ACTIONS = ("useitem",)  # press-and-release instead of hold

ACTION_KEYS: dict[Action, tuple[str, ...]] = {
    Action.IDLE: (),
    Action.MOVE_FORWARD: ("forward",),
    Action.MOVE_BACK: ("backward",),
    Action.MOVE_LEFT: ("left",),
    Action.MOVE_RIGHT: ("right",),
    Action.ROLL_FORWARD: ("forward", "roll"),
    Action.ROLL_BACK: ("backward", "roll"),
    Action.ROLL_LEFT: ("left", "roll"),
    Action.ROLL_RIGHT: ("right", "roll"),
    Action.LIGHT_ATTACK: ("lightattack",),
    Action.HEAVY_ATTACK: ("heavyattack",),
    Action.BLOCK: ("block",),
    Action.PARRY: ("parry",),
    Action.USE_ESTUS: ("useitem",),
    Action.BACKSTEP: ("roll",),  # roll with no direction held is a backstep in DS3
    Action.TOGGLE_LOCK: ("lock_on",),
}


class _OncePerAnimation(logging.Filter):
    """SoulsGym warns about an unknown animation on every tick it is seen; keep the first."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Unknown" not in msg or "animation" not in msg:
            return True
        if msg in self.seen:
            return False
        self.seen.add(msg)
        return True


def install_extra_bindings() -> None:
    """Register SoulsBench's extra keys. Must run before any soulsgym GameInput is created."""
    for action, (key, vk) in EXTRA_BINDINGS.items():
        sg_static.keybindings[GAME_ID][action] = key
        sg_static.keymap[GAME_ID][key] = vk


def check_character(game: DarkSoulsIII, allow_any: bool) -> None:
    """Refuse to touch a character that is not a fresh SL9 Knight.

    SoulsGym overwrites stats, last bonfire, boss flags and play time of the loaded character. That
    must only ever happen to a throwaway save.
    """
    stats = tuple(game.player_stats)
    if stats == KNIGHT_BASE_STATS or allow_any:
        return
    raise UnsafeSaveError(
        f"Loaded character has stats {stats}, not a fresh SL9 Knight {KNIGHT_BASE_STATS}. SoulsGym "
        "would overwrite this character's stats, bonfire and boss flags. Load a dedicated Knight "
        "save made for SoulsBench, or pass --allow-any-save if this character is disposable."
    )


def check_window_mode(game: DarkSoulsIII) -> None:
    """Refuse exclusive fullscreen: SoulsGym re-focuses the game window at every reset, and an
    exclusive-fullscreen game minimizes or mode-switches whenever focus moves."""
    try:
        mode = game.screen_mode
    except Exception as e:  # unreadable: let the user-facing doctor check cover it
        logger.warning("could not read the screen mode (%s); make sure DS3 is windowed", e)
        return
    if mode != "window":
        raise WindowModeError(
            "DS3 is running fullscreen. Set Settings > Graphics > Screen Mode: Windowed (any "
            "resolution, e.g. 1920x1080 for recording), then restart the bridge."
        )


class FocusOnlyWindow:
    """Drop-in for SoulsGym's ``GameWindow`` that focuses the game but never captures it.

    SoulsBench reads the fight from game memory, never from pixels. SoulsGym's ``GameWindow`` starts
    a screen capture in its constructor and refuses any window that is not ~800x450 (its image
    crop check). Keeping only what the env uses (``focus``, ``focused``, ``close``) lets DS3 run at
    any windowed resolution and leaves no capture session running next to your own recording.
    """

    def __init__(self, game_id: str, *args, **kwargs) -> None:
        self.title = GameWindow.window_ids[game_id]

    def _hwnd(self) -> int:
        import win32gui

        hwnd = win32gui.FindWindow(None, self.title)
        if not hwnd:
            raise FocusError(f"no window titled {self.title!r}")
        return hwnd

    @property
    def focused(self) -> bool:
        import win32gui

        return win32gui.GetForegroundWindow() == self._hwnd()

    def focus(self, attempts: int = 3) -> None:
        """Bring DS3 to the foreground and verify it, or raise. Keystrokes go to whatever window
        is in front, so never return without DS3 being there."""
        import win32api
        import win32con
        import win32gui

        hwnd = self._hwnd()
        if self.hung(hwnd):
            raise FocusError(
                "DS3 is not responding (a game paused for more than ~5 s looks hung to Windows); "
                "resume it before focusing"
            )
        for _ in range(attempts):
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            # Windows only lets the foreground process move focus, and the bridge usually is not
            # it (you typed in a terminal last). Tapping Alt makes this process the last input
            # source, which lifts that lock: the standard SetForegroundWindow workaround.
            win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
            try:
                win32gui.SetForegroundWindow(hwnd)
            except win32gui.error:
                pass
            finally:
                win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
            time.sleep(0.1)
            if win32gui.GetForegroundWindow() == hwnd:
                break
        # Like SoulsGym, click inside the window so the game takes the input focus (and, unlike
        # SoulsGym, release the button again).
        left, top, _, _ = win32gui.GetWindowRect(hwnd)
        try:
            win32api.SetCursorPos((left + 100, top + 100))
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        except win32api.error as e:
            raise FocusError(
                f"Windows refused mouse input ({e}): is the screen locked, a UAC prompt open, or DS3 "
                "not responding? No keys were sent."
            ) from None
        time.sleep(0.05)
        if win32gui.GetForegroundWindow() != hwnd:
            raise FocusError(
                "could not bring the DS3 window to the front; no keys were sent. Make sure it is "
                "not minimized or covered, and keep hands off while episodes run."
            )

    def hung(self, hwnd: int | None = None) -> bool:
        import ctypes

        return bool(ctypes.windll.user32.IsHungAppWindow(hwnd or self._hwnd()))

    @property
    def img(self):
        raise RuntimeError("SoulsBench never captures the screen; the state comes from memory")

    def close(self) -> None:
        pass


def install_focus_only_window() -> None:
    """Swap SoulsGym's capturing GameWindow for FocusOnlyWindow. Run before any Game/Env exists."""
    import soulsgym.envs.soulsenv as sg_soulsenv
    import soulsgym.games.game as sg_game

    sg_game.GameWindow = FocusOnlyWindow
    sg_soulsenv.GameWindow = FocusOnlyWindow


class SoulsBenchIudexEnv(IudexEnv):
    def __init__(self, *, game_speed: float, allow_any_save: bool) -> None:
        self._allow_any_save = (
            allow_any_save  # read by _set_game_properties during super().__init__
        )
        super().__init__(game_speed=game_speed, phase=1, random_player_pose=False)
        self._game_input.press_and_release_actions = (
            GameInput.press_and_release_actions + TAP_ACTIONS
        )
        self.extra = ExtraMemory(self.game)
        self.estus_spent = False  # set by the bridge when an episode drank; next reset refills
        self.pin_boss_hp = True  # phase-1 scenarios; full-fight scenarios turn it off
        self.force_reload = False  # boss transformed or died: only a reload restores phase 1

    def _set_game_properties(self):
        check_character(self.game, self._allow_any_save)  # before SoulsGym writes anything
        super()._set_game_properties()

    def _apply_action(self, action: int):
        # Keys go to the foreground window. If anything took focus since the last tick, get DS3
        # back first; focus() raises rather than let keystrokes land in another program.
        if not self._game_window.focused:
            self._game_input.reset()
            self._game_window.focus()
        keys = ACTION_KEYS[Action(action)]
        if keys:
            self._game_input.add_actions(list(keys))
        self._game_input.update_input()

    def _step_hook(self):
        if self.pin_boss_hp:
            self.game.reset_boss_hp(
                self.ENV_ID
            )  # phase lock: Gundyr never reaches his transform HP
        if self._game_state.player_hp == 0:
            self.game.reset_player_hp()  # episode is over; keep the game from processing a death

    def _step_check(self, game_state) -> bool:
        if game_state.player_hp <= 0:
            return True  # a death, handled in _update_game_state instead of raising
        return super()._step_check(game_state)

    def _update_game_state(self, game_state, player_animation_td: float, boss_animation_td: float):
        if self._game_state is None or self.terminated:
            raise ResetNeeded("SoulsEnv.step() called before reset")
        prev = self._game_state
        if game_state.player_animation == prev.player_animation:
            player_animation_td += prev.player_animation_duration
        if game_state.boss_animation == prev.boss_animation:
            boss_animation_td += prev.boss_animation_duration
        # The boss was refilled to max at the end of the previous step, so max - now is this step's
        # damage. The player's HP is the real game value.
        if self.pin_boss_hp:
            boss_hp = prev.boss_hp - (game_state.boss_max_hp - game_state.boss_hp)
        else:  # real HP; the global no-death flag floors it at 1 like the player's
            boss_hp = 0 if game_state.boss_hp <= DEATH_HP else game_state.boss_hp
        self._game_state = game_state
        self._game_state.player_animation_duration = player_animation_td
        self._game_state.boss_animation_duration = boss_animation_td
        self._game_state.boss_hp = max(boss_hp, 0)
        if game_state.player_hp <= DEATH_HP:
            self._game_state.player_hp = 0
        self.terminated = self._game_state.player_hp == 0 or self._game_state.boss_hp == 0

    def _reload_required(self) -> bool:
        if super()._reload_required():
            return True
        count, cap = self.extra.estus()
        if self.estus_spent or (count is not None and cap is not None and count < cap):
            logger.info("Estus was used: reloading at the bonfire to refill the flask")
            return True
        if self.force_reload:
            logger.info("Boss transformed or died last episode: reloading to restore phase 1")
            return True
        return False

    def _reload(self):
        super()._reload()
        self.extra.invalidate()
        self.estus_spent = False
        self.force_reload = False


@dataclass
class _Episode:
    episode_id: str
    scenario: Scenario
    boss_hp_max: int
    boss_phase: int = 1
    state: CombatState | None = None
    tick: int = 0
    t_s: float = 0.0
    done: bool = False
    history: deque[HistoryEntry] = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    estus: EstusTracker = field(default_factory=lambda: EstusTracker.begin(None, None))
    realtime: bool = False
    clock: _AnimClock | None = None  # realtime: started by the first /act
    t_game_ms: int = 0  # realtime: game time of the last observation


def _tables(attack_windows_path: Path) -> AnimTables:
    player = {
        name: (int(v["ID"]), tuple(float(t) for t in v["timings"]))
        for name, v in sg_static.player_animations[GAME_ID].items()
    }
    boss = {
        name: (int(v["ID"]), str(v["type"]))
        for name, v in sg_static.boss_animations[GAME_ID]["iudex"]["all"].items()
    }
    return AnimTables(
        player=player,
        boss=boss,
        attack_windows=load_attack_windows("iudex", attack_windows_path),
        followups=load_followups("iudex", attack_windows_path),
    )


class _AnimClock:
    """Times player and boss animations while the game runs.

    SoulsGym measures animation durations inside its step loop by polling the animation names.
    Realtime has no step loop (the game just runs), so this thread polls instead, every few ms.
    """

    def __init__(self, env: SoulsBenchIudexEnv, lock: threading.RLock, period_s: float = 0.004):
        self.env, self.lock, self.period_s = env, lock, period_s
        gs, t = env._game_state, env.game.time
        self.player = (gs.player_animation, t - int(1000 * gs.player_animation_duration))
        self.boss = (gs.boss_animation, t - int(1000 * gs.boss_animation_duration))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="anim-clock", daemon=True)
        self._thread.start()

    def sample(self) -> int:
        """Record animation changes; returns the game time (ms). Call with ``lock`` held."""
        game = self.env.game
        t = game.time
        player = game.player_animation
        boss = getattr(game, self.env.ENV_ID + "_animation")
        if player != self.player[0]:
            self.player = (player, t)
        if boss != self.boss[0]:
            self.boss = (boss, t)
        return t

    def durations(self, player_anim: str, boss_anim: str, t_ms: int) -> tuple[float, float]:
        """Seconds into the given animations at game time ``t_ms``; 0 if one just started."""

        def since(track: tuple[str, int], anim: str) -> float:
            return max(0.0, (t_ms - track[1]) / 1000.0) if track[0] == anim else 0.0

        return since(self.player, player_anim), since(self.boss, boss_anim)

    def _run(self) -> None:
        while not self._stop.wait(self.period_s):
            try:
                with self.lock:
                    self.sample()
            except Exception as e:  # noqa: BLE001 - a missed sample only blurs timing
                logger.debug("anim clock sample failed: %s", e)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


class LiveBridgeEnv:
    """The bridge's backend: CombatState in, Discrete(16) out, on the live game."""

    backend = "souls_gym"

    def __init__(
        self,
        *,
        game_speed: float = 1.0,
        allow_any_save: bool = False,
        attack_windows_path: Path = DEFAULT_ATTACK_WINDOWS,
    ) -> None:
        install_extra_bindings()
        install_focus_only_window()
        logging.getLogger("soulsgym.envs.soulsenv").addFilter(_OncePerAnimation())
        self.game_speed = game_speed
        self.allow_any_save = allow_any_save
        self.tables = _tables(attack_windows_path)
        self.status = "starting"
        self.detail: dict[str, object] = {
            "soulsgym": pkg_version("soulsgym"),
            "game_speed": game_speed,
            "calibrated_attacks": len(self.tables.attack_windows),
        }
        self._env: SoulsBenchIudexEnv | None = None
        self._probe: DarkSoulsIII | None = None
        self._ep: _Episode | None = None
        self._game_lock = threading.RLock()  # realtime: request thread vs. animation clock
        self._recorder = ObsRecorder()

    # --- lifecycle ----------------------------------------------------------------------------------

    def start(self, stop: threading.Event | None = None, poll_s: float = 2.0) -> None:
        """Block until the player is in-game, then build the SoulsGym env (runs the save check)."""
        self.status = "waiting for the player to load into the game"
        stop = stop or threading.Event()
        probe = None
        alive_polls = 0  # require two polls in a row: right after a death the HP reads 0 first
        while not stop.is_set():
            try:
                probe = probe or DarkSoulsIII()
                probe.clear_cache()
                alive_polls = alive_polls + 1 if probe.is_ingame and probe.player_hp > 0 else 0
                if alive_polls >= 2:
                    break
            except Exception as e:  # window/process not up yet, or mid-load
                logger.debug("waiting for game: %s", e)
                alive_polls = 0
            stop.wait(poll_s)
        if stop.is_set():
            return
        # Keep the probe alive: it owns the speedhack connection, a singleton the env's game object
        # reuses. The DLL serves one controller at a time, so a second connection would stall.
        self._probe = probe
        self.status = "building SoulsGym env"
        try:
            check_window_mode(probe)
            self._env = SoulsBenchIudexEnv(
                game_speed=self.game_speed, allow_any_save=self.allow_any_save
            )
        except BridgeRefused as e:
            self.status = f"refused: {e}"
            raise
        self.status = "ready"

    def health(self) -> BridgeHealth:
        return BridgeHealth(
            ok=True,
            ready=self.status == "ready",
            bridge_version=__version__,
            backend=self.backend,
            detail={
                "status": self.status,
                "episode": self._ep.episode_id if self._ep else None,
                **self.detail,
            },
        )

    def close(self) -> None:
        """End of a harness session: release every key and leave the game running but peaceful.

        A game left paused (speed 0) for more than a few seconds is flagged "Not Responding" by
        Windows, and focusing it fails. So between sessions it runs, with attacks disabled
        globally; the next reset re-enables them (SoulsGym's reset does).
        """
        env = self._env
        if env is not None:
            env._game_input.reset()
            env.game.allow_attacks = False
            env.game.resume()
        self._ep = None

    def _kill_boss_on_screen(self, env: SoulsBenchIudexEnv) -> None:
        """Full-fight win: lift the no-death flag and let the death animation play out."""
        env.force_reload = True
        env._game_input.reset()
        env.game.allow_deaths = True
        env.game.iudex_hp = 0
        env.game.resume()

    def _wait_responsive(self, env: SoulsBenchIudexEnv, timeout_s: float = 10.0) -> None:
        deadline = time.monotonic() + timeout_s
        while env._game_window.hung():
            if time.monotonic() > deadline:
                raise FocusError(f"DS3 still not responding after {timeout_s:.0f} s of running")
            time.sleep(0.05)

    def _wake(self, env: SoulsBenchIudexEnv, timeout_s: float = 10.0) -> None:
        """Run the game until Windows sees it responding again (it may have sat paused)."""
        env.game.allow_attacks = False  # nobody hits anybody while we wait
        env.game.resume()
        deadline = time.monotonic() + timeout_s
        while env._game_window.hung():
            if time.monotonic() > deadline:
                raise FocusError(f"DS3 still not responding after {timeout_s:.0f} s of running")
            time.sleep(0.1)

    def shutdown(self) -> None:
        """Bridge exit: SoulsGym's close (resume, restore game flags, reload via respawn)."""
        if self._env is not None:
            self._env.close()
            self._env = None
        self.status = "stopped"

    # --- episode ------------------------------------------------------------------------------------

    def _require_env(self) -> SoulsBenchIudexEnv:
        if self._env is None or self.status != "ready":
            raise BridgeNotReady(f"bridge not ready: {self.status}")
        return self._env

    def reset(self, req: ResetRequest) -> ResetResponse:
        scenario = get_scenario(req.scenario)
        env = self._require_env()
        self._stop_clock()
        self._ep = None
        env._random_player_pose = scenario.random_start_pose
        env.step_size = req.control_period_ms / 1000.0 + SOULSGYM_EARLY_STOP_S
        random.seed(req.seed)  # SoulsGym draws random start poses with random.choice
        np.random.seed(req.seed % 2**32)
        env.pin_boss_hp = scenario.pin_boss_hp
        self._wake(env)
        env.game.allow_deaths = False  # a full-fight win lifts it to let the boss die on screen
        try:
            env.reset()  # leaves the game paused
        except RetriesExceededError:
            # Seen live right after a full-fight kill: the first reload comes back before the boss
            # is loaded and the fog-gate walk fails. A second reload from scratch works.
            logger.warning("SoulsGym reset failed; reloading once more and retrying")
            env.force_reload, env._arena_init = True, False
            self._wake(env)
            env.game.allow_deaths = False
            env.reset()
        boss_hp_max = int(env._game_state.boss_hp)
        ep = _Episode(
            episode_id=req.episode_id,
            scenario=scenario,
            boss_hp_max=boss_hp_max,
            realtime=req.realtime,
        )
        snap = self._snapshot(ep)
        ep.estus = EstusTracker.begin(snap.estus, snap.estus_max)
        ep.state = build_state(
            snap,
            self.tables,
            episode_id=ep.episode_id,
            scenario=scenario,
            tick=0,
            t_s=0.0,
            history=(),
        )
        self._ep = ep
        info = {
            "start_pose": [round(float(v), 3) for v in env._game_state.player_pose],
            "estus": snap.estus,
            "estus_max": snap.estus_max,
            "control_period_ms": req.control_period_ms,
            "game_speed": self.game_speed,
        }
        return ResetResponse(state=ep.state, info=info)

    def step(self, req: StepRequest) -> StepResponse:
        """Paused mode: play one control period with the action, then pause."""
        env, ep = self._running_episode(realtime=False)
        prev = ep.state
        requested = Action(req.discrete_id)
        executed = requested if prev.legal_actions_mask[requested] else Action.IDLE
        t_wall = time.perf_counter()
        game = env.game
        t_game = game.time
        try:
            env.step(int(executed))  # resume -> keys -> run control period -> pause
        except Exception:
            ep.done = True
            env._game_input.reset()
            raise
        dt = game.timed(game.time, t_game)
        return self._finish(env, ep, prev, requested, executed, self._snapshot(ep), dt, t_wall)

    def act(self, req: StepRequest) -> StepResponse:
        """Realtime: the game keeps running. Apply the action now (legality checked against the
        game as it is at this moment) and return the state read right after the keys went in."""
        env, ep = self._running_episode(realtime=True)
        prev = ep.state
        requested = Action(req.discrete_id)
        t_wall = time.perf_counter()
        try:
            if ep.clock is None:  # first action of the episode: start the game
                with self._game_lock:
                    ep.t_game_ms = env.game.time
                    env.game.game_speed = env._game_speed
                    ep.clock = _AnimClock(env, self._game_lock)
                # The game sat paused through reset and the model's warm-up; past ~5 s Windows
                # flags it "Not Responding" and focusing it fails. It clears once it runs.
                self._wait_responsive(env)
            now = self._observe(env, ep)
            now.estus = ep.estus.remaining(now.estus)  # memory never shows the last drink
            mask, _ = legal_mask(now, self.tables)
            executed = requested if mask[requested] else Action.IDLE
            env._apply_action(int(executed))  # taps hold their key ~50 ms
            snap = self._observe(env, ep)
        except Exception:
            ep.done = True
            self._stop_clock()
            env._game_input.reset()
            raise
        with self._game_lock:
            t_now = env.game.time
        dt = env.game.timed(t_now, ep.t_game_ms)
        ep.t_game_ms = t_now
        resp = self._finish(env, ep, prev, requested, executed, snap, dt, t_wall)
        if ep.done:
            self._stop_clock()
            if resp.outcome != "win":  # a win lets the boss's death animation play out
                env.game.pause()  # as in paused mode: the game waits for the next reset
        return resp

    def record_start(self, req: RecordStartRequest) -> None:
        folder = Path.home() / "Videos" / "SoulsBench" / req.subdir  # never anywhere else
        self._recorder.start(req.name, req.label, str(folder), req.password)

    def record_stop(self, req: RecordStopRequest) -> str | None:
        return self._recorder.stop(req.suffix)

    def _running_episode(self, realtime: bool) -> tuple[SoulsBenchIudexEnv, _Episode]:
        env = self._require_env()
        ep = self._ep
        if ep is None or ep.done or ep.state is None:
            raise BridgeStateError("no running episode: call /reset first")
        if ep.realtime != realtime:
            mode = "realtime (use /act)" if ep.realtime else "paused (use /step)"
            raise BridgeStateError(f"episode was reset in {mode} mode")
        return env, ep

    def _observe(self, env: SoulsBenchIudexEnv, ep: _Episode) -> RawSnapshot:
        """Realtime counterpart of SoulsGym's step post-processing, without the step: read the
        game now, time animations with the clock, re-lock, refill pinned HP."""
        with self._game_lock:
            t_ms = ep.clock.sample()
            gs = env.game_state()
            p_dur, b_dur = ep.clock.durations(gs.player_animation, gs.boss_animation, t_ms)
        if not gs.lock_on and gs.player_animation not in ("ThrowAtk", "ThrowDef"):
            env._lock_on()  # during grabs the lock cannot be established
        else:
            env._lock_on_timer = 0
        if env.terminated:
            pass  # died or won on the pre-action read; keep that final state
        elif not env._step_check(gs):
            env._handle_critical_game_state(gs)
        else:
            env._update_game_state(gs, p_dur, b_dur)
            env._game_state.player_animation_duration = p_dur
            env._game_state.boss_animation_duration = b_dur
            env._step_hook()
        return self._snapshot(ep)

    def _stop_clock(self) -> None:
        ep = self._ep
        if ep is not None and ep.clock is not None:
            ep.clock.stop()
            ep.clock = None

    def _finish(
        self,
        env: SoulsBenchIudexEnv,
        ep: _Episode,
        prev: CombatState,
        requested: Action,
        executed: Action,
        snap: RawSnapshot,
        dt: float,
        t_wall: float,
    ) -> StepResponse:
        """Shared bookkeeping after a step or an act: estus, phase 2, win/death, history."""
        ep.tick += 1
        ep.t_s += dt
        hp_delta = snap.player_hp - prev.player.hp
        ep.estus.update(snap.player_anim, max(0, hp_delta), snap.estus)
        snap.estus = ep.estus.remaining(snap.estus)
        if ep.estus.used:
            env.estus_spent = True
        won, died = snap.boss_hp <= 0, snap.player_hp <= 0
        if not ep.scenario.pin_boss_hp:
            if snap.boss_anim == TRANSFORM_ANIM and ep.boss_phase == 1:
                ep.boss_phase = snap.boss_phase = 2
                env.force_reload = True
            if won:
                self._kill_boss_on_screen(env)
        terminated = won or died
        truncated = not terminated and ep.t_s >= ep.scenario.max_time_s
        outcome = "win" if won else "death" if died else "timeout" if truncated else "running"
        ep.history.append(
            HistoryEntry(
                tick=ep.tick,
                t_s=ep.t_s,
                action=int(executed),
                player_hp=snap.player_hp,
                target_hp=snap.boss_hp,
                player_anim=snap.player_anim,
                target_anim=snap.boss_anim,
            )
        )
        ep.state = build_state(
            snap,
            self.tables,
            episode_id=ep.episode_id,
            scenario=ep.scenario,
            tick=ep.tick,
            t_s=ep.t_s,
            history=ep.history,
        )
        if terminated or truncated:
            ep.done = True
            env._game_input.reset()  # release held keys; the game stays paused until next reset
        return StepResponse(
            state=ep.state,
            terminated=terminated,
            truncated=truncated,
            outcome=outcome,
            info=StepInfo(
                requested_action=int(requested),
                executed_action=int(executed),
                illegal=executed != requested,
                game_dt_s=dt,
                damage_dealt=max(0, prev.target.hp - snap.boss_hp),
                damage_taken=max(0, -hp_delta),
                healed=max(0, hp_delta),
                step_wall_ms=(time.perf_counter() - t_wall) * 1000.0,
            ),
        )

    def _snapshot(self, ep: _Episode) -> RawSnapshot:
        env = self._env
        gs = env._game_state
        pp, bp = gs.player_pose, gs.boss_pose
        return RawSnapshot(
            player_hp=int(gs.player_hp),
            player_hp_max=int(gs.player_max_hp),
            player_sp=int(gs.player_sp),
            player_sp_max=int(gs.player_max_sp),
            player_anim=str(gs.player_animation),
            player_anim_time_s=float(gs.player_animation_duration),
            player_pos=(float(pp[0]), float(pp[1]), float(pp[2])),
            player_heading=float(pp[3]),
            locked_on=bool(gs.lock_on),
            boss_hp=int(gs.boss_hp),
            boss_hp_max=ep.boss_hp_max,
            boss_phase=ep.boss_phase,
            boss_anim=str(gs.boss_animation),
            boss_anim_time_s=float(gs.boss_animation_duration),
            boss_pos=(float(bp[0]), float(bp[1]), float(bp[2])),
            boss_heading=float(bp[3]),
            **env.extra.read_all(),
        )
