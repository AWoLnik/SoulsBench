"""Windows bridge CLI. Run from a *Windows* Python, next to the game, never from WSL.

python -m soulsbench.windows_bridge doctor     check Python, soulsgym, DS3 version and settings
python -m soulsbench.windows_bridge serve      launch/attach DS3 and serve CombatState over HTTP
python -m soulsbench.windows_bridge probe      one read-only dump of the extra memory values
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
from pathlib import Path

from soulsbench import __version__

DEFAULT_PORT = 47800
PREFIX = Path(os.environ.get("LOCALAPPDATA", ".")) / "soulsbench"
BACKUP_ROOT = PREFIX / "save_backups"
LOG_FILE = PREFIX / "bridge.log"  # readable from WSL: /mnt/c/Users/<you>/AppData/Local/soulsbench
RECOMMENDED_PYTHON = (3, 12)


def _require_windows() -> None:
    if platform.system() != "Windows":
        sys.exit(
            "The SoulsBench bridge must run in a native Windows Python next to Dark Souls III, not in "
            "WSL. See README: windows\\setup_bridge.ps1 and windows\\run_bridge.ps1."
        )


def cmd_doctor(args: argparse.Namespace) -> int:
    from soulsbench.windows_bridge import game as g

    ok = True

    def line(good: bool | None, label: str, value: object) -> None:
        nonlocal ok
        mark = {True: "ok  ", False: "FAIL", None: "warn"}[good]
        ok &= good is not False
        print(f"[{mark}] {label}: {value}")

    v = sys.version_info
    line(v >= (3, 11, 2), "python", f"{platform.python_version()} ({sys.executable})")
    if v[:2] != RECOMMENDED_PYTHON:
        line(None, "python", "tested with CPython 3.12.x; other versions are untested")
    try:
        from importlib.metadata import version

        line(True, "soulsgym", version("soulsgym"))
        import soulsgym  # noqa: F401
    except Exception as e:
        line(False, "soulsgym", f"{type(e).__name__}: {e}")
    install = g.ds3_install_dir()
    line(install is not None, "DS3 install", install)
    if install is not None:
        ver = g.file_version(install / g.PROCESS_NAME)
        line(
            ver == g.EXPECTED_GAME_VERSION,
            "DS3 version",
            f"{ver} (SoulsGym needs {g.EXPECTED_GAME_VERSION})",
        )
    cfg = g.graphics_config()
    mode = cfg.get("ScreenMode")
    line(mode == "WINDOW", "screen mode", f"{mode} (must be WINDOW)")
    res = f"{cfg.get('Resolution-WindowScreenWidth')}x{cfg.get('Resolution-WindowScreenHeight')}"
    line(True, "window resolution", f"{res} (any windowed size works; no screen capture)")
    layouts = g.keyboard_layouts()
    line(
        None if len(layouts) != 1 else True,
        "keyboard layouts",
        f"{layouts} (SoulsGym wants a single en-US layout)",
    )
    line(
        bool(g.wsl_host_addresses()) or None,
        "WSL host address",
        g.wsl_host_addresses() or "none (mirrored networking?)",
    )
    actions = g.firewall_actions(sys._base_executable)
    if actions and "Allow" in actions and "Block" not in actions:
        line(True, "firewall", "allowed: direct mode (--host <WSL adapter IP>) available")
    else:
        line(True, "firewall", "no rule needed: WSL reaches 127.0.0.1 through the interop relay")
    pid = g.ds3_pid()
    line(True if pid else None, "DS3 running", pid or "no (serve will launch it)")
    print(
        "\nIn-game checklist (not machine-checkable): Play Offline, dedicated Knight save, key bindings"
        " from the README, Estus Flask in the active quick-item slot."
    )
    return 0 if ok else 1


def cmd_serve(args: argparse.Namespace) -> int:
    from soulsbench.windows_bridge import game as g
    from soulsbench.windows_bridge.errors import BridgeRefused
    from soulsbench.windows_bridge.live_env import LiveBridgeEnv
    from soulsbench.windows_bridge.server import BridgeServer

    log = logging.getLogger("soulsbench.bridge")
    # Loopback only by default: never exposed, never triggers a firewall prompt. WSL reaches it
    # through an interop relay (soulsbench.backends.relay); --host adds direct bind addresses.
    hosts = ["127.0.0.1", *(args.host or [])]
    backend = LiveBridgeEnv(
        game_speed=args.game_speed,
        allow_any_save=args.allow_any_save,
        **({"attack_windows_path": Path(args.attack_windows)} if args.attack_windows else {}),
    )
    server = BridgeServer(backend, hosts, args.port)
    server.start()
    log.info("SoulsBench bridge %s listening on %s", __version__, ", ".join(server.addresses))
    print(
        "\n  !! Keep DS3 offline (System > Network > Play Offline) and use a throwaway Knight save."
    )
    print("  !! While episodes run, the bridge owns the keyboard and the DS3 window: hands off.\n")
    stop = server.shutdown_requested  # set by POST /shutdown (python -m soulsbench.bridge stop)
    try:
        g.ensure_ds3_running(launch=not args.no_launch)
        if not args.no_backup:
            dest = g.backup_saves(BACKUP_ROOT)
            log.info("Backed up DS3 saves to %s", dest)
        log.info(
            "Waiting for the player to load into the game (Load Game on your SoulsBench Knight)..."
        )
        backend.start(stop)
        if not stop.is_set():
            log.info(
                "Ready. From WSL: python -m soulsbench.run --backend souls_gym --scenario gundyr_p1 ..."
            )
        while not stop.wait(1.0):
            pass
        log.info("Shutdown requested over HTTP")
    except KeyboardInterrupt:
        log.info("Ctrl+C")
    except Exception as e:
        log.error("%s: %s", type(e).__name__, e)
        if isinstance(e, BridgeRefused) and not backend.status.startswith("refused"):
            backend.status = f"refused: {e}"  # refused before the env existed (e.g. at attach)
        if not backend.status.startswith("refused"):
            return 1
        log.error(
            "The bridge stays up so WSL can see why; Ctrl+C or `soulsbench.bridge stop` exits."
        )
        try:
            while not stop.wait(1.0):
                pass
        except KeyboardInterrupt:
            pass
    finally:
        log.info("Restoring game flags and resuming the game (SoulsGym close)")
        backend.shutdown()
        server.stop()
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    from soulsgym.games import DarkSoulsIII

    from soulsbench.windows_bridge.live_env import install_focus_only_window
    from soulsbench.windows_bridge.memory import ExtraMemory

    install_focus_only_window()
    game = DarkSoulsIII()
    if not game.is_ingame:
        print("player is not in-game")
        return 1
    out = {
        "player_hp": game.player_hp,
        "player_max_hp": game.player_max_hp,
        "player_sp": game.player_sp,
        "player_stats": list(game.player_stats),
        "player_animation": game.player_animation,
        "player_pose": [round(float(x), 3) for x in game.player_pose],
        "lock_on": game.lock_on,
        **ExtraMemory(game).probe(),
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m soulsbench.windows_bridge",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check the Windows side")
    serve = sub.add_parser("serve", help="run the bridge")
    serve.add_argument(
        "--port", type=int, default=int(os.environ.get("SOULSBENCH_BRIDGE_PORT", DEFAULT_PORT))
    )
    serve.add_argument(
        "--host",
        action="append",
        help="extra bind address besides 127.0.0.1 (repeatable), e.g. the WSL adapter IP for "
        "direct mode, which needs windows\\allow_wsl_firewall.ps1",
    )
    serve.add_argument(
        "--game-speed",
        type=float,
        default=1.0,
        help="SoulsGym game speed while stepping (1.0 = real time)",
    )
    serve.add_argument("--no-launch", action="store_true", help="attach only; never start DS3")
    serve.add_argument("--no-backup", action="store_true", help="skip the save backup on start")
    serve.add_argument(
        "--allow-any-save", action="store_true", help="skip the fresh-Knight save check (DANGEROUS)"
    )
    serve.add_argument(
        "--attack-windows", help="calibrated attack windows JSON (default: bundled file)"
    )
    sub.add_parser("probe", help="dump extra memory reads once (read-only)")
    args = parser.parse_args(argv)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.cmd == "serve":
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(LOG_FILE, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    _require_windows()
    try:
        return {"doctor": cmd_doctor, "serve": cmd_serve, "probe": cmd_probe}[args.cmd](args)
    except Exception:
        logging.getLogger("soulsbench.bridge").exception("bridge %s crashed", args.cmd)
        raise


if __name__ == "__main__":
    sys.exit(main())
