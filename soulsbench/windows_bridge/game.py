"""Windows-side helpers: find/launch/attach Dark Souls III, back up saves, pick bind addresses.

Standard library only at import time; psutil (a soulsgym dependency) and the Windows-only modules
are imported inside the functions that need them.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROCESS_NAME = "DarkSoulsIII.exe"
APP_ID = 374320
EXPECTED_GAME_VERSION = "1.15.2.0"


def steam_exe() -> Path | None:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
            return Path(winreg.QueryValueEx(key, "SteamExe")[0])
    except OSError:
        return None


def ds3_install_dir() -> Path | None:
    exe = steam_exe()
    if exe is None:
        return None
    vdf = exe.parent / "steamapps" / "libraryfolders.vdf"
    libraries = [exe.parent]
    if vdf.exists():
        libraries += [
            Path(p.replace("\\\\", "\\"))
            for p in re.findall(r'"path"\s+"([^"]+)"', vdf.read_text())
        ]
    for lib in libraries:
        game = lib / "steamapps" / "common" / "DARK SOULS III" / "Game"
        if (game / PROCESS_NAME).exists():
            return game
    return None


def file_version(path: Path) -> str | None:
    try:
        import win32api

        info = win32api.GetFileVersionInfo(str(path), "\\")
        ms, ls = info["FileVersionMS"], info["FileVersionLS"]
        return f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}"
    except Exception:
        return None


def ds3_pid() -> int | None:
    import psutil

    for proc in psutil.process_iter(["name"]):
        if proc.info["name"] == PROCESS_NAME:
            return proc.pid
    return None


def launch_ds3() -> None:
    exe = steam_exe()
    if exe is not None and exe.exists():
        subprocess.Popen([str(exe), "-applaunch", str(APP_ID)])
    else:
        os.startfile(f"steam://rungameid/{APP_ID}")  # type: ignore[attr-defined]


SPEEDHACK_PORT = 51285  # SoulsGym's speedhack DLL control port (soulsgym.core.speedhack)


def speedhack_loaded() -> bool:
    import socket

    try:
        socket.create_connection(("127.0.0.1", SPEEDHACK_PORT), timeout=0.5).close()
        return True
    except OSError:
        return False


def ensure_ds3_running(launch: bool, timeout_s: float = 180.0) -> int:
    pid = ds3_pid()
    if pid is not None:
        # Injecting the speedhack into a game that is already past startup crashed DS3 twice
        # (0xc0000005 at address 0, 2026-10-05); injection while it boots has always worked.
        if not speedhack_loaded():
            from soulsbench.windows_bridge.errors import BridgeRefused

            raise BridgeRefused(
                f"{PROCESS_NAME} (pid {pid}) was started without the bridge, and injecting the "
                "speedhack into a running game crashes it. Quit DS3 and start the bridge again: "
                "it launches the game itself."
            )
        logger.info("Attached to running %s (pid %d)", PROCESS_NAME, pid)
        return pid
    if not launch:
        raise RuntimeError(f"{PROCESS_NAME} is not running and --no-launch was given")
    logger.info("Launching Dark Souls III through Steam (app %d)...", APP_ID)
    launch_ds3()
    deadline = time.monotonic() + timeout_s
    while (pid := ds3_pid()) is None:
        if time.monotonic() > deadline:
            raise RuntimeError(f"{PROCESS_NAME} did not start within {timeout_s:.0f} s")
        time.sleep(2.0)
    logger.info("%s started (pid %d)", PROCESS_NAME, pid)
    return pid


def save_dir() -> Path:
    return Path(os.environ["APPDATA"]) / "DarkSoulsIII"


def backup_saves(dest_root: Path) -> Path | None:
    """Copy every DS3 save and the graphics config to dest_root/<timestamp>/ before touching anything."""
    src = save_dir()
    if not src.exists():
        return None
    dest = dest_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.copytree(src, dest)
    return dest


def graphics_config() -> dict[str, str]:
    path = save_dir() / "GraphicsConfig.xml"
    if not path.exists():
        return {}
    text = path.read_bytes().decode("utf-16", errors="replace")
    return dict(re.findall(r"<([A-Za-z-]+)>([^<]*)</\1>", text))


def wsl_host_addresses() -> list[str]:
    """IPv4 addresses of the Hyper-V 'vEthernet (WSL...)' adapters (reachable from WSL2 NAT)."""
    import socket

    import psutil

    out = []
    for name, addrs in psutil.net_if_addrs().items():
        if "WSL" in name:
            out += [a.address for a in addrs if a.family == socket.AF_INET]
    return out


def firewall_actions(program: str) -> list[str] | None:
    """Actions (Allow/Block) of enabled inbound firewall rules for ``program``; None if unknown."""
    script = (
        f"$p = '{program}'; "
        "Get-NetFirewallApplicationFilter | Where-Object { $_.Program -eq $p } | Get-NetFirewallRule | "
        "Where-Object { $_.Enabled -eq 'True' -and $_.Direction -eq 'Inbound' } | "
        "ForEach-Object { [string]$_.Action }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.split() if out.returncode == 0 else None


def keyboard_layouts() -> list[str]:
    try:
        import win32api

        return [f"{hkl & 0xFFFF:04x}" for hkl in win32api.GetKeyboardLayoutList()]
    except Exception:
        return []
