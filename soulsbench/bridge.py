"""Control the Windows bridge from WSL.

    python -m soulsbench.bridge status              is the bridge up and ready? (auto-discovered)
    python -m soulsbench.bridge start [-- args]     open the bridge in a new *Windows* console window
    python -m soulsbench.bridge stop                restore the game (SoulsGym close) and exit

``start`` runs windows/run_bridge.ps1 in its own PowerShell window on Windows, i.e. a native Windows
process with its own console (Ctrl+C there works too). Arguments after ``--`` go to ``serve``. The
window closes when the bridge exits; everything it logged is in %LOCALAPPDATA%\\soulsbench\\bridge.log.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import httpx

from soulsbench.backends.souls_gym import BridgeUnavailable, connect_bridge

REPO = Path(__file__).resolve().parent.parent


def windows_path(path: Path) -> str:
    out = subprocess.run(["wslpath", "-w", str(path)], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _quote(arg: str) -> str:
    return "'" + arg.replace("'", "''") + "'"


def cmd_status(args: argparse.Namespace) -> int:
    conn = connect_bridge(args.bridge_url)
    try:
        health = httpx.get(f"{conn.url}/health", timeout=5.0).json()
    finally:
        conn.close()
    print(f"bridge reachable via {conn.via}")
    print(json.dumps(health, indent=2))
    return 0 if health.get("ready") else 3


def cmd_start(args: argparse.Namespace) -> int:
    script = windows_path(REPO / "windows" / "run_bridge.ps1")
    extra = [a for a in args.serve_args if a != "--"]
    ps_args = ["-ExecutionPolicy", "Bypass", "-File", script, *extra]
    command = "Start-Process powershell -ArgumentList " + ",".join(map(_quote, ps_args))
    subprocess.run(["powershell.exe", "-NoProfile", "-Command", command], check=True)
    print("Bridge console opened on Windows. Load your SoulsBench Knight, then poll with:")
    print("  python -m soulsbench.bridge status")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    conn = connect_bridge(args.bridge_url)
    try:
        httpx.post(f"{conn.url}/shutdown", json={}, timeout=5.0).raise_for_status()
    finally:
        conn.close()
    print(f"Shutdown requested (via {conn.via}): the bridge restores the game and exits.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m soulsbench.bridge",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bridge-url", help="default: $SOULSBENCH_BRIDGE_URL, else auto-detect")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    start = sub.add_parser("start")
    start.add_argument("serve_args", nargs=argparse.REMAINDER)
    sub.add_parser("stop")
    args = parser.parse_args(argv)
    try:
        return {"status": cmd_status, "start": cmd_start, "stop": cmd_stop}[args.cmd](args)
    except BridgeUnavailable as e:
        print(e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
