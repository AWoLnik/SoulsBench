"""Per-episode OBS recordings over obs-websocket v5 (OBS 28+, Tools -> WebSocket Server Settings).

The bridge runs next to OBS on Windows, so it drives it: start a recording when an episode
starts, stop it when the episode ends, then move the file to ``<folder>/<name>_<outcome>.<ext>``.
A text source named ``SoulsBench`` in the current scene, if there is one, shows the episode label
(model and seed) on screen. Recording is best-effort: any OBS problem is reported to the caller,
which carries on without video.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

OVERLAY_SOURCE = "SoulsBench"


class RecorderError(RuntimeError):
    pass


class ObsRecorder:
    def __init__(self, host: str = "localhost", port: int = 4455) -> None:
        self.host, self.port = host, port
        self._client = None
        self._password: str | None = None
        self._pending: tuple[Path, str] | None = None  # (folder, name) of the running recording

    def _connect(self, password: str):
        if self._client is not None and password == self._password:
            return self._client
        try:
            import obsws_python as obs
        except ImportError as e:
            raise RecorderError("obsws-python is not installed in the bridge venv") from e
        try:
            self._client = obs.ReqClient(
                host=self.host, port=self.port, password=password, timeout=5
            )
        except Exception as e:  # noqa: BLE001 - connection refused, bad password, ...
            self._client = None
            raise RecorderError(f"cannot reach OBS on {self.host}:{self.port}: {e}") from e
        self._password = password
        return self._client

    def start(self, name: str, label: str, folder: str, password: str) -> None:
        client = self._connect(password)
        try:
            if client.get_record_status().output_active:
                client.stop_record()  # a recording left over from an aborted episode
            inputs = {i["inputName"] for i in client.get_input_list().inputs}
            if OVERLAY_SOURCE in inputs:
                client.set_input_settings(OVERLAY_SOURCE, {"text": label}, overlay=True)
            client.start_record()
        except Exception as e:  # noqa: BLE001
            self._client = None
            raise RecorderError(f"OBS start_record failed: {e}") from e
        self._pending = (Path(folder), name)
        logger.info("Recording %s", name)

    def stop(self, suffix: str) -> str | None:
        if self._pending is None or self._client is None:
            return None
        folder, name = self._pending
        self._pending = None
        try:
            src = Path(self._client.stop_record().output_path)
        except Exception as e:  # noqa: BLE001
            self._client = None
            raise RecorderError(f"OBS stop_record failed: {e}") from e
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / f"{name}_{suffix}{src.suffix}"
        for _ in range(50):  # OBS finalizes the file for a moment after stopping
            try:
                os.replace(src, dest)
                break
            except OSError:
                time.sleep(0.2)
        else:
            raise RecorderError(f"could not move {src} to {dest}")
        logger.info("Saved recording %s", dest)
        return str(dest)
