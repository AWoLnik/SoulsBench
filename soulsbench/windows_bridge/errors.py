"""Bridge errors that map to specific HTTP statuses."""


class BridgeNotReady(RuntimeError):
    """503: game not attached / player not in-game / env not built yet."""


class BridgeStateError(RuntimeError):
    """409: request out of order, e.g. /step without a running episode."""


class BridgeRefused(RuntimeError):
    """The bridge will not drive this game: the setup is unsafe or unsupported."""


class UnsafeSaveError(BridgeRefused):
    """The loaded character is not a disposable SoulsBench save; the bridge refuses to modify it."""


class WindowModeError(BridgeRefused):
    """DS3 is fullscreen. SoulsGym's focus click lands inside the game unless it runs windowed."""


class FocusError(RuntimeError):
    """DS3 could not be brought to the foreground, so no keystrokes were sent."""
