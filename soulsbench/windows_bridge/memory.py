"""Read-only DS3 memory values SoulsGym does not expose: estus, poise, i-frame and parry flags.

Pointer chains come from The Grand Archives DS3 cheat table that ships with SoulsGym
(``ce_tables/DS3_TGA.CT``), for game version 1.15.2. Cheat Engine lists offsets innermost-last;
SoulsGym address records list them in chain order, which is what is written here.

Nothing in this module writes to game memory. Every read degrades to ``None`` on failure, so a wrong
offset costs a state field, never a crash or a corrupted save.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# [[[[WorldChrMan]+80]+1F90]+0]+58: bit 1 "iFrames (Read Only)", bit 2 "Parry (Read Only)".
PLAYER_CHR_FLAGS = {
    "base": "WorldChrMan",
    "offsets": [0x80, 0x1F90, 0x0, 0x58],
    "type": "bytes",
    "length": 1,
}
IFRAMES_BIT = 1
PARRY_BIT = 2
# Super armor module: "SADurability" current / max (floats).
PLAYER_POISE = {"base": "WorldChrMan", "offsets": [0x80, 0x1F90, 0x40, 0x28], "type": "float"}
PLAYER_POISE_MAX = {"base": "WorldChrMan", "offsets": [0x80, 0x1F90, 0x40, 0x2C], "type": "float"}
# [[GameDataMan]+10] is PlayerGameData.
PLAYER_GAME_DATA = {"base": "GameDataMan", "offsets": [0x10, 0x0], "type": "int"}
ESTUS_MAX_OFFSET = 0x102  # byte, "Estus Flask Max"
INVENTORY_PTR_OFFSET = 0x3E8  # -> array of 0x10-byte entries: handle u32, item id u32, qty u8 @+8
INVENTORY_NUM_OFFSET = 0x3F0
INVENTORY_TAIL_OFFSET = 0x3F4
ENTRY_SIZE = 0x10
MAX_SCAN_ENTRIES = 4096
# Goods 0x96..0xAB: "Estus Flask" +0..+10, two ids per level (charged / empty variants).
ESTUS_ITEM_IDS = frozenset(0x40000000 | g for g in range(0x96, 0xAC))


class ExtraMemory:
    """Extra reads against a soulsgym ``DarkSoulsIII`` game object."""

    def __init__(self, game: Any) -> None:
        self.game = game
        self.mem = game.mem
        self._estus_entry: int | None = None
        self._failed: set[str] = set()

    def invalidate(self) -> None:
        """Forget cached addresses (call after a game reload)."""
        self._estus_entry = None

    def _safe(self, key: str, fn, default=None):
        try:
            return fn()
        except Exception as e:  # any read failure is a missing field, never a crash
            if key not in self._failed:
                self._failed.add(key)
                logger.warning(
                    "memory read %r failed (%s: %s); reporting None", key, type(e).__name__, e
                )
            return default

    # --- player -----------------------------------------------------------------------------------

    def player_flags(self) -> tuple[bool | None, bool | None]:
        raw = self._safe("player_flags", lambda: self.mem.read_record(PLAYER_CHR_FLAGS))
        if raw is None:
            return None, None
        return bool(raw[0] >> IFRAMES_BIT & 1), bool(raw[0] >> PARRY_BIT & 1)

    def player_poise(self) -> tuple[float | None, float | None]:
        cur = self._safe("poise", lambda: self.mem.read_record(PLAYER_POISE))
        cap = self._safe("poise_max", lambda: self.mem.read_record(PLAYER_POISE_MAX))
        # Live on 1.15.2 this chain read 0.0 / 0.0 on every tick for the Knight, i.e. it is not
        # the player's poise. Report "unknown" rather than a misleading zero.
        if cap is None or not 0.0 < cap < 1e5 or cur is None or not -1.0 <= cur < 1e5:
            return None, None
        return cur, cap

    # --- estus ------------------------------------------------------------------------------------

    def _player_game_data(self) -> int:
        return self.mem.resolve_record(PLAYER_GAME_DATA)

    def _scan(self, start: int, count: int) -> int | None:
        for chunk_start in range(0, count, 256):
            n = min(256, count - chunk_start)
            base = start + chunk_start * ENTRY_SIZE
            try:
                block = self.mem.read_bytes(base, n * ENTRY_SIZE)
            except Exception:
                return None
            for i in range(n):
                entry = block[i * ENTRY_SIZE : (i + 1) * ENTRY_SIZE]
                if int.from_bytes(entry[4:8], "little") in ESTUS_ITEM_IDS and entry[8] <= 99:
                    return base + i * ENTRY_SIZE
        return None

    def find_estus_entry(self) -> int | None:
        pgd = self._player_game_data()
        arr = int.from_bytes(self.mem.read_bytes(pgd + INVENTORY_PTR_OFFSET, 8), "little")
        num = self.mem.read_int(pgd + INVENTORY_NUM_OFFSET)
        tail = self.mem.read_int(pgd + INVENTORY_TAIL_OFFSET)
        count = min(max(num, 0) + 64, MAX_SCAN_ENTRIES)
        starts = [arr]
        if 0 < tail < MAX_SCAN_ENTRIES:  # the CE table indexes entries as arr + (idx - tail) * 0x10
            starts.append(arr - tail * ENTRY_SIZE)
            count = min(count + tail, MAX_SCAN_ENTRIES)
        for start in starts:
            found = self._scan(start, count)
            if found is not None:
                return found
        return None

    def _read_estus(self) -> tuple[int | None, int | None]:
        if self._estus_entry is None:
            self._estus_entry = self.find_estus_entry()
            if self._estus_entry is None:
                raise LookupError("no Estus Flask entry found in the inventory")
        entry = self.mem.read_bytes(self._estus_entry, ENTRY_SIZE)
        if int.from_bytes(entry[4:8], "little") not in ESTUS_ITEM_IDS:
            self._estus_entry = None
            raise LookupError("cached estus inventory entry moved")
        count = entry[8]
        cap = self.mem.read_bytes(self._player_game_data() + ESTUS_MAX_OFFSET, 1)[0]
        return count, cap if 0 < cap <= 20 else None

    def estus(self) -> tuple[int | None, int | None]:
        return self._safe("estus", self._read_estus, default=(None, None))

    # --- everything for one snapshot --------------------------------------------------------------

    def read_all(self) -> dict[str, Any]:
        game = self.game
        iframes, parry = self.player_flags()
        poise, poise_max = self.player_poise()
        estus, estus_max = self.estus()
        return {
            "player_anim_mem_time_s": self._safe(
                "player_anim_time", lambda: float(game.player_animation_time)
            ),
            "player_anim_max_s": self._safe(
                "player_anim_max", lambda: float(game.player_animation_max_time)
            ),
            "player_poise": poise,
            "player_poise_max": poise_max,
            "player_iframes": iframes,
            "player_parry_window": parry,
            "estus": estus,
            "estus_max": estus_max,
            "boss_anim_raw": self._safe(
                "boss_anim_raw",
                lambda: str(self.mem.read_record(game.data.addresses["IudexAnimation"])),
            ),
            "boss_anim_mem_time_s": self._safe(
                "boss_anim_time", lambda: float(game.iudex_animation_time)
            ),
            "boss_anim_max_s": self._safe(
                "boss_anim_max", lambda: float(game.iudex_animation_max_time)
            ),
        }

    def probe(self) -> dict[str, Any]:
        """Diagnostic dump for ``python -m soulsbench.windows_bridge probe``."""
        out: dict[str, Any] = {}
        try:
            pgd = self._player_game_data()
            out["player_game_data"] = hex(pgd)
            out["inventory_ptr"] = hex(
                int.from_bytes(self.mem.read_bytes(pgd + INVENTORY_PTR_OFFSET, 8), "little")
            )
            out["inventory_num"] = self.mem.read_int(pgd + INVENTORY_NUM_OFFSET)
            out["inventory_tail"] = self.mem.read_int(pgd + INVENTORY_TAIL_OFFSET)
            out["estus_max_byte"] = self.mem.read_bytes(pgd + ESTUS_MAX_OFFSET, 2).hex()
            entry = self.find_estus_entry()
            out["estus_entry"] = hex(entry) if entry else None
            if entry:
                out["estus_entry_bytes"] = self.mem.read_bytes(entry, ENTRY_SIZE).hex()
        except Exception as e:
            out["inventory_error"] = f"{type(e).__name__}: {e}"
        out.update(self.read_all())
        return out
