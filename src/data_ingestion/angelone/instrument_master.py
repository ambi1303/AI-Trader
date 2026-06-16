"""Angel One instrument master.

Angel One publishes its full instrument list (NSE, BSE, NFO, MCX, ...) as
a single ~30MB JSON file at a public, unauthenticated URL. We download
it once a day, persist a slim NSE-equity-only slice to disk, and keep
an in-memory ``symbol -> token`` map for fast lookups.

Why we don't put this in the database:
* The full file is ~50k rows; we only need the few hundred NSE-EQ entries
  we trade and we re-download every trading day to capture renames.
* Keeping it on disk as JSON is auditable -- you can grep last week's
  cache to verify a token mapping if a price diff shows up.

Security:
* The endpoint is HTTPS-only, public, and well-known -- we still call it
  with ``verify=True`` and a short timeout.
* The downloaded file is parsed with the stdlib ``json`` module which is
  safe-by-default (no code execution path).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import requests

from src.utils.logger import get_logger
from src.utils.secrets import project_root

log = get_logger("angelone.instruments")

# Angel One's public instrument master. Updated daily ~8:30 AM IST.
_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/"
    "OpenAPIScripMaster.json"
)

# Per Angel One docs and observed behaviour, NSE cash-equity rows look like:
#   {"token": "2885", "symbol": "RELIANCE-EQ", "name": "RELIANCE",
#    "exch_seg": "NSE", "instrumenttype": "", "lotsize": "1", ...}
_TARGET_EXCHANGE = "NSE"
_TARGET_SUFFIX = "-EQ"


@dataclass(frozen=True)
class Instrument:
    symbol: str          # canonical e.g. "RELIANCE"
    angel_symbol: str    # raw from API e.g. "RELIANCE-EQ"
    token: str
    exchange: str        # "NSE"
    lot_size: int

    @property
    def yfinance_symbol(self) -> str:
        return f"{self.symbol}.NS"


def _cache_dir() -> Path:
    p = project_root() / "data" / "cache" / "angelone_instruments"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _today_cache_path() -> Path:
    return _cache_dir() / f"nse_eq_{date.today().isoformat()}.json"


def download_instrument_master(
    *,
    timeout: int = 30,
    force: bool = False,
) -> list[Instrument]:
    """Return the NSE equity instrument list, refreshing once per day.

    The slim cache file (NSE-EQ rows only) is small (~few hundred KB) and
    lives under ``data/cache/angelone_instruments/``. Pass ``force=True``
    to re-download mid-day after a corporate action.
    """
    cache_path = _today_cache_path()
    if cache_path.exists() and not force:
        try:
            return _load_cache(cache_path)
        except json.JSONDecodeError:
            log.warning("Instrument master cache corrupt; re-downloading")
            cache_path.unlink(missing_ok=True)

    log.info("Downloading Angel One instrument master ({})", _MASTER_URL)
    try:
        resp = requests.get(_MASTER_URL, timeout=timeout, verify=True)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Failed to download Angel One instrument master. The endpoint "
            "is public so this is usually a network issue."
        ) from exc

    raw = resp.json()
    if not isinstance(raw, list):
        raise RuntimeError(
            "Unexpected instrument master shape (expected list of dicts)."
        )

    instruments: list[Instrument] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        if row.get("exch_seg") != _TARGET_EXCHANGE:
            continue
        sym = (row.get("symbol") or "").upper()
        if not sym.endswith(_TARGET_SUFFIX):
            continue
        instruments.append(Instrument(
            symbol=sym[:-len(_TARGET_SUFFIX)],
            angel_symbol=sym,
            token=str(row.get("token") or ""),
            exchange=_TARGET_EXCHANGE,
            lot_size=int(row.get("lotsize") or 1),
        ))
    log.info("Cached {} NSE equity instruments to {}",
             len(instruments), cache_path.as_posix())
    _save_cache(cache_path, instruments)
    return instruments


def _save_cache(path: Path, instruments: list[Instrument]) -> None:
    payload = [
        {
            "symbol": i.symbol,
            "angel_symbol": i.angel_symbol,
            "token": i.token,
            "exchange": i.exchange,
            "lot_size": i.lot_size,
        }
        for i in instruments
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_cache(path: Path) -> list[Instrument]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Instrument(
            symbol=row["symbol"],
            angel_symbol=row["angel_symbol"],
            token=row["token"],
            exchange=row["exchange"],
            lot_size=int(row.get("lot_size") or 1),
        )
        for row in raw
    ]


def resolve_token(
    symbol: str,
    instruments: list[Instrument] | None = None,
) -> Instrument | None:
    """Find an :class:`Instrument` by canonical symbol (e.g. ``RELIANCE``)."""
    sym = symbol.upper().strip()
    insts = instruments if instruments is not None else download_instrument_master()
    for i in insts:
        if i.symbol == sym:
            return i
    return None
