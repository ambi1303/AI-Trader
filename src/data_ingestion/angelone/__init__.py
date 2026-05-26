"""Angel One SmartAPI integration -- READ ONLY.

This package consumes Angel One's REST API for live LTPs and EOD candles.
It does NOT place orders. The decision is structural, not configurable:
no module here imports any order-placement endpoint and the session
helper sets ``allow_orders=False`` until ``ANGEL_ALLOW_ORDERS=true`` is
explicitly set. Adding live execution later requires a deliberate code
change AND that env flag, so accidental activation is hard.

Why direct REST and not the official ``smartapi-python`` package?
* We only need 4 endpoints (login, instrument master, candles, LTP).
* ``requests`` + ``pyotp`` are already in our dependency tree.
* Direct calls are trivially mockable in tests, the official client is not.
* Smaller dep surface = smaller supply-chain risk.

Public surface
--------------
* :class:`AngelOneSession` -- TOTP-based login, JWT lifecycle, redacted logs
* :func:`download_instrument_master` / :func:`resolve_token`
* :func:`fetch_daily_candles` -- backfills price_data with source='angelone'
* :func:`fetch_ltp` -- live LTP for one or many instruments (used by the
  data-freshness check in the daily report)

All public functions accept an optional ``session`` argument so callers
can re-use one login per orchestration pass.
"""

from src.data_ingestion.angelone.historical import fetch_daily_candles
from src.data_ingestion.angelone.instrument_master import (
    download_instrument_master,
    resolve_token,
)
from src.data_ingestion.angelone.quote import fetch_ltp
from src.data_ingestion.angelone.session import (
    AngelOneAuthError,
    AngelOneSession,
    load_session_from_env,
)

__all__ = [
    "AngelOneAuthError",
    "AngelOneSession",
    "download_instrument_master",
    "fetch_daily_candles",
    "fetch_ltp",
    "load_session_from_env",
    "resolve_token",
]
