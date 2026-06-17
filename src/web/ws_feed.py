"""Angel One SmartStream WebSocket 2.0 client -- live LTP feed for the dashboard.

Why this exists
---------------
The dashboard used to poll Angel One's REST ``getLtpData`` every ~10s. REST is
a request/response *snapshot* endpoint and is explicitly "not for live updates"
per Angel One's own docs -- so the displayed price trailed the real market by
minutes. SmartStream WS 2.0 is the only Angel One path that pushes genuine
tick-by-tick prices. This module keeps one persistent WS connection, subscribes
to symbols on demand (LTP mode), and maintains an in-memory ``token -> tick``
cache that the web layer reads instead of hitting REST.

Design
------
* **Lazy, singleton, thread-safe.** A single :class:`LiveFeed` is started on the
  first live quote during market hours (mirrors the existing lazy REST session).
* **Subscribe-on-demand.** We only subscribe to tokens the dashboard actually
  views (top candidates, open positions, the stock page you open). Tokens
  accumulate over a session but stay far below Angel One's 1000-token/mode cap.
* **No regression.** If the WS is still connecting, disconnected, or has no tick
  for a symbol yet, the caller falls back to the existing REST->EOD path. The
  stream is strictly an *upgrade* layered on top.
* **Self-healing.** ``run_forever`` reconnects with backoff; on every (re)open we
  re-subscribe the full known token set, and an app-level ``ping`` heartbeat
  keeps the socket from idling out.

Security
--------
* Credentials (JWT auth_token, feed_token, api_key, client_code) come from
  :class:`AngelOneSession` and are passed only as WS handshake headers. They are
  never logged: log lines reference counts and token ids, never secrets.
* TLS is enforced by the ``wss://`` scheme; we never disable certificate
  verification.
* This is a READ-ONLY market-data stream. It cannot place orders.
"""

from __future__ import annotations

import json
import struct
import threading
import time
import uuid
from typing import Any

from src.utils.logger import get_logger

log = get_logger("web.wsfeed")

# Endpoint + protocol constants (Angel One SmartStream WS 2.0).
_WS_URL = "wss://smartapisocket.angelone.in/smart-stream"
_ACTION_SUBSCRIBE = 1
_MODE_LTP = 1
_EXCHANGE_NSE_CM = 1            # all our instruments are NSE cash equities
_HEARTBEAT_S = 25.0            # send "ping" well under the server idle timeout
_LTP_PACKET_LEN = 51          # LTP-mode binary packet size
_TICK_FRESH_S = 30.0          # a tick older than this is treated as stale

# Angel One sends prices in paise for equities -> rupees = value / 100.
_PAISE_PER_RUPEE = 100.0


class LiveFeed:
    """Maintains one SmartStream WS connection and a live ``token -> tick`` map."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # token -> (ltp_rupees, feed_epoch_seconds)
        self._ticks: dict[str, tuple[float, float]] = {}
        # set of tokens we want subscribed (re-sent on every reconnect)
        self._want: set[str] = set()
        # tokens already sent to the server in the current connection
        self._sent: set[str] = set()
        self._symbol_to_token: dict[str, str] = {}
        self._token_to_symbol: dict[str, str] = {}

        self._ws: Any = None
        self._connected = False
        self._started = False
        self._stop = threading.Event()
        self._creds: dict[str, str] | None = None

    # -- Lifecycle ----------------------------------------------------------

    def start(self, creds: dict[str, str], instruments: list[Any]) -> bool:
        """Start the background WS thread once. Returns True if the feed is
        running (or was already), False if it could not start.

        ``creds`` is the dict from ``AngelOneSession.websocket_credentials()``.
        ``instruments`` is the Angel One instrument master (for symbol->token).
        """
        with self._lock:
            # (Re)build the symbol/token maps every call -- cheap and keeps us
            # current if the instrument master was refreshed mid-day.
            for inst in instruments:
                tok = str(getattr(inst, "token", "") or "")
                sym = str(getattr(inst, "symbol", "") or "").upper()
                if tok and sym:
                    self._symbol_to_token[sym] = tok
                    self._token_to_symbol[tok] = sym

            if self._started:
                return True

            try:
                import websocket  # noqa: F401  -- ensure the dep is present
            except Exception as exc:  # noqa: BLE001
                log.warning("websocket-client not available; live stream disabled: {}", exc)
                return False

            self._creds = dict(creds)
            self._stop.clear()
            self._started = True

            t = threading.Thread(target=self._run_forever, name="smartstream-ws",
                                 daemon=True)
            t.start()
            log.info("SmartStream WS feed thread started")
            return True

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass

    # -- Subscription -------------------------------------------------------

    def ensure_subscribed(self, symbols: list[str]) -> None:
        """Make sure each symbol's token is (or becomes) subscribed.

        Safe to call on every poll: only *new* tokens trigger a subscribe
        message; already-subscribed tokens are a no-op.
        """
        new_tokens: list[str] = []
        with self._lock:
            for s in symbols:
                tok = self._symbol_to_token.get((s or "").upper().strip())
                if not tok:
                    continue
                if tok not in self._want:
                    self._want.add(tok)
                if self._connected and tok not in self._sent:
                    new_tokens.append(tok)
        if new_tokens:
            self._send_subscribe(new_tokens)

    def get_ltp(self, symbol: str) -> tuple[float, float] | None:
        """Return ``(ltp, age_seconds)`` for a symbol, or None if no fresh tick.

        ``age_seconds`` is measured against the exchange feed timestamp, so the
        caller can surface true data freshness.
        """
        with self._lock:
            tok = self._symbol_to_token.get((symbol or "").upper().strip())
            if not tok:
                return None
            tick = self._ticks.get(tok)
        if tick is None:
            return None
        ltp, feed_ts = tick
        age = time.time() - feed_ts
        if age > _TICK_FRESH_S or ltp <= 0:
            return None
        return ltp, max(0.0, age)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def started(self) -> bool:
        """True once the background thread has been launched (idempotent guard
        so the caller doesn't re-fetch credentials / re-login on every poll)."""
        return self._started

    # -- Internal: connection loop -----------------------------------------

    def _run_forever(self) -> None:
        import websocket

        creds = self._creds or {}
        headers = [
            f"Authorization: {creds.get('auth_token', '')}",
            f"x-api-key: {creds.get('api_key', '')}",
            f"x-client-code: {creds.get('client_code', '')}",
            f"x-feed-token: {creds.get('feed_token', '')}",
        ]

        backoff = 1.0
        while not self._stop.is_set():
            try:
                ws = websocket.WebSocketApp(
                    _WS_URL,
                    header=headers,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                with self._lock:
                    self._ws = ws
                # ping_interval/payload here are protocol-level pings; Angel One
                # also wants an app-level "ping" text frame, sent by _heartbeat.
                ws.run_forever(ping_interval=0)
            except Exception as exc:  # noqa: BLE001
                log.warning("SmartStream WS loop error: {}", exc)

            if self._stop.is_set():
                break
            # Reconnect with capped exponential backoff.
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)
        log.info("SmartStream WS loop exited")

    def _on_open(self, ws: Any) -> None:
        log.info("SmartStream WS connected")
        with self._lock:
            self._connected = True
            self._sent.clear()
            tokens = sorted(self._want)
        if tokens:
            self._send_subscribe(tokens)
        # Start a heartbeat thread bound to this connection.
        threading.Thread(target=self._heartbeat, args=(ws,),
                         name="smartstream-hb", daemon=True).start()

    def _on_close(self, ws: Any, status_code: Any, msg: Any) -> None:
        with self._lock:
            self._connected = False
            self._sent.clear()
        log.info("SmartStream WS closed (status={})", status_code)

    def _on_error(self, ws: Any, error: Any) -> None:
        # Never echo headers/credentials -- only the error text.
        log.warning("SmartStream WS error: {}", error)

    def _heartbeat(self, ws: Any) -> None:
        while not self._stop.is_set() and self._connected:
            try:
                ws.send("ping")
            except Exception:  # noqa: BLE001
                break
            self._stop.wait(_HEARTBEAT_S)

    def _send_subscribe(self, tokens: list[str]) -> None:
        with self._lock:
            ws = self._ws
            if not self._connected or ws is None:
                return
        msg = {
            "correlationID": uuid.uuid4().hex[:10],
            "action": _ACTION_SUBSCRIBE,
            "params": {
                "mode": _MODE_LTP,
                "tokenList": [
                    {"exchangeType": _EXCHANGE_NSE_CM, "tokens": tokens},
                ],
            },
        }
        try:
            ws.send(json.dumps(msg))
            with self._lock:
                self._sent.update(tokens)
            log.info("SmartStream subscribed {} token(s)", len(tokens))
        except Exception as exc:  # noqa: BLE001
            log.warning("SmartStream subscribe failed: {}", exc)

    # -- Internal: message parsing -----------------------------------------

    def _on_message(self, ws: Any, message: Any) -> None:
        # Text frames are heartbeat acks ("pong") or JSON error envelopes.
        if isinstance(message, str):
            if message.strip().lower() != "pong":
                log.debug("SmartStream text frame: {}", message[:200])
            return
        if not isinstance(message, (bytes, bytearray)):
            return
        # LTP mode: each packet is 51 bytes. A frame may carry one or several
        # concatenated packets; parse every whole 51-byte packet present.
        buf = bytes(message)
        n = len(buf)
        off = 0
        parsed = 0
        while off + _LTP_PACKET_LEN <= n:
            self._parse_ltp_packet(buf, off)
            off += _LTP_PACKET_LEN
            parsed += 1
        if parsed == 0 and n >= _LTP_PACKET_LEN:
            self._parse_ltp_packet(buf, 0)

    def _parse_ltp_packet(self, buf: bytes, base: int) -> None:
        try:
            mode = buf[base + 0]
            if mode != _MODE_LTP:
                return
            token = buf[base + 2: base + 27].split(b"\x00", 1)[0].decode(
                "ascii", "ignore"
            ).strip()
            if not token:
                return
            # Exchange feed timestamp (epoch ms) and LTP (paise), int64 LE.
            feed_ms = struct.unpack_from("<q", buf, base + 35)[0]
            ltp_paise = struct.unpack_from("<q", buf, base + 43)[0]
        except (struct.error, IndexError):
            return
        ltp = ltp_paise / _PAISE_PER_RUPEE
        if ltp <= 0:
            return
        # Some feeds send 0/garbage timestamps; fall back to local clock so the
        # freshness check still behaves (treats the tick as "just received").
        feed_ts = (feed_ms / 1000.0) if feed_ms > 0 else time.time()
        with self._lock:
            self._ticks[token] = (ltp, feed_ts)


# Module-level singleton used by src/web/live.py.
_feed: LiveFeed | None = None
_feed_lock = threading.Lock()


def get_feed() -> LiveFeed:
    global _feed
    if _feed is None:
        with _feed_lock:
            if _feed is None:
                _feed = LiveFeed()
    return _feed
