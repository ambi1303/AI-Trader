"""Paper-trading layer.

Bridges ``signal_outbox`` -> ``paper_trades`` -> mark-to-market.

The reconciler runs once per trading day. On each pass it does TWO things,
in order:

1. Close existing open positions if today's bar triggered a stop, target,
   trailing-stop, or the time-stop. Intraday hit logic is reused from the
   backtester so live and historical results stay consistent.
2. Open new positions from yesterday's pending signals using *today's open*
   as the fill price, applying entry costs from the canonical Indian-equity
   cost model and updating the linked ``signal_outbox`` row to ``executed``.

Idempotency
-----------
Every reconciliation pass is idempotent for a given ``as_of`` date:

* Already-closed paper trades are skipped.
* Already-executed signals (``status != 'pending'``) are skipped.
* Mark-to-market updates are simple overwrites.
"""
