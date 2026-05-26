"""Signal generation: turn calibrated predictions into actionable BUY rows.

A *signal* is the bridge between an ML probability and a tradable order:
it carries the entry price, stop-loss, take-profit, and quantity computed
under our risk + cost framework. Signals live in ``signal_outbox`` and
become *paper trades* once the reconciler fills them.

This package is intentionally read-mostly w.r.t. predictions, prices, and
features -- it never re-trains a model or recomputes features. Its only
write is to ``signal_outbox`` (and indirectly ``validation_failures`` if
inputs look wrong).
"""
