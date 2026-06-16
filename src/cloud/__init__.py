"""Cloud mirror: publish the dashboard-relevant subset of the local SQLite
database to a managed Postgres (Neon) so the web dashboard can be viewed from
any device without exposing the full 750 MB training dataset.

Only a handful of small tables (predictions, signals, paper trades,
fundamentals, model runs, reference data) plus a recent slice of price/feature
data are mirrored. The local pipeline remains the single source of truth and
stays on SQLite.
"""
