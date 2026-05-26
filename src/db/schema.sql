-- Schema for ai_trading_system. Idempotent (CREATE IF NOT EXISTS).
-- Versioned via the schema_version table. Bump SCHEMA_VERSION in migrate.py
-- when adding new statements; migrations append, never destroy.

CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- ---------------------------------------------------------------
-- Reference / static
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nifty_constituents (
    symbol      TEXT NOT NULL,
    start_date  TEXT NOT NULL,           -- ISO YYYY-MM-DD
    end_date    TEXT,                    -- NULL = currently in index
    index_name  TEXT NOT NULL DEFAULT 'NIFTY50',
    notes       TEXT,
    PRIMARY KEY (symbol, start_date, index_name)
);
CREATE INDEX IF NOT EXISTS ix_constituents_index_name ON nifty_constituents(index_name);
CREATE INDEX IF NOT EXISTS ix_constituents_dates      ON nifty_constituents(start_date, end_date);

CREATE TABLE IF NOT EXISTS trading_calendar (
    cal_date            TEXT PRIMARY KEY, -- ISO YYYY-MM-DD
    is_holiday          INTEGER NOT NULL,
    description         TEXT,
    is_special_session  INTEGER NOT NULL DEFAULT 0,
    session_open        TEXT,             -- HH:MM, only for special sessions
    session_close       TEXT
);

-- ---------------------------------------------------------------
-- Market data
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS price_data (
    symbol       TEXT NOT NULL,
    bar_date     TEXT NOT NULL,           -- ISO YYYY-MM-DD
    open         REAL NOT NULL,
    high         REAL NOT NULL,
    low          REAL NOT NULL,
    close        REAL NOT NULL,
    volume       INTEGER NOT NULL,
    adj_close    REAL,
    source       TEXT NOT NULL,           -- yfinance | bhavcopy | nsepython
    ingested_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (symbol, bar_date, source),
    CHECK (open  > 0),
    CHECK (high  > 0),
    CHECK (low   > 0),
    CHECK (close > 0),
    CHECK (volume >= 0),
    CHECK (high >= low),
    CHECK (high >= open),
    CHECK (high >= close),
    CHECK (low  <= open),
    CHECK (low  <= close)
);
CREATE INDEX IF NOT EXISTS ix_price_data_symbol_date ON price_data(symbol, bar_date);
CREATE INDEX IF NOT EXISTS ix_price_data_date        ON price_data(bar_date);

CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol       TEXT NOT NULL,
    ex_date      TEXT NOT NULL,           -- ISO YYYY-MM-DD
    action_type  TEXT NOT NULL,           -- split | bonus | dividend | rights | merger | demerger
    ratio_from   INTEGER,
    ratio_to     INTEGER,
    amount       REAL,
    notes        TEXT,
    source       TEXT NOT NULL DEFAULT 'seed',
    PRIMARY KEY (symbol, ex_date, action_type)
);
CREATE INDEX IF NOT EXISTS ix_corp_actions_symbol ON corporate_actions(symbol);

CREATE TABLE IF NOT EXISTS circuit_flags (
    symbol      TEXT NOT NULL,
    bar_date    TEXT NOT NULL,
    hit_upper   INTEGER NOT NULL DEFAULT 0,
    hit_lower   INTEGER NOT NULL DEFAULT 0,
    band_pct    REAL,
    PRIMARY KEY (symbol, bar_date)
);

-- ---------------------------------------------------------------
-- News (Week 2)
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS news_headlines (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol         TEXT,                  -- NULL for market-wide news
    published_at   TEXT NOT NULL,
    source         TEXT NOT NULL,
    title          TEXT NOT NULL,
    url            TEXT,
    sentiment      REAL,                  -- filled later by FinBERT
    sentiment_label TEXT,
    ingested_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE (source, url)
);
CREATE INDEX IF NOT EXISTS ix_news_symbol_date ON news_headlines(symbol, published_at);

-- ---------------------------------------------------------------
-- Models / signals (Weeks 3-6)
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_runs (
    run_id          TEXT PRIMARY KEY,
    model_name      TEXT NOT NULL,
    git_sha         TEXT,
    feature_hash    TEXT,
    trained_from    TEXT,
    trained_to      TEXT,
    metrics_json    TEXT,
    artifact_path   TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS predictions_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    prediction_date TEXT NOT NULL,        -- date the prediction is made for
    raw_prob        REAL,
    calibrated_prob REAL,
    feature_snapshot_json TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    FOREIGN KEY (run_id) REFERENCES model_runs(run_id)
);
CREATE INDEX IF NOT EXISTS ix_pred_symbol_date ON predictions_log(symbol, prediction_date);

CREATE TABLE IF NOT EXISTS signal_outbox (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT NOT NULL,
    signal_date  TEXT NOT NULL,
    side         TEXT NOT NULL,           -- BUY | SELL | EXIT
    entry_price  REAL,
    stop_loss    REAL,
    take_profit  REAL,
    qty          INTEGER,
    confidence   REAL,
    status       TEXT NOT NULL DEFAULT 'pending',
                                          -- pending | sent | executed | skipped | failed
    payload_json TEXT,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    sent_at      TEXT,
    error        TEXT
);
CREATE INDEX IF NOT EXISTS ix_outbox_status ON signal_outbox(status);

CREATE TABLE IF NOT EXISTS paper_trades (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id      INTEGER,
    symbol         TEXT NOT NULL,
    side           TEXT NOT NULL,
    entry_date     TEXT,
    exit_date      TEXT,
    entry_price    REAL,
    exit_price     REAL,
    qty            INTEGER,
    pnl_rupees     REAL,
    pnl_pct        REAL,
    cost_rupees    REAL,
    notes          TEXT,
    -- v4: lifecycle + risk fields so reconciliation can be stateless w.r.t.
    -- the original signal row.
    sector         TEXT,
    status         TEXT NOT NULL DEFAULT 'open',   -- 'open' | 'closed'
    stop_loss      REAL,
    take_profit    REAL,
    trailing_stop  REAL,
    entry_atr      REAL,
    high_watermark REAL,
    exit_reason    TEXT,                           -- stop|target|trail|time|forced|manual
    entry_prob     REAL,
    threshold      REAL,
    run_id         TEXT,                           -- model_runs.run_id traceability
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    FOREIGN KEY (signal_id) REFERENCES signal_outbox(id),
    FOREIGN KEY (run_id) REFERENCES model_runs(run_id)
);

-- NOTE: indexes on the v4-added columns of paper_trades and the new unique
-- index on signal_outbox(symbol, signal_date) live in src/db/migrate.py
-- (function `_migrate_to_v4`). Putting them here would break upgrades from
-- v3 because those columns don't yet exist on the live DB at the moment
-- this script is replayed by `execute_script`.

-- ---------------------------------------------------------------
-- Validation / observability
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS validation_failures (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    check_name   TEXT NOT NULL,
    symbol       TEXT,
    issue_date   TEXT,
    severity     TEXT NOT NULL,           -- info | warning | error | critical
    message      TEXT NOT NULL,
    details_json TEXT,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS ix_valfail_run    ON validation_failures(run_id);
CREATE INDEX IF NOT EXISTS ix_valfail_check  ON validation_failures(check_name, severity);

-- ---------------------------------------------------------------
-- Week 2: Features & sector mapping
-- ---------------------------------------------------------------

-- Indices: Nifty 50 (^NSEI), India VIX (^INDIAVIX), and NSE sectoral indices
-- (^NSEBANK, ^CNXIT, ^CNXFMCG, ...). Schema mirrors price_data but is
-- separate so equity/index data don't collide on PK and so we can model
-- them differently (no volume for VIX, etc.).
CREATE TABLE IF NOT EXISTS index_data (
    index_symbol TEXT NOT NULL,            -- e.g. ^NSEI, ^INDIAVIX, ^NSEBANK
    bar_date     TEXT NOT NULL,
    open         REAL,
    high         REAL,
    low          REAL,
    close        REAL NOT NULL,
    volume       INTEGER,
    source       TEXT NOT NULL,            -- yfinance | nse
    ingested_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (index_symbol, bar_date, source),
    CHECK (close > 0)
);
CREATE INDEX IF NOT EXISTS ix_index_data_idx_date ON index_data(index_symbol, bar_date);

-- Stock -> sector mapping (each stock can map to one sector index for v1).
-- Sourced from config/stock_to_sector.csv. Used by regime_features to compute
-- sector-relative strength.
CREATE TABLE IF NOT EXISTS stock_sectors (
    symbol        TEXT PRIMARY KEY,
    sector        TEXT NOT NULL,           -- e.g. "BANK", "IT", "FMCG", ...
    sector_index  TEXT NOT NULL,           -- e.g. ^NSEBANK
    notes         TEXT
);

-- The wide feature table: one row per (symbol, feature_date), all features
-- as columns. We use a wide schema (not key-value) for fast model training
-- and easy column-level introspection. The schema is intentionally
-- enumerated; adding a feature requires a migration so the contract is
-- explicit and reproducible.
CREATE TABLE IF NOT EXISTS feature_data (
    symbol               TEXT NOT NULL,
    feature_date         TEXT NOT NULL,
    -- raw price refs
    close                REAL,
    volume               INTEGER,
    -- returns
    ret_1d               REAL,
    ret_5d               REAL,
    ret_10d              REAL,
    ret_20d              REAL,
    log_ret_1d           REAL,
    -- volatility
    vol_5d               REAL,
    vol_20d              REAL,
    vol_60d              REAL,
    -- momentum
    mom_5d               REAL,
    mom_20d              REAL,
    mom_60d              REAL,
    -- drawdowns
    dd_from_high_20d     REAL,
    dd_from_high_60d     REAL,
    dd_from_high_252d    REAL,
    -- gaps
    gap_pct              REAL,
    -- technicals
    rsi_14               REAL,
    macd                 REAL,
    macd_signal          REAL,
    macd_hist            REAL,
    ema_20               REAL,
    ema_50               REAL,
    ema_200              REAL,
    dist_ema_20_pct      REAL,
    dist_ema_50_pct      REAL,
    dist_ema_200_pct     REAL,
    bb_upper             REAL,
    bb_lower             REAL,
    bb_pct_b             REAL,
    bb_bandwidth         REAL,
    atr_14               REAL,
    atr_pct              REAL,
    adx_14               REAL,
    plus_di_14           REAL,
    minus_di_14          REAL,
    obv                  REAL,
    stoch_k              REAL,
    stoch_d              REAL,
    -- volume features
    vol_avg_20d          REAL,
    vol_z_20d            REAL,
    vol_ratio_20d        REAL,
    -- regime
    nifty_dist_ma50_pct  REAL,
    nifty_dist_ma200_pct REAL,
    vix_level            REAL,
    vix_chg_5d_pct       REAL,
    beta_60d             REAL,
    corr_60d             REAL,
    sector_rs_20d        REAL,
    -- circuit / liquidity
    hit_upper_circuit    INTEGER,
    hit_lower_circuit    INTEGER,
    days_since_circuit   INTEGER,
    low_volume_flag      INTEGER,
    -- meta
    feature_set_version  INTEGER NOT NULL DEFAULT 1,
    computed_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (symbol, feature_date)
);
CREATE INDEX IF NOT EXISTS ix_feature_data_date ON feature_data(feature_date);

-- ---------------------------------------------------------------
-- Week 4: Backtesting
-- ---------------------------------------------------------------

-- One row per backtest invocation. Holds the params and aggregate metrics so
-- multiple backtests can be compared over time. Equity curves, daily returns,
-- and the trade ledger live in dedicated child tables joined by run_id.
CREATE TABLE IF NOT EXISTS backtest_runs (
    bt_run_id      TEXT PRIMARY KEY,
    model_run_id   TEXT,                    -- the model that produced the predictions
    name           TEXT,                    -- human-readable label, e.g. "smoke" / "stress_covid"
    start_date     TEXT,
    end_date       TEXT,
    initial_capital REAL NOT NULL,
    config_json    TEXT,                    -- sizing/risk/cost knobs used
    metrics_json   TEXT,                    -- Sharpe/DD/HitRate/etc.
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    FOREIGN KEY (model_run_id) REFERENCES model_runs(run_id)
);
CREATE INDEX IF NOT EXISTS ix_bt_runs_model ON backtest_runs(model_run_id);

CREATE TABLE IF NOT EXISTS backtest_equity (
    bt_run_id   TEXT NOT NULL,
    bar_date    TEXT NOT NULL,
    cash        REAL NOT NULL,
    equity      REAL NOT NULL,              -- cash + sum(qty * close) for open positions
    open_count  INTEGER NOT NULL,
    daily_pnl   REAL NOT NULL,
    PRIMARY KEY (bt_run_id, bar_date),
    FOREIGN KEY (bt_run_id) REFERENCES backtest_runs(bt_run_id)
);

-- Trade ledger for backtests. Mirrors paper_trades (Week 5 will reuse the
-- same shape) but is namespaced by bt_run_id so we can re-run any number of
-- backtests without colliding with paper-trading rows.
CREATE TABLE IF NOT EXISTS backtest_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    bt_run_id       TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,          -- LONG | SHORT (v1: LONG only)
    entry_date      TEXT NOT NULL,
    exit_date       TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    exit_price      REAL NOT NULL,
    qty             INTEGER NOT NULL,
    gross_pnl       REAL NOT NULL,
    cost_rupees     REAL NOT NULL,
    net_pnl         REAL NOT NULL,
    holding_days    INTEGER NOT NULL,
    exit_reason     TEXT NOT NULL,          -- stop | target | trail | time | end
    entry_prob      REAL,
    threshold       REAL,
    FOREIGN KEY (bt_run_id) REFERENCES backtest_runs(bt_run_id)
);
CREATE INDEX IF NOT EXISTS ix_bt_trades_run    ON backtest_trades(bt_run_id);
CREATE INDEX IF NOT EXISTS ix_bt_trades_symbol ON backtest_trades(symbol);

-- ---------------------------------------------------------------
-- Useful views
-- ---------------------------------------------------------------
DROP VIEW IF EXISTS v_universe_today;
CREATE VIEW v_universe_today AS
SELECT symbol
FROM   nifty_constituents
WHERE  end_date IS NULL
   OR  end_date >= strftime('%Y-%m-%d','now');
