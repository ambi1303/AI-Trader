-- Postgres mirror schema for the AI Trader dashboard (Neon free tier).
--
-- This is a deliberately LOOSE mirror, NOT the authoritative schema:
--   * No FOREIGN KEY / CHECK constraints -- the data already passed them in
--     the local SQLite source of truth, and dropping them lets the publisher
--     TRUNCATE + reload tables in any order.
--   * Date/time columns are TEXT (ISO 'YYYY-MM-DD' / ISO-8601), exactly as
--     stored in SQLite, so psycopg returns plain strings and the dashboard
--     code behaves identically across both backends (no datetime objects).
--   * Only the columns the dashboard actually reads are mirrored; heavy
--     blobs (feature_snapshot_json, payload_json, the wide feature table) are
--     omitted to stay well under Neon's 0.5 GB free cap.
--
-- All statements are idempotent (CREATE TABLE IF NOT EXISTS).

-- ---------------------------------------------------------------
-- Reference / static
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nifty_constituents (
    symbol      TEXT NOT NULL,
    start_date  TEXT NOT NULL,
    end_date    TEXT,
    index_name  TEXT NOT NULL DEFAULT 'NIFTY50',
    PRIMARY KEY (symbol, start_date, index_name)
);

CREATE TABLE IF NOT EXISTS trading_calendar (
    cal_date            TEXT PRIMARY KEY,
    is_holiday          INTEGER NOT NULL DEFAULT 0,
    description         TEXT,
    is_special_session  INTEGER NOT NULL DEFAULT 0,
    session_open        TEXT,
    session_close       TEXT
);

CREATE TABLE IF NOT EXISTS stock_sectors (
    symbol        TEXT PRIMARY KEY,
    sector        TEXT NOT NULL,
    sector_index  TEXT
);

-- ---------------------------------------------------------------
-- Market data (recent slice only -- ~1 trading year)
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS price_data (
    symbol       TEXT NOT NULL,
    bar_date     TEXT NOT NULL,
    open         DOUBLE PRECISION,
    high         DOUBLE PRECISION,
    low          DOUBLE PRECISION,
    close        DOUBLE PRECISION,
    volume       BIGINT,
    adj_close    DOUBLE PRECISION,
    source       TEXT NOT NULL,
    ingested_at  TEXT,
    PRIMARY KEY (symbol, bar_date, source)
);
CREATE INDEX IF NOT EXISTS ix_price_data_symbol_date ON price_data(symbol, bar_date);

-- Narrow feature slice: only the overlay indicators the chart draws.
CREATE TABLE IF NOT EXISTS feature_data (
    symbol        TEXT NOT NULL,
    feature_date  TEXT NOT NULL,
    ema_20        DOUBLE PRECISION,
    ema_50        DOUBLE PRECISION,
    ema_200       DOUBLE PRECISION,
    rsi_14        DOUBLE PRECISION,
    macd          DOUBLE PRECISION,
    macd_signal   DOUBLE PRECISION,
    macd_hist     DOUBLE PRECISION,
    PRIMARY KEY (symbol, feature_date)
);

-- Index data (Nifty / VIX / sector indices) -- needed by the cloud pipeline
-- to compute regime features. Tiny, so mirrored in full.
CREATE TABLE IF NOT EXISTS index_data (
    index_symbol TEXT NOT NULL,
    bar_date     TEXT NOT NULL,
    open         DOUBLE PRECISION,
    high         DOUBLE PRECISION,
    low          DOUBLE PRECISION,
    close        DOUBLE PRECISION,
    volume       BIGINT,
    source       TEXT NOT NULL,
    ingested_at  TEXT,
    PRIMARY KEY (index_symbol, bar_date, source)
);
CREATE INDEX IF NOT EXISTS ix_index_data_idx_date ON index_data(index_symbol, bar_date);

-- Corporate actions -- needed for split/bonus-aware feature computation.
CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol       TEXT NOT NULL,
    ex_date      TEXT NOT NULL,
    action_type  TEXT NOT NULL,
    ratio_from   BIGINT,
    ratio_to     BIGINT,
    amount       DOUBLE PRECISION,
    notes        TEXT,
    source       TEXT NOT NULL DEFAULT 'seed',
    PRIMARY KEY (symbol, ex_date, action_type)
);

-- ---------------------------------------------------------------
-- Fundamentals
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fundamental_data (
    symbol           TEXT NOT NULL,
    as_of_date       TEXT NOT NULL,
    source           TEXT NOT NULL,
    pe_ttm           DOUBLE PRECISION,
    pb               DOUBLE PRECISION,
    roe              DOUBLE PRECISION,
    debt_to_equity   DOUBLE PRECISION,
    profit_margin    DOUBLE PRECISION,
    revenue_growth   DOUBLE PRECISION,
    earnings_growth  DOUBLE PRECISION,
    dividend_yield   DOUBLE PRECISION,
    market_cap       DOUBLE PRECISION,
    eps_ttm          DOUBLE PRECISION,
    book_value       DOUBLE PRECISION,
    PRIMARY KEY (symbol, as_of_date)
);
CREATE INDEX IF NOT EXISTS ix_fundamental_symbol_date ON fundamental_data(symbol, as_of_date);

-- ---------------------------------------------------------------
-- Models / signals / paper trades
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_runs (
    run_id        TEXT PRIMARY KEY,
    model_name    TEXT NOT NULL,
    trained_from  TEXT,
    trained_to    TEXT,
    metrics_json  TEXT,
    created_at    TEXT
);

CREATE TABLE IF NOT EXISTS predictions_log (
    id               BIGINT PRIMARY KEY,
    run_id           TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    prediction_date  TEXT NOT NULL,
    raw_prob         DOUBLE PRECISION,
    calibrated_prob  DOUBLE PRECISION,
    verdict          TEXT,
    prob_buy         DOUBLE PRECISION,
    prob_hold        DOUBLE PRECISION,
    prob_sell        DOUBLE PRECISION,
    predicted_return DOUBLE PRECISION,
    target_price     DOUBLE PRECISION,
    stop_price       DOUBLE PRECISION,
    created_at       TEXT
);
CREATE INDEX IF NOT EXISTS ix_pred_symbol_date ON predictions_log(symbol, prediction_date);
CREATE INDEX IF NOT EXISTS ix_pred_date ON predictions_log(prediction_date);

CREATE TABLE IF NOT EXISTS signal_outbox (
    id           BIGINT PRIMARY KEY,
    symbol       TEXT NOT NULL,
    signal_date  TEXT NOT NULL,
    side         TEXT NOT NULL,
    entry_price  DOUBLE PRECISION,
    stop_loss    DOUBLE PRECISION,
    take_profit  DOUBLE PRECISION,
    qty          BIGINT,
    confidence   DOUBLE PRECISION,
    status       TEXT NOT NULL DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS ix_outbox_date ON signal_outbox(signal_date);

CREATE TABLE IF NOT EXISTS paper_trades (
    id            BIGINT PRIMARY KEY,
    symbol        TEXT NOT NULL,
    sector        TEXT,
    side          TEXT,
    qty           BIGINT,
    entry_date    TEXT,
    exit_date     TEXT,
    entry_price   DOUBLE PRECISION,
    exit_price    DOUBLE PRECISION,
    pnl_rupees    DOUBLE PRECISION,
    pnl_pct       DOUBLE PRECISION,
    exit_reason   TEXT,
    status        TEXT NOT NULL DEFAULT 'open',
    stop_loss     DOUBLE PRECISION,
    take_profit   DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS ix_paper_status ON paper_trades(status);

-- ---------------------------------------------------------------
-- View: today's tradable universe (mirrors the SQLite view, Postgres dialect)
-- ---------------------------------------------------------------
DROP VIEW IF EXISTS v_universe_today;
CREATE VIEW v_universe_today AS
SELECT symbol
FROM   nifty_constituents
WHERE  end_date IS NULL
   OR  end_date >= to_char(CURRENT_DATE, 'YYYY-MM-DD');
