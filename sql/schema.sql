-- PostgreSQL-shaped version of the panel table DuckDB actually builds and
-- queries in xsalpha/db.py (SERIAL/TIMESTAMPTZ types, an index matching the
-- (ticker, date) access pattern of the ADV and forward-return window
-- queries). Designed, not exercised live in this session: no PostgreSQL
-- server is stood up here, the same accepted precedent as
-- futures-strategy-backtester's storage layer. DuckDB's read of the
-- identical column set is what is actually measured; see README.

CREATE TABLE bars (
    id          SERIAL PRIMARY KEY,
    date        DATE NOT NULL,
    ticker      VARCHAR(16) NOT NULL,
    sector      VARCHAR(32) NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL,
    vwap        DOUBLE PRECISION NOT NULL,
    earn_yield  DOUBLE PRECISION NOT NULL,
    quality     DOUBLE PRECISION NOT NULL,
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (ticker, date)
);

CREATE INDEX idx_bars_ticker_date ON bars (ticker, date);
