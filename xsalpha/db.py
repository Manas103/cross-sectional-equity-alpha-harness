"""DuckDB-backed panel storage and a real SQL window-function query for
trailing average daily dollar volume (the ADV series the impact-cost
model in impact.py needs). This mirrors the store-item-demand-forecast
repo's pattern: the SQL dependency does real work (a `ROWS BETWEEN ...
PRECEDING` window frame), not just table storage.

`sql/schema.sql` also documents a PostgreSQL-shaped version of the same
table (SERIAL/TIMESTAMPTZ types) as a designed-but-not-exercised-live
target, the same accepted precedent pattern as
futures-strategy-backtester's db.py.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def build_duckdb(panel: dict, db_path: str) -> None:
    Path(db_path).unlink(missing_ok=True)
    con = duckdb.connect(db_path)
    frames = []
    for tkr in panel["tickers"]:
        df = pd.DataFrame({
            "date": panel["dates"],
            "ticker": tkr,
            "sector": panel["sector"][tkr],
            "open": panel["open"][tkr].values,
            "high": panel["high"][tkr].values,
            "low": panel["low"][tkr].values,
            "close": panel["close"][tkr].values,
            "volume": panel["volume"][tkr].values,
            "vwap": panel["vwap"][tkr].values,
            "earn_yield": panel["earn_yield"][tkr].values,
            "quality": panel["quality"][tkr].values,
        })
        frames.append(df)
    bars = pd.concat(frames, ignore_index=True)
    con.register("bars_df", bars)
    con.execute("CREATE OR REPLACE TABLE bars AS SELECT * FROM bars_df")
    con.close()


def sql_dollar_adv(db_path: str, window: int = 60) -> pd.DataFrame:
    """Trailing (window)-day average daily dollar volume per ticker, via
    a real DuckDB window-function query. Causal: the frame for date t
    covers t-window .. t-1, never t itself."""
    con = duckdb.connect(db_path)
    query = f"""
        SELECT date, ticker,
               AVG(volume * close) OVER (
                   PARTITION BY ticker ORDER BY date
                   ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
               ) AS adv
        FROM bars
        ORDER BY ticker, date
    """
    df = con.execute(query).fetchdf()
    con.close()
    return df.pivot(index="date", columns="ticker", values="adv")


def sql_forward_return(db_path: str, horizon: int) -> pd.DataFrame:
    """close[t+horizon]/close[t] - 1 per ticker via DuckDB LEAD()."""
    con = duckdb.connect(db_path)
    query = f"""
        SELECT date, ticker,
               LEAD(close, {horizon}) OVER (PARTITION BY ticker ORDER BY date) / close - 1.0 AS fwd_ret
        FROM bars
        ORDER BY ticker, date
    """
    df = con.execute(query).fetchdf()
    con.close()
    return df.pivot(index="date", columns="ticker", values="fwd_ret")
