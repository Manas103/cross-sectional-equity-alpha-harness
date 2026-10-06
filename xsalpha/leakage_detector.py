"""A guard that fails a run when it reads a vintage published after the simulation clock.

LeakageDetector wraps `VintageStore.as_of` with one extra check the store
itself does not make: that the query_date actually passed in is not later
than the simulation's own rebalance clock, and that nothing returned carries
a knowledge_date later than that clock. The store will honor whatever
query_date it is handed (that is a deliberate design choice, see its
docstring); the detector is the independent second check that the call site
handed it the right one.
"""
from __future__ import annotations

import pandas as pd

from .vintage_store import VintageStore


class LeakageError(RuntimeError):
    pass


class LeakageDetector:
    def __init__(self, store: VintageStore) -> None:
        self.store = store

    def guarded_as_of(self, value_date: pd.Timestamp, query_date: pd.Timestamp, simulation_clock: pd.Timestamp) -> pd.DataFrame:
        if query_date > simulation_clock:
            raise LeakageError(
                f"query_date {query_date.date()} is after simulation clock {simulation_clock.date()}"
            )
        rows = self.store.as_of(value_date, query_date)
        if not rows.empty:
            max_knowledge = rows["knowledge_date"].max()
            if max_knowledge > simulation_clock:
                raise LeakageError(
                    f"vintage with knowledge_date {max_knowledge.date()} exceeds simulation clock {simulation_clock.date()}"
                )
        return rows
