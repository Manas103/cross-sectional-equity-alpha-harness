"""Point-in-time vintage store: every value carries the date it was actually known.

A real data vendor restates things: an earnings-yield input published on day
T is sometimes corrected days later (a late filing, a data-vendor fix). This
store keeps every vintage of every (entity, field) value, each tagged with
the date it became knowable (`knowledge_date`), and `as_of(query_date)`
returns, for each (entity, field), the latest vintage whose `knowledge_date`
is on or before `query_date`, never a later one. That is the single
invariant the leakage detector in leakage_detector.py exists to enforce from
the outside: this store will happily hand back a later vintage if asked for
one, because filtering correctly is the caller's job to get right, which is
exactly why a leakage detector needs to exist as a second, independent
check rather than trusting every call site.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Vintage:
    entity: int
    field: str
    value_date: pd.Timestamp
    knowledge_date: pd.Timestamp
    value: float


class VintageStore:
    def __init__(self) -> None:
        self._rows: list[Vintage] = []
        self._frame: pd.DataFrame | None = None

    def ingest(self, vintages: list[Vintage]) -> None:
        self._rows.extend(vintages)
        self._frame = None

    def _as_frame(self) -> pd.DataFrame:
        if self._frame is None:
            self._frame = pd.DataFrame(
                [(v.entity, v.field, v.value_date, v.knowledge_date, v.value) for v in self._rows],
                columns=["entity", "field", "value_date", "knowledge_date", "value"],
            )
        return self._frame

    def as_of(self, value_date: pd.Timestamp, query_date: pd.Timestamp) -> pd.DataFrame:
        """Every vintage for `value_date`, as known on `query_date`.

        Returns one row per (entity, field): the latest knowledge_date that
        is <= query_date for that (entity, field, value_date). If no vintage
        of a given field was known yet at query_date, that field is absent.
        """
        frame = self._as_frame()
        candidates = frame[(frame["value_date"] == value_date) & (frame["knowledge_date"] <= query_date)]
        if candidates.empty:
            return candidates
        latest_idx = candidates.groupby(["entity", "field"])["knowledge_date"].idxmax()
        return candidates.loc[latest_idx].reset_index(drop=True)

    def latest_known_knowledge_date(self, value_date: pd.Timestamp, query_date: pd.Timestamp) -> pd.Timestamp | None:
        rows = self.as_of(value_date, query_date)
        if rows.empty:
            return None
        return rows["knowledge_date"].max()

    def max_knowledge_date_overall(self) -> pd.Timestamp:
        return self._as_frame()["knowledge_date"].max()


def build_simulated_vintage_store(
    num_entities: int,
    trading_days: pd.DatetimeIndex,
    field: str = "earnings_yield",
    restatement_probability: float = 0.08,
    restatement_delay_days: int = 10,
    seed: int = 909,
) -> VintageStore:
    """Every entity/day gets a same-day preliminary vintage; ~8% also get a
    later-knowledge restated vintage with a different value, which is the
    mechanism a naive backtester can leak by accident.
    """
    rng = np.random.default_rng(seed)
    store = VintageStore()
    rows: list[Vintage] = []
    n_days = len(trading_days)

    preliminary = rng.normal(0.0, 1.0, size=(n_days, num_entities))
    gets_restated = rng.random((n_days, num_entities)) < restatement_probability
    restated_value = preliminary + rng.normal(0.0, 0.6, size=(n_days, num_entities))

    for day_idx, value_date in enumerate(trading_days):
        for entity in range(num_entities):
            rows.append(
                Vintage(
                    entity=entity,
                    field=field,
                    value_date=value_date,
                    knowledge_date=value_date,
                    value=float(preliminary[day_idx, entity]),
                )
            )
            if gets_restated[day_idx, entity]:
                delay = int(rng.integers(1, restatement_delay_days + 1))
                restate_idx = min(day_idx + delay, n_days - 1)
                rows.append(
                    Vintage(
                        entity=entity,
                        field=field,
                        value_date=value_date,
                        knowledge_date=trading_days[restate_idx],
                        value=float(restated_value[day_idx, entity]),
                    )
                )
    store.ingest(rows)
    return store
