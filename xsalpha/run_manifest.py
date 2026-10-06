"""Pins a run to a data vintage and a code digest so a re-run can be checked
for exact reproduction.

`build_manifest` hashes the vintage store's full content (every (entity,
field, value_date, knowledge_date, value) row, sorted into a stable order
first) and the source bytes of every .py file in this package, and records
the random seed. Re-running `compute_pinned_statistic` with the same
manifest's seed against the same store must return bit-identical floats;
`scripts/run_pinned_reproducibility.py` checks this twice.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .vintage_store import VintageStore

PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RunManifest:
    data_vintage_digest: str
    code_digest: str
    seed: int

    def to_json(self) -> str:
        return json.dumps(
            {"data_vintage_digest": self.data_vintage_digest, "code_digest": self.code_digest, "seed": self.seed},
            indent=2,
        )


def _hash_store(store: VintageStore) -> str:
    frame = store._as_frame().copy()
    frame = frame.sort_values(["entity", "field", "value_date", "knowledge_date"]).reset_index(drop=True)
    payload = frame.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_code() -> str:
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_manifest(store: VintageStore, seed: int) -> RunManifest:
    return RunManifest(data_vintage_digest=_hash_store(store), code_digest=_hash_code(), seed=seed)


def compute_pinned_statistic(store: VintageStore, manifest: RunManifest, as_of_date: pd.Timestamp) -> dict:
    """A small, deterministic statistic over the as-of panel: mean and std of
    every field known as of `as_of_date`, plus a seeded random resample mean,
    all driven only by the manifest's seed and the store's content.
    """
    frame = store._as_frame()
    known = frame[frame["knowledge_date"] <= as_of_date]
    latest_idx = known.groupby(["entity", "field", "value_date"])["knowledge_date"].idxmax()
    latest = known.loc[latest_idx]

    rng = np.random.default_rng(manifest.seed)
    values = latest["value"].to_numpy()
    resample_idx = rng.integers(0, len(values), size=min(5000, len(values)))
    resample_mean = float(np.mean(values[resample_idx])) if len(values) else 0.0

    return {
        "n_rows": int(len(latest)),
        "mean": float(np.mean(values)) if len(values) else 0.0,
        "std": float(np.std(values)) if len(values) else 0.0,
        "seeded_resample_mean": resample_mean,
        "data_vintage_digest": manifest.data_vintage_digest,
        "code_digest": manifest.code_digest,
    }
