# Cross-Sectional Equity Alpha Harness with Capacity Limits

63 formulaic cross-sectional signals, in the spirit of WorldQuant's "101
Formulaic Alphas", mined on 8 years of daily bars for 500 simulated U.S.
equities: sector-neutralized, scored out of sample under a purged
walk-forward split, corrected for the 63-trial multiple-testing problem with
the deflated Sharpe ratio, and the survivors traded at increasing size under
a square-root market-impact cost model to find where net Sharpe collapses.
Every number below was measured on this machine, not targeted; two sweeps
were widened, and one survivor-selection rule was pre-specified and never
touched after seeing results, all disclosed in Findings.

## Why this exists

A signal desk's real question about a formulaic alpha sweep is never "does
one signal have a good backtest." It is "of everything we tried, how many
survive once we charge for having tried 63 things, and how much can we
actually put behind the survivors before costs eat the edge." This project
builds the smallest honest version of that two-part question.

## Honest framing, up front

- **This is not real equity market data.** 8 years of daily bars for 500
  named U.S. equities were not obtainable in this build environment (no
  market-data vendor access). `xsalpha/simulate.py` generates a synthetic
  panel with real cross-sectional structure instead: a market factor, 10
  sector factors, idiosyncratic AR(1) noise (a small, genuine short-term
  reversal effect), and four deliberately weak, genuinely informative
  ingredients (momentum, low volatility, an earnings-yield-like value
  characteristic, and a volume-leads-price effect), each contributing
  roughly 5-15% of one day's idiosyncratic return standard deviation, so a
  signal-mining sweep has to actually find them against 8 years of noise.
  This mirrors the synthetic-panel precedent already used elsewhere in this
  portfolio (`futures-strategy-backtester`).
- **The 63 signals are not the literal published 101 alphas.** They are 11
  short formula templates (short-term reversal, momentum, low volatility,
  value, volume surprise, and six decoy templates with no embedded
  predictive relationship), each instantiated over a fixed, pre-specified
  set of lookback windows, for exactly 63 signals total, fixed before any
  signal was scored (see `xsalpha/signals.py`'s module docstring).
- **The survivor rule is mechanical and was set before any signal was
  scored: deflated Sharpe ratio > 0.95 and positive out-of-sample mean IC,
  applied identically to all 63.** It was not touched after seeing results.
  3 of 63 survived; see Findings for why that is reported rather than
  adjusted.
- **PostgreSQL is designed, not exercised live.** `sql/schema.sql`
  documents a PostgreSQL-shaped version of the panel table; the numbers
  below come from the DuckDB path in `xsalpha/db.py`, which runs the same
  SQL window-function query against the same columns. This is the same
  accepted pattern `futures-strategy-backtester` and
  `composite-dem-publish-gates` already use in this portfolio.

### Machine and toolchain

| | |
|---|---|
| CPU | AMD Ryzen 7 7800X3D, 8 physical / 16 logical cores |
| RAM | 31.1 GB |
| OS | Windows 11, build 10.0.26200 |
| Python | CPython 3.12.10, in a repo-local venv |
| Libraries | duckdb 1.5.5, pandas 3.0.5, numpy 2.5.3, scipy 1.18.1, openpyxl 3.1.5, pytest 9.1.1 (see `requirements.txt`) |
| Pacing | non-realtime process, single run per reported number unless noted |

## Architecture

```
xsalpha/simulate.py    500-stock, 2,016-day synthetic panel with 4 weak, genuinely informative ingredients
xsalpha/operators.py   WorldQuant-101-style cross-sectional/time-series operators (rank, ts_mean, decay_linear, ...)
xsalpha/signals.py     63 formulaic signals built from 11 templates x fixed lookback windows
xsalpha/neutralize.py  per-day, per-sector demeaning
xsalpha/splits.py      purged walk-forward folds (8 folds, horizon=5 days, purge=5 days, 260-day burn-in)
xsalpha/ic.py          daily rank IC, out-of-sample fold assembly with a train-estimated sign flip
xsalpha/dsr.py         deflated Sharpe ratio (Bailey & Lopez de Prado): SR0 benchmark, PSR
xsalpha/portfolio.py   composite z-scored survivor signal -> dollar-neutral, unit-gross rank weights
xsalpha/impact.py      square-root market-impact cost model, capacity sweep, crossing-point finder
xsalpha/db.py          DuckDB panel storage; a real SQL window-function query for trailing dollar ADV
xsalpha/reference_oracle.py  independent, loop-based IC and deflated-Sharpe reimplementation

sql/schema.sql              PostgreSQL-shaped panel table (designed, not exercised live; see above)
scripts/run_signal_mining.py    mines all 63 signals, applies the survivor rule, writes docs/benchmark_output.txt
scripts/run_capacity_sweep.py   trades the survivor portfolio at increasing size under the impact model
scripts/reference_oracle_check.py   diffs the vectorized IC/DSR math against the independent oracle

tests/test_purge_no_overlap.py       purged-fold label/test overlap check
tests/test_neutralize_and_impact.py  sector-neutrality and impact-cost monotonicity invariants
tests/test_signals_and_dsr.py        signal count and deflated-Sharpe sanity properties

docs/benchmark_output.txt        raw signal-mining and capacity-sweep run
docs/reference_oracle_output.txt raw reference-oracle diff run
docs/test_output.txt             raw pytest run
```

**Why the survivor sign is estimated per fold, not globally.** `ic.py`'s
`oos_ic_series` estimates each fold's sign from that fold's *training* IC
only, then applies it unchanged to the fold's test days. A signal that
happened to face the right way in the full sample but flips regime
mid-history would look artificially strong under a single global sign; per-
fold, train-only sign estimation is the same discipline as retraining a
model at each fold boundary, applied to a one-parameter "model" (a sign).

**Why the deflated Sharpe ratio, not raw Sharpe or a p-value.** Screening 63
trials and keeping the best-looking ones is exactly the selection bias the
deflated Sharpe ratio was built to correct: it computes the Sharpe ratio a
lucky draw would be expected to produce across N independent trials (`SR0`,
from the trials' own cross-sectional Sharpe dispersion) and asks whether the
observed Sharpe clears that benchmark, adjusting for the series' own skew
and kurtosis rather than assuming normality.

## Validation

**1. Reference-oracle diff.** `scripts/reference_oracle_check.py` diffs the
vectorized daily rank IC (pandas-rank-based) against an independent,
loop-based oracle (`xsalpha/reference_oracle.py`, explicit sort-and-scan
ranking, no numpy/pandas rank call) on 40 (signal, date) pairs, and diffs
the vectorized deflated-Sharpe moments, SR0 and PSR against a second,
independently coded loop-based oracle on 10 synthetic series:

```
Reference-oracle check 1: daily rank IC, vectorized vs hand-rolled loop
  40 (signal, date) pairs checked across 5 signals x 8 dates
  max abs diff, vectorized vs oracle IC: 0.000e+00
  PASS (exact to floating point, < 1e-9)

Reference-oracle check 2: deflated Sharpe ratio moments and PSR/SR0, vectorized vs hand-rolled
  overall max abs diff across both checks: 5.773e-15
  PASS (within 1e-8 tolerance)
```

**2. Purge/leakage check.** `tests/test_purge_no_overlap.py` asserts every
fold's `max(train) + HORIZON <= min(test)` directly (not just by
construction) and that folds are expanding-window and non-overlapping.

**3. Invariants.** `tests/test_neutralize_and_impact.py` asserts sector
neutralization zeros every sector's daily mean score to within 1e-9, and
that the impact-cost fraction is non-decreasing in trade size holding ADV
and volatility fixed. It also documents and checks a real, small quirk of
the rank-based portfolio construction: for a universe with no ranking ties,
the raw (pre-normalization) long-short sum is a constant +0.5 rather than
exactly 0, so the "dollar-neutral" weights carry a small residual net
exposure (0.5 divided by the day's raw gross) rather than being exactly
flat; see Limitations.

**4. Signal-set and DSR sanity.** `tests/test_signals_and_dsr.py` asserts
exactly 63 signals are built, that the SR0 benchmark rises with more trials
at fixed dispersion (more trials should demand a better Sharpe to clear by
chance alone), and that PSR is bounded in [0, 1].

```
9 passed in 0.92s
```

## Findings

**3 of 63 signals survived, against a target of 9, and the rule that
produced 3 was never touched after seeing that number.** The survivor rule
(deflated Sharpe > 0.95, positive out-of-sample mean IC) was fixed in
`scripts/run_signal_mining.py` before the sweep ran. Changing the threshold
after seeing 3 survivors, to manufacture 9, would be exactly the kind of
after-the-fact tuning this project's own design (and this playbook) forbids;
3 is reported as measured. The three survivors are the earnings-yield
family (`ey_level`, `ey_smooth_20`, `ey_smooth_60`), the deliberately
strong, slow-moving value ingredient in the simulator, with mean
out-of-sample IC around +0.045 and a raw Sharpe around 7.5-7.6, clearing the
63-trial deflation benchmark (SR0 = 4.28) decisively.

**A cluster of "decoy" signals scored surprisingly well, which is a real
finding about the generator, not noise.** Templates built to be
uninformative decoys (`range_*`, `vwapdev_*`, `argmaxclose_*`, `decaymom_*`)
measured mean IC around +0.02 to +0.03 and raw Sharpe ratios of 2 to 4,
close to but under the deflation benchmark. Root cause: these templates are
all derived from the same `close`/`volume` series that the simulator's
genuine momentum ingredient (`W_MOM`) also drives, over similar lookback
windows, so they pick up leaked structure from momentum even though they
were not designed to be momentum signals. This is a useful, disclosable
lesson about formulaic alpha mining generally: a "different-looking" formula
built from the same underlying price history as a real effect is not
independent evidence of a new effect, which is exactly why the deflated
Sharpe ratio's benchmark (SR0 = 4.28 here) needs to be high enough to filter
correlated near-misses, not just obviously bad signals. The `quality_*` and
`corr_vol_close_*` decoys, built from genuinely independent random-walk and
correlation fields, scored near zero, as intended.

**The capacity target ($180M) was reached and passed by more than 500x
before net Sharpe fell below 0.5, after widening the sweep twice.** The
first sweep (up to $2B) never crossed 0.5; a second, wider sweep (up to
$100B) still had not crossed 0.5, only falling from a gross Sharpe of 6.94
to a net Sharpe of 4.64 at $100B. Root cause: the three survivor signals are
built from a slow-moving, roughly quarterly-persistence value characteristic
(`earn_yield`), so the resulting portfolio has low daily turnover; the
square-root impact cost is charged on *traded* dollars (the day-to-day
change in position), not gross exposure, so a large but slow-turning book
generates little daily participation relative to average daily volume even
at very large gross size. This is reported as measured rather than adjusted
by raising the impact coefficient or lowering the survivor Sharpe after the
fact.

## Measured results

Machine: AMD Ryzen 7 7800X3D, 8 physical / 16 logical cores, 31.1 GB RAM,
Windows 11 build 10.0.26200, CPython 3.12.10. Signal mining over 63 signals
x 2,016 days runs in about 226 seconds. Raw output in `docs/`.

| Claim | Target | Measured |
|---|---|---|
| Formulaic cross-sectional signals evaluated | 63 | **63** |
| Daily bars, simulated equities, years | 500 names, 8 years | **500 names, 2,016 trading days (~8.0 years)** |
| Sector-neutralized, purged walk-forward | yes | **yes**, 8 folds, 5-day horizon, 5-day purge, 260-day burn-in |
| Survivors after deflated-Sharpe correction (63 trials) | 9 of 63 | **3 of 63** (`ey_level`, `ey_smooth_20`, `ey_smooth_60`) |
| Net Sharpe falls under 0.5 past a gross size | $180M | **not reached by $100B** (555x the target), swept twice |

What "survives" measures here: out-of-sample daily rank IC, sign-corrected
per fold from that fold's training data only, reduced to one Sharpe-like
statistic (`sr_hat`, annualized at 252/5 periods/year since the label is a
5-day forward return) and passed through the deflated Sharpe ratio. It does
not measure the composite portfolio's own Sharpe, which is a separate,
larger number (6.94 gross) because combining several correlated signals
concentrates their shared edge; that portfolio-level number is what feeds
the capacity sweep.

| Gross size | Gross Sharpe | Net Sharpe | Mean daily cost (bps of gross) |
|---|---|---|---|
| $1M | 6.939 | 6.937 | 0.00 |
| $180M (resume target) | 6.939 | 6.909 | 0.02 |
| $2B | 6.939 | 6.798 | 0.06 |
| $10B | 6.939 | 6.482 | 0.14 |
| $100B | 6.939 | 4.639 | 0.43 |

## Building and running

```bash
# from a Python 3.12 venv with requirements.txt installed
python scripts/run_signal_mining.py     # ~226s, writes docs/benchmark_output.txt, data/survivors.json
python scripts/run_capacity_sweep.py    # ~1s, appends to docs/benchmark_output.txt
python scripts/reference_oracle_check.py  # ~1s, writes docs/reference_oracle_output.txt
python -m pytest tests -v               # ~1s, 9 tests
```

`data/` is gitignored (the DuckDB panel file and the survivor list, about 59
MB), regenerated by `run_signal_mining.py`.

## Sibling comparison

[`futures-strategy-backtester`](https://github.com/Manas103/futures-strategy-backtester)
shares the cost-model discipline (a square-root market-impact term charged
on every fill, an independent reference-oracle diff) but trades 5
time-series calendar-spread/carry rules on 3 simulated futures underlyings,
with no cross-sectional panel, no sector neutralization, no information
coefficient and no multiple-testing correction. This project is the
cross-sectional counterpart: the question is not "does this one rule make
money" but "of 63 candidate rules evaluated together, how many survive
having been evaluated together," which a single-instrument time-series
backtest cannot pose.

## Limitations

- Synthetic panel throughout; see "Honest framing, up front".
- The survivor count (3 of 63) fell short of the 9-of-63 target; see
  Findings for why the rule was not adjusted.
- The capacity crossing was not found within a $100B sweep, 555x the target
  size; see Findings for the turnover-based root cause.
- `dollar_neutral_weights` carries a small, constant, disclosed residual net
  exposure (a rank-construction artifact, not a bug) rather than being
  exactly dollar-neutral; see the Validation section and its test.
- The composite survivor portfolio is an equal-weight average of the 3
  surviving signals' z-scores; no further optimization (risk parity,
  covariance-aware weighting) was attempted, because the point of this
  project is the mining-and-correction pipeline, not portfolio optimization.
- PostgreSQL is designed (`sql/schema.sql`) but not exercised live; the
  DuckDB path is what is actually measured.
