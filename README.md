# Cross-Sectional Equity Alpha Harness with Capacity Limits and a Point-in-Time Leakage Detector

63 formulaic cross-sectional signals, in the spirit of WorldQuant's "101
Formulaic Alphas", mined on 8 years of daily bars for 500 simulated U.S.
equities: sector-neutralized, scored out of sample under a purged
walk-forward split, corrected for the 63-trial multiple-testing problem with
the deflated Sharpe ratio, and the survivors traded at increasing size under
a square-root market-impact cost model to find where net Sharpe collapses.
Extended (Oct. 2026) with a point-in-time vintage store, a leakage detector
that fails a run reading a later vintage than its own simulation clock, and
a pinned run manifest (data vintage digest plus code digest plus seed) that
reproduces its own statistic exactly. Extended a second time (Oct. 2026,
Schonfeld Quantitative Research Intern req) with `altdata.py`, `factors.py`
and `combine.py`: two alternative-data signal proxies combined with the
63-signal mining's deflated-Sharpe survivors into one portfolio, a sector,
size and momentum factor attribution of that combined portfolio's daily
return, and the residual's Sharpe gross and net of a half-spread plus the
existing square-root impact model at $250M. Every number below was measured
on this machine, not targeted; two sweeps were widened, one survivor-
selection rule was pre-specified and never touched after seeing results, and
the factor attribution came in well under its target, all disclosed in
Findings. Extended a third time (Oct. 2026, Two Sigma Quantitative Researcher
Intern req) with `realscreen/`, a pre-registered screen of 44 real public
FRED datasets (energy, freight, housing, labor, credit) for quarter-over-
quarter autocorrelation, reusing this repository's multiple-testing
discipline but swapping the deflated Sharpe ratio for a Romano-Wolf (2005)
stepdown, since the quantity being screened here is forecast content in a
real macro series, not a trading signal's return. Unlike the rest of this
repository, every number `realscreen/` reports comes from real downloaded
data, not the synthetic panel.

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
- **The vintage store is a separate, synthetic restatement model, not the
  signal panel's prices.** `xsalpha/vintage_store.py` generates its own
  500-entity, 2,016-day series of preliminary and (about 8% of the time)
  restated values, independent of `xsalpha/simulate.py`'s price panel. It
  exists to exercise the point-in-time and leakage-detection machinery
  honestly, not to re-run the 63-signal mining under restatement.
- **The leakage detector is a second, independent check, not the only
  safeguard.** `VintageStore.as_of` will honor whatever query_date it is
  handed; `LeakageDetector.guarded_as_of` is the call-site-independent check
  that the query_date and every returned row's knowledge_date are no later
  than the simulation clock. The 7 seeded bugs in `xsalpha/leakage_bugs.py`
  are 7 distinct, realistic ways a backtester call site gets this wrong.
- **The two alternative-data signals are proxies built on this repository's
  own panel, not the actual datasets from the sibling alt-data projects.**
  `xsalpha/altdata.py` builds both from the panel's one genuinely
  informative fundamental-like field, `earn_yield`, through a coverage-
  weighted noisy re-read (the same mechanism as
  [`point-in-time-activity-index`](https://github.com/Manas103/point-in-time-activity-index)'s
  `nowcast/` extension) rather than reusing that project's 120-name
  consumer panel or the Filing-Language Signal project's real SEC filings,
  which have no entities in common with this panel's 500 simulated tickers.
  `xsalpha/simulate.py`'s return-generating process itself is untouched, so
  every already-measured number in this README stays valid.
- **The factor attribution is return-based, not a variance R^2.** This
  portfolio is already sector-neutral by construction (`neutralize.py`
  zeros every sector's daily mean score), so a day-to-day variance
  regression against sector returns would show almost nothing by design.
  `xsalpha/factors.py::attribute` instead compares the *sum* of the factor-
  loadings-implied daily return against the sum of the actual daily return,
  which answers "how much of the compounded return came from factor tilts"
  rather than "how much of the day-to-day wiggle."
- **`realscreen/` is entirely real, public FRED data, downloaded with no API
  key.** `fred.stlouisfed.org/graph/fredgraph.csv?id=<SID>` returns a
  series' current observation history directly; no ALFRED vintage history
  and no `api.stlouisfed.org` key were needed or used, because this
  screen's claim is about forecast content in each series' own growth rate,
  not about point-in-time release vintages (that is `point-in-time-activity-
  index`'s `energy_nowcast/` extension, a different claim).
- **Every series is resampled to quarterly by keeping its last real
  observation in that quarter, a declared simplification.** Daily (crude
  oil), weekly (gasoline), monthly (payrolls) and quarterly (mortgage
  delinquency) series are forced onto one common quarterly grid so the same
  one-lag autocorrelation specification applies identically to all 44;
  `realscreen/ingest.py::load_quarterly_levels` is the one place this
  happens.
- **"Freight" is freight and goods-movement broadly, not all 8 series
  literally freight-rate indices.** `TSIFRGHT`, `RAILFRTCARLOADSD11`,
  `TRUCKD11` and `FRGSHPUSM649NCIS` are transportation-specific; `IPCONGD`,
  `TOTBUSSMSA`, `RSAFS` and `TOTBUSIMNSA` are trade/inventory-flow series
  used as the closest available real public proxies once FRED's own
  freight-specific series ran out at 6; `realscreen/config.py::SERIES`
  names every one honestly by its real FRED title.
- **One housing series was swapped before any test was run.** `EXHOSLUSM495S`
  (existing home sales) had only 14 real monthly observations on FRED,
  nowhere near enough for an AR(1) test; it was replaced with `MSACSR`
  (months' supply of new houses, 60+ years of history) before
  `scripts/ingest_real_datasets.py` was ever pointed at the full list, not
  after seeing a weak result from it.
- **The Romano-Wolf bootstrap resamples one common time path across all 44
  series, not 44 independent bootstraps.** This is what lets the corrected
  family-wise error rate account for real cross-sectional correlation (WTI
  and Brent crude oil move together; so do several credit series), the same
  problem a Bonferroni correction ignores entirely; see "How the Romano-Wolf
  stepdown is built" below.

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
xsalpha/vintage_store.py     point-in-time store: every value tagged with the date it was actually known
xsalpha/leakage_detector.py  guard that fails a run reading a vintage published after its simulation clock
xsalpha/leakage_bugs.py      7 seeded, named leakage bugs exercised against the guard
xsalpha/run_manifest.py      pins a data-vintage digest, a code digest and a seed for exact reproduction
xsalpha/altdata.py            two alt-data signal proxies: a coverage-noisy re-read of earn_yield, and a
                               noisier read of its change, both with noise drawn once per quarter
xsalpha/factors.py             sector (per-sector equal-weight return), size and momentum (decile
                                long-short) factor returns, and the return-based attribution regression
xsalpha/combine.py            combines survivor + alt-data scores into one portfolio, runs the
                               attribution, and prices the residual under the existing impact model

sql/schema.sql              PostgreSQL-shaped panel table (designed, not exercised live; see above)
scripts/run_signal_mining.py    mines all 63 signals, applies the survivor rule, writes docs/benchmark_output.txt
scripts/run_capacity_sweep.py   trades the survivor portfolio at increasing size under the impact model
scripts/reference_oracle_check.py   diffs the vectorized IC/DSR math against the independent oracle
scripts/run_leakage_check.py        runs the 7 seeded leakage bugs through the guard, writes docs/leakage_check_output.txt
scripts/run_pinned_reproducibility.py  pins a manifest and checks two runs reproduce exactly, writes docs/pinned_reproducibility_output.txt
scripts/run_combination_attribution.py  combines, attributes, prices the residual, writes docs/attribution_output.txt

tests/test_purge_no_overlap.py       purged-fold label/test overlap check
tests/test_neutralize_and_impact.py  sector-neutrality and impact-cost monotonicity invariants
tests/test_signals_and_dsr.py        signal count and deflated-Sharpe sanity properties
tests/test_vintage_leakage.py        vintage-store invariant, all 7 leakage bugs caught, pinned reproducibility
tests/test_combination_attribution.py  alt-data quarterly cadence, sector-factor shape, attribution
                                        recovers a known loading, combined weights stay dollar-neutral

docs/benchmark_output.txt        raw signal-mining and capacity-sweep run
docs/reference_oracle_output.txt raw reference-oracle diff run
docs/test_output.txt             raw pytest run
docs/leakage_check_output.txt    raw 7-of-7 leakage bug run
docs/pinned_reproducibility_output.txt  raw pinned-manifest reproducibility run
docs/attribution_output.txt      raw combination and factor-attribution run

realscreen/config.py          44 real FRED series ids with category and title, alpha levels,
                               bootstrap size/block length/seed
realscreen/ingest.py          parses raw FRED CSVs, resamples to quarterly, computes growth
realscreen/testing.py         vectorized OLS AR(1) t-stats; joint circular block-bootstrap
                               Romano-Wolf stepdown
realscreen/oracle.py          independent hand-rolled OLS t-stat, no numpy, diffed against
                               testing.py
realscreen/measurements.py    runs every series, naive winners, Romano-Wolf survivors
scripts/ingest_real_datasets.py  builds data/real_dataset_screen_panel.csv from 44 raw CSVs
scripts/run_real_screen.py       runs the screen, writes one docs/dataset_cards/<SID>.md per
                                  dataset
tests/test_real_screen.py        oracle agreement, survivors subset naive, a 3-series
                                  correlated-vs-independent stepdown exercise
docs/real_ingest_output.txt      raw ingest run (datasets found, common quarters)
docs/real_screen_output.txt      raw screen run (naive and Romano-Wolf results)
docs/dataset_cards/<SID>.md      one one-page card per dataset: spec, t-stat, p-value, verdict
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

### How the Romano-Wolf stepdown is built

A naive "test 44 things at 5%" rule expects about 2.2 false positives by
chance alone even if every series were truly a random walk; the real risk
is worse than that arithmetic suggests because several of the 44 series
share real macro-cycle correlation (WTI and Brent crude, or the two
delinquency-rate series), so their test statistics are not independent
draws. Romano-Wolf's stepdown corrects for exactly this by bootstrapping
the *joint* distribution of all 44 statistics rather than treating each one
separately: `testing.romano_wolf_stepdown` draws one common circular-block
bootstrap time path per replication and applies it to every series' own
null (beta=0) residuals, so two series whose real residuals move together
also have bootstrap residuals that move together. At each step, the
largest surviving `|t|` is compared against the bootstrap distribution of
the *maximum* `|t|` over the still-undecided series; anything that clears
it is rejected, the critical value is carried forward (it can only rise,
never fall, across steps), and the process repeats over the shrinking
remaining set until nothing more is rejected.

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

**4b. Attribution recovers a known loading.** `test_attribute_recovers_known_loading`
builds a synthetic return series with known factor loadings (2.0, 0.5) and a known
alpha, runs it through `factors.attribute`, and checks the recovered loadings are within
0.1 and that the known alpha survives in the residual rather than being absorbed into
"explained return". `test_combined_weights_are_dollar_neutral_and_unit_gross` checks the
5-signal combined portfolio keeps the same unit-gross, (near-)dollar-neutral property the
3-survivor portfolio already has.

**5. Leakage detector.** `scripts/run_leakage_check.py` builds a 500-entity,
2,016-day vintage store with restatements, then drives all 7 named bugs in
`xsalpha/leakage_bugs.py` through `LeakageDetector.guarded_as_of` and
confirms every one raises, plus that a correctly-written read
(`query_date == simulation_clock`) is not flagged:

```
7 of 7 seeded leakage bugs caught
caught: bug_1_use_final_restated_value: query_date 2024-09-24 is after simulation clock 2022-10-04
caught: bug_2_off_by_one_day_ahead: query_date 2022-10-05 is after simulation clock 2022-10-04
caught: bug_3_bfill_across_restatement: query_date 2022-11-03 is after simulation clock 2022-10-04
caught: bug_4_stale_future_cache: query_date 2022-10-19 is after simulation clock 2022-10-04
caught: bug_5_wrong_value_date_future_row: query_date 2022-10-05 is after simulation clock 2022-10-04
caught: bug_6_global_max_knowledge_join: query_date 2024-09-24 is after simulation clock 2022-10-04
caught: bug_7_wall_clock_today: query_date 2026-10-05 is after simulation clock 2022-10-04
clean read (query_date == simulation_clock): no false positive
```

**6. Pinned reproducibility.** `scripts/run_pinned_reproducibility.py` hashes
the vintage store's full content and every `.py` file in `xsalpha/` into a
manifest alongside a seed, computes a small statistic (row count, mean, std,
a seeded resample mean) from the as-of panel, and checks a second,
independent build of the same vintage store under the same manifest
reproduces every field exactly, while a different seed changes the seeded
resample mean (so the check is not vacuously true):

```
exact match across two independent runs of the same manifest: True
a different seed changes the seeded resample mean: True
```

**7. Alt-data quarterly cadence.** `test_altdata_signals_are_piecewise_constant_per_quarter`
checks that both alt-data signals' cross-sectional rank is identical across every day of
a 55-trading-day span known to sit entirely inside one simulated quarter, and
`test_altdata_signal_changes_across_quarter_boundary` checks the rank does change across
a quarter boundary, so the invariant is not vacuously true from an inert signal.

**8. Real-screen oracle diff and stepdown properties.** `tests/test_real_screen.py`
diffs `testing.ols_ar1_tstat` (vectorized) against `oracle.ols_ar1_tstat_oracle`
(hand-rolled, no numpy) for all 44 real series, max abs diff `3.55e-15`
(`docs/real_screen_output.txt`'s `oracle_vs_fast_path_max_abs_tstat_diff`).
`test_romano_wolf_survivors_are_a_subset_of_naive_winners` checks the
stepdown can only remove naive winners, never add one the uncorrected test
missed; `test_stepdown_critical_value_is_non_decreasing` exercises the
monotonicity rule directly on a hand-built 3-series case with two
correlated series and one independent one.

```
26 passed in 18.71s
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

**Sector, size and momentum exposure explained 7.5% of the combined
portfolio's gross return, far under the 62% target, and the residual Sharpe
(6.37 gross, 6.18 net at $250M) came in far above the targeted 1.62/0.38,
both for the same underlying reason.** This portfolio's edge is entirely
built from `earn_yield` and noisy re-reads of it; `earn_yield`'s per-stock
value tilt (`ey_char` in `xsalpha/simulate.py`) is drawn independently of
every other characteristic in the generator, with no designed relationship
to firm size, trailing momentum, or sector membership. A clean, by-
construction-neutral signal set like this one has almost nothing for a
sector/size/momentum regression to find: the sector factors contribute
~0 by construction (the portfolio is already sector-neutral), and size and
momentum pick up only the small, coincidental correlation that 1,756 days
of a slow-moving value tilt happens to have with those factors in this
particular simulated history. Real cross-sectional value signals do carry
a structural tilt toward smaller, less-followed names that this clean
synthetic panel does not encode, which is the most likely reason the real-
world 62% figure does not reproduce here; this is reported as measured
rather than adjusted by injecting a designed size/momentum correlation
after seeing the shortfall, which would be exactly the kind of after-the-
fact tuning this playbook forbids. The residual Sharpe overshoot follows
directly: if factors explain almost none of the return, the residual is
almost the whole, very strong 6.76 combined gross Sharpe, barely dented by
trading costs because the composite is still dominated by the original,
slow-turning survivor signals (see the capacity-sweep finding above);
adding two quarterly-cadence alt-data signals did not materially change
that turnover profile.

**13 of 44 real datasets looked significant under a naive 5% test; the
Romano-Wolf stepdown at 10% family-wise error cut that to 4.** Both numbers
come from one fixed specification (quarter-over-quarter growth, one-lag
AR(1), two-sided test) decided before any series was scored. The 4
survivors (`CSUSHPISA`, home prices; `DRSFRMACBS`, mortgage delinquency;
`IPG2211S`, electric-power industrial production; `REVOLSL`, revolving
consumer credit) are real series with genuine, slow-moving quarter-to-
quarter persistence; the 9 naive winners that did not survive (including
both crude-oil benchmarks, which move together) are exactly the kind of
correlated near-misses a joint bootstrap is supposed to catch and a
Bonferroni correction would not.

**The survivor count was sensitive to Monte Carlo noise at 2,000 bootstrap
replications, and this was caught before being reported.** The first run,
at the seed fixed in `config.SEED`, gave 2 survivors at `N_BOOT=2000`;
re-running at 5 different seeds and `N_BOOT=2000` gave 4 survivors in 4 of
those 5 cases, pointing at Monte Carlo noise near the FWER cutoff rather
than a real, seed-dependent answer. Raising `N_BOOT` to 5000 gave 4
survivors at every one of the 6 seeds tried, including the one that had
given 2 at the lower replication count; 5000 is what is reported below.
This is a real measurement decision (more bootstrap draws reduce estimator
noise), not a search for a seed that produces a target count.

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
| Point-in-time vintage store holding only what was known at each rebalance | yes | **yes**: `VintageStore.as_of` filters strictly on `knowledge_date <= query_date`; checked directly over the full 500 x 2,016 simulated series |
| Leakage detector fails any run reading a later vintage | yes | **yes**, 7 of 7 seeded bugs caught (see Validation 5), 0 false positives on a clean read |
| Seeded leakage bugs caught | 7 of 7 | **7 of 7** |
| Pinned runs reproduce exactly | yes | **yes**: two independent builds of the same vintage content under the same manifest produce bit-identical statistics; a different seed changes the result (see Validation 6) |
| Both alt-data signals combined with the 63 formulaic signals | yes | **yes**: 3 deflated-Sharpe survivors + 2 alt-data signals = 5 signals combined, evaluated over 1,756 test days |
| Simulated 500-name, 8-year daily panel | 500 names, 8 years | **500 names, 2,016 trading days (~8.0 years)**, same panel as above |
| Purged walk-forward splits | yes | **yes**, same 8 purged folds as the 63-signal mining |
| Sector, size and momentum exposure explains combined gross return | 62% | **7.5%** (return-based attribution; see Findings for the root cause) |
| Residual Sharpe gross to net of a half-spread and square-root impact at $250M | 1.62 to 0.38 | **6.37 gross to 6.18 net** (see Findings) |

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

**Real-screen measured results.** Same machine as above. 44 real FRED series, resampled to
quarterly, 40 common quarters (2016 Q3 to 2026 Q2), one AR(1) test each.

| Claim | Target | Measured |
|---|---|---|
| Real release-stamped public series across energy, freight, housing, labor, credit | 44 | **44** (9 energy, 8 freight/goods-movement, 9 housing, 9 labor, 9 credit) |
| One pre-registered specification and one test per dataset | yes | **yes**: quarter-over-quarter growth, OLS AR(1), one two-sided t-test each, fixed in `config.py` before any series was scored |
| Datasets surviving a Romano-Wolf stepdown at 10% family-wise error | 3 | **4** (`CSUSHPISA`, `DRSFRMACBS`, `IPG2211S`, `REVOLSL`) |
| Naive per-test winners without the correction | 11 | **13** |
| A one-page card per dataset stating its specification and result | yes | **yes**, `docs/dataset_cards/<SID>.md`, 44 of 44 generated |

## Building and running

```bash
# from a Python 3.12 venv with requirements.txt installed
python scripts/run_signal_mining.py     # ~226s, writes docs/benchmark_output.txt, data/survivors.json
python scripts/run_capacity_sweep.py    # ~1s, appends to docs/benchmark_output.txt
python scripts/reference_oracle_check.py  # ~1s, writes docs/reference_oracle_output.txt
python scripts/run_leakage_check.py       # ~14s, writes docs/leakage_check_output.txt
python scripts/run_pinned_reproducibility.py  # ~14s, writes docs/pinned_reproducibility_output.txt
python scripts/run_combination_attribution.py  # ~1s, writes docs/attribution_output.txt
python -m pytest tests -v               # ~21s, 19 tests
```

`data/` is gitignored (the DuckDB panel file and the survivor list, about 59
MB), regenerated by `run_signal_mining.py`.

**Real-screen data.** `data/real_dataset_screen_panel.csv` (about 33 KB, 40 quarters x 44
series' growth rates) and `docs/dataset_cards/` (44 small markdown files, about 56 KB total) are
small enough to commit directly, the same precedent as this repository's other committed
artifacts. The 44 raw per-series FRED downloads behind the panel are not committed:

```bash
# no API key needed; one CSV per series id in realscreen/config.py::SERIES
curl -o <SID>.csv "https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SID>"

python scripts/ingest_real_datasets.py --raw-dir <dir of 44 CSVs> --out-csv data/real_dataset_screen_panel.csv
python scripts/run_real_screen.py data/real_dataset_screen_panel.csv > docs/real_screen_output.txt
```

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
- The vintage store models one field's restatement behavior and is not
  wired into the 63-signal mining pipeline; the leakage detector is proven
  against the vintage store directly, not by re-running signal mining under
  simulated restatements.
- The leakage detector's guard is only as good as every call site choosing
  to go through `guarded_as_of`; nothing prevents a future call site from
  calling `VintageStore.as_of` directly and bypassing it, which is why the 7
  seeded bugs are written as if they already go through the guard rather
  than as an end-to-end test of an entire backtest loop.
- Both alt-data signal proxies are noisy re-reads of this panel's own
  `earn_yield` field, not the actual datasets measured in
  [`point-in-time-activity-index`](https://github.com/Manas103/point-in-time-activity-index)'s
  `nowcast/` extension or the separate Filing-Language Signal project; see
  "Honest framing, up front".
- The factor attribution explained 7.5% of the combined portfolio's gross
  return, far short of the 62% target, because this panel's value
  characteristic is drawn independently of size, momentum and sector; see
  Findings for why this was not corrected by adding a designed correlation.
  The residual Sharpe (6.37 gross, 6.18 net at $250M) is correspondingly far
  above the targeted 1.62/0.38, for the same reason.
- Sector factors are reconstructed empirically (the equal-weight realized
  return of each sector's own stocks), not read from `simulate_panel`'s
  internal `sector_ret` array, which the function does not expose.
- `realscreen/`'s naive-winner count (13) and Romano-Wolf survivor count (4)
  both overshoot the resume's targeted 11 and 3; both are reported as
  measured, not adjusted. 8 of the 44 "freight" series are really
  trade/inventory-flow proxies, not freight-rate indices, because FRED's
  own freight-specific series ran out at 6; see "Honest framing, up front".
  Every series is forced onto one common quarterly grid regardless of its
  real reporting frequency, a declared simplification. The Romano-Wolf
  critical value is bootstrap-estimated (5,000 replications after 2,000
  proved sensitive to the seed near the cutoff; see Findings), not a
  closed-form quantity, so it carries residual Monte Carlo noise even at
  5,000 draws.
