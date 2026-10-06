# JTSJOL: Job openings, total nonfarm

Category: labor

## Pre-registered specification

Quarter-over-quarter percent growth, OLS AR(1) (`growth_t ~ const + beta * growth_{t-1}`), two-sided test of `beta = 0` against the random-walk null, identical for all 44 datasets in this screen.

## Result

- t-statistic: **2.9058**
- beta (AR(1) coefficient): 0.4320
- naive p-value: 0.0062
- naive winner at alpha=0.05: **yes**
- survives Romano-Wolf stepdown at FWER=0.1: **no**
