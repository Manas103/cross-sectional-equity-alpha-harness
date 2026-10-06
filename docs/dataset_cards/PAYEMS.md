# PAYEMS: Total nonfarm payroll employment

Category: labor

## Pre-registered specification

Quarter-over-quarter percent growth, OLS AR(1) (`growth_t ~ const + beta * growth_{t-1}`), two-sided test of `beta = 0` against the random-walk null, identical for all 44 datasets in this screen.

## Result

- t-statistic: **-0.7002**
- beta (AR(1) coefficient): -0.1143
- naive p-value: 0.4882
- naive winner at alpha=0.05: **no**
- survives Romano-Wolf stepdown at FWER=0.1: **no**
