# AWHAETP: Average weekly hours, production employees

Category: labor

## Pre-registered specification

Quarter-over-quarter percent growth, OLS AR(1) (`growth_t ~ const + beta * growth_{t-1}`), two-sided test of `beta = 0` against the random-walk null, identical for all 44 datasets in this screen.

## Result

- t-statistic: **-0.1221**
- beta (AR(1) coefficient): -0.0203
- naive p-value: 0.9035
- naive winner at alpha=0.05: **no**
- survives Romano-Wolf stepdown at FWER=0.1: **no**
