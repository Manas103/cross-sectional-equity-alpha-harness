# DTCOLNVHFNM: Motor vehicle loans outstanding

Category: credit

## Pre-registered specification

Quarter-over-quarter percent growth, OLS AR(1) (`growth_t ~ const + beta * growth_{t-1}`), two-sided test of `beta = 0` against the random-walk null, identical for all 44 datasets in this screen.

## Result

- t-statistic: **0.7179**
- beta (AR(1) coefficient): 0.1173
- naive p-value: 0.4773
- naive winner at alpha=0.05: **no**
- survives Romano-Wolf stepdown at FWER=0.1: **no**
