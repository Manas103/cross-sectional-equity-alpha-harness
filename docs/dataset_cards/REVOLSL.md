# REVOLSL: Revolving consumer credit outstanding

Category: credit

## Pre-registered specification

Quarter-over-quarter percent growth, OLS AR(1) (`growth_t ~ const + beta * growth_{t-1}`), two-sided test of `beta = 0` against the random-walk null, identical for all 44 datasets in this screen.

## Result

- t-statistic: **4.1015**
- beta (AR(1) coefficient): 0.5584
- naive p-value: 0.0002
- naive winner at alpha=0.05: **yes**
- survives Romano-Wolf stepdown at FWER=0.1: **yes**
