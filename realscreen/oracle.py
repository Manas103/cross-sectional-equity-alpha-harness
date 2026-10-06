"""Independent, pure-Python recomputation of the per-series AR(1) OLS
t-statistic: explicit sums, no numpy, no scipy, sharing no code with
`testing.ols_ar1_tstat`. `scripts/run_real_screen.py` diffs every one of
the 44 series' t-statistics from this module against the fast path.
"""

from __future__ import annotations

import math


def ols_ar1_tstat_oracle(y: list[float]) -> float:
    x = y[:-1]
    yt = y[1:]
    n = len(yt)

    sum_x = sum(x)
    sum_y = sum(yt)
    sum_xx = sum(xi * xi for xi in x)
    sum_xy = sum(xi * yi for xi, yi in zip(x, yt))

    denom = n * sum_xx - sum_x * sum_x
    beta = (n * sum_xy - sum_x * sum_y) / denom
    alpha = (sum_y - beta * sum_x) / n

    resid = [yt[i] - (alpha + beta * x[i]) for i in range(n)]
    dof = n - 2
    sigma2 = sum(r * r for r in resid) / dof

    mean_x = sum_x / n
    sxx = sum((xi - mean_x) ** 2 for xi in x)
    se_beta = math.sqrt(sigma2 / sxx)

    return beta / se_beta if se_beta > 0 else 0.0
