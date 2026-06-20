import numpy as np
import pandas as pd


def standardized_mean_diff(
    df: pd.DataFrame, covariate: str, treatment_col: str, weights: pd.Series = None
) -> float:
    """SMD between treated and control on one covariate, optionally weighted
    (weighted version is used to check balance after IPW)."""
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    if weights is None:
        t_mean, c_mean = treated[covariate].mean(), control[covariate].mean()
        pooled_sd = np.sqrt((treated[covariate].var() + control[covariate].var()) / 2)
    else:
        w_t = weights.loc[treated.index]
        w_c = weights.loc[control.index]
        t_mean = np.average(treated[covariate], weights=w_t)
        c_mean = np.average(control[covariate], weights=w_c)
        pooled_sd = np.sqrt((treated[covariate].var() + control[covariate].var()) / 2)

    if pooled_sd == 0:
        return 0.0
    return (t_mean - c_mean) / pooled_sd


def balance_table(
    df: pd.DataFrame, covariates: list, treatment_col: str, weights: pd.Series = None
) -> pd.DataFrame:
    """SMD for every covariate, sorted by absolute imbalance (worst first)."""
    rows = []
    for cov in covariates:
        smd = standardized_mean_diff(df, cov, treatment_col, weights)
        rows.append({"covariate": cov, "smd": smd, "abs_smd": abs(smd)})
    return pd.DataFrame(rows).sort_values("abs_smd", ascending=False).reset_index(drop=True)


"""
Covariate balance diagnostics.

The standardized mean difference (SMD) is the standard way to check whether
treated and control groups look alike on observed covariates. As a rule of
thumb, |SMD| < 0.1 is considered well balanced; the naive observational
comparison in this project blows way past that, which is the whole problem
propensity score methods are trying to fix.
"""

