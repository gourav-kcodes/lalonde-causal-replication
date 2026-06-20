"""
Causal effect estimators.

Three estimators of the Average Treatment Effect on the Treated (ATT),
in increasing order of sophistication:

- Naive difference in means: ignores confounding entirely.
- IPW: reweights controls by their odds of being treated, so a control
  that looked like a likely treatment candidate counts more.
- AIPW (augmented IPW / doubly robust): combines IPW with an outcome
  regression model. It stays consistent if *either* the propensity
  model or the outcome model is correctly specified, not just one —
  hence "doubly robust".
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def naive_att(df: pd.DataFrame, treatment_col: str, outcome_col: str) -> float:
    treated = df[df[treatment_col] == 1][outcome_col]
    control = df[df[treatment_col] == 0][outcome_col]
    return float(treated.mean() - control.mean())


def ipw_att(df: pd.DataFrame, pscore: pd.Series, treatment_col: str, outcome_col: str) -> float:
    """ATT-weighted IPW: treated units get weight 1, controls get
    weight ps / (1 - ps), so the weighted control group is reshaped to
    resemble the treated group's covariate distribution."""
    treat = df[treatment_col].values
    y = df[outcome_col].values
    ps = pscore.loc[df.index].values

    weights = np.where(treat == 1, 1.0, ps / (1 - ps))

    weighted_treated = np.average(y[treat == 1], weights=weights[treat == 1])
    weighted_control = np.average(y[treat == 0], weights=weights[treat == 0])
    return float(weighted_treated - weighted_control)


def doubly_robust_att(
    df: pd.DataFrame, pscore: pd.Series, covariates: list, treatment_col: str, outcome_col: str
) -> float:
    """AIPW estimator for the ATT (Lunceford & Davidian, 2004).

    Fits an outcome model on the control group only, to predict what each
    unit's earnings would have been *without* training — including for
    treated units, where that's the missing counterfactual. The first term
    below is the average gap between treated units' real outcomes and that
    counterfactual; the second term corrects for any remaining imbalance by
    reweighting control residuals with the IPW weight. If either the outcome
    model or the propensity model is roughly right, the estimate stays
    consistent — that's the "doubly robust" part.
    """
    treat = df[treatment_col].values
    y = df[outcome_col].values
    X = df[covariates].values
    ps = pscore.loc[df.index].values

    control_model = LinearRegression().fit(X[treat == 0], y[treat == 0])
    mu0_hat = control_model.predict(X)

    n_treated = treat.sum()
    treated_term = np.sum(treat * (y - mu0_hat)) / n_treated
    control_correction = np.sum((1 - treat) * (ps / (1 - ps)) * (y - mu0_hat)) / n_treated

    return float(treated_term - control_correction)


def bootstrap_ci(estimator_fn, df: pd.DataFrame, n_boot: int = 500, seed: int = 42) -> tuple:
    """95% bootstrap confidence interval for any estimator function that
    takes a dataframe and returns a single ATT estimate. Resampling the
    whole dataset (not separately by group) keeps the treated/control
    ratio realistic in each replicate."""
    rng = np.random.default_rng(seed)
    estimates = []
    n = len(df)
    for _ in range(n_boot):
        sample_idx = rng.choice(df.index, size=n, replace=True)
        boot_df = df.loc[sample_idx].reset_index(drop=True)
        try:
            estimates.append(estimator_fn(boot_df))
        except Exception:
            continue
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)
