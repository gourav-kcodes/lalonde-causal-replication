"""
Propensity score estimation and nearest-neighbor matching.

The propensity score is P(treated | covariates). Two units with similar
propensity scores looked similarly likely to end up treated, so matching on
the score (rather than on every raw covariate) is a practical way to build
a comparable control group out of the much larger, much-less-similar CPS
sample.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def estimate_propensity(df: pd.DataFrame, covariates: list, treatment_col: str) -> pd.Series:
    """Logistic regression propensity model. Simple on purpose — for this
    sample size a flexible model (e.g. gradient boosting) tends to overfit
    and produce scores near 0 or 1, which destroys common support.

    Covariates here mix raw dollar amounts (earnings, in the thousands)
    with 0/1 flags, so they're standardized first — without that, lbfgs
    doesn't converge cleanly and the dollar-valued features end up
    dominating the fit just because of their scale, not their actual
    relevance.
    """
    X = StandardScaler().fit_transform(df[covariates].values)
    y = df[treatment_col].values
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    scores = model.predict_proba(X)[:, 1]
    return pd.Series(scores, index=df.index, name="propensity_score")


def trim_common_support(df: pd.DataFrame, pscore: pd.Series, treatment_col: str) -> pd.DataFrame:
    """Drop units outside the region where both groups overlap in propensity
    score. Without this, matches at the edges are comparing units that
    aren't really comparable, which quietly reintroduces bias."""
    treated_scores = pscore[df[treatment_col] == 1]
    control_scores = pscore[df[treatment_col] == 0]
    lower = max(treated_scores.min(), control_scores.min())
    upper = min(treated_scores.max(), control_scores.max())
    keep = pscore[(pscore >= lower) & (pscore <= upper)].index
    return df.loc[keep].copy(), pscore.loc[keep]


def nearest_neighbor_match(
    df: pd.DataFrame, pscore: pd.Series, treatment_col: str, caliper: float = 0.01
) -> pd.DataFrame:
    """For every treated unit, find its nearest control by propensity score
    (matching with replacement, so one control can match multiple treated
    units). Matches further apart than `caliper` are dropped rather than
    forced — a bad match is worse than no match."""
    treated_idx = df[df[treatment_col] == 1].index
    control_idx = df[df[treatment_col] == 0].index

    control_scores = pscore.loc[control_idx].values.reshape(-1, 1)
    nn = NearestNeighbors(n_neighbors=1).fit(control_scores)

    treated_scores = pscore.loc[treated_idx].values.reshape(-1, 1)
    distances, neighbor_pos = nn.kneighbors(treated_scores)

    matches = []
    for t_idx, dist, pos in zip(treated_idx, distances.ravel(), neighbor_pos.ravel()):
        if dist <= caliper:
            matches.append({"treated_idx": t_idx, "control_idx": control_idx[pos], "distance": dist})

    return pd.DataFrame(matches)


def att_from_matches(df: pd.DataFrame, matches: pd.DataFrame, outcome_col: str) -> float:
    """ATT = average outcome gap between each treated unit and its matched
    control."""
    treated_outcomes = df.loc[matches["treated_idx"], outcome_col].values
    control_outcomes = df.loc[matches["control_idx"], outcome_col].values
    return float(np.mean(treated_outcomes - control_outcomes))
