"""
Sanity checks using synthetic data with a known, hand-built treatment
effect. If an estimator can't recover a treatment effect we built into
the data ourselves, it has no business being trusted on real data.
"""

import numpy as np
import pandas as pd
import pytest

from src.estimators import naive_att, ipw_att, doubly_robust_att
from src.propensity import estimate_propensity, trim_common_support, nearest_neighbor_match, att_from_matches
from src.balance import standardized_mean_diff

TRUE_EFFECT = 500.0


def make_synthetic_data(n=2000, seed=0):
    """Treatment assignment depends on x1 (confounded), and the outcome
    depends on x1 too — so naive comparison is biased on purpose, the
    same way the real LaLonde/CPS comparison is."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)

    propensity = 1 / (1 + np.exp(-(0.8 * x1)))
    treat = rng.binomial(1, propensity)

    baseline = 1000 + 300 * x1 + 100 * x2
    y = baseline + TRUE_EFFECT * treat + rng.normal(0, 50, n)

    return pd.DataFrame({"x1": x1, "x2": x2, "treat": treat, "y": y})


@pytest.fixture
def synthetic_df():
    return make_synthetic_data()


def test_naive_is_biased(synthetic_df):
    """With confounding baked in, naive comparison should NOT match the
    true effect — if it does, the synthetic data isn't testing anything."""
    estimate = naive_att(synthetic_df, "treat", "y")
    assert abs(estimate - TRUE_EFFECT) > 50


def test_ipw_recovers_true_effect(synthetic_df):
    pscore = estimate_propensity(synthetic_df, ["x1", "x2"], "treat")
    trimmed, ps = trim_common_support(synthetic_df, pscore, "treat")
    estimate = ipw_att(trimmed, ps, "treat", "y")
    assert abs(estimate - TRUE_EFFECT) < 50


def test_doubly_robust_recovers_true_effect(synthetic_df):
    pscore = estimate_propensity(synthetic_df, ["x1", "x2"], "treat")
    trimmed, ps = trim_common_support(synthetic_df, pscore, "treat")
    estimate = doubly_robust_att(trimmed, ps, ["x1", "x2"], "treat", "y")
    assert abs(estimate - TRUE_EFFECT) < 50


def test_matching_recovers_true_effect(synthetic_df):
    pscore = estimate_propensity(synthetic_df, ["x1", "x2"], "treat")
    trimmed, ps = trim_common_support(synthetic_df, pscore, "treat")
    matches = nearest_neighbor_match(trimmed, ps, "treat", caliper=0.02)
    estimate = att_from_matches(trimmed, matches, "y")
    assert abs(estimate - TRUE_EFFECT) < 75


def test_balance_improves_after_matching(synthetic_df):
    """The whole point of matching: imbalance on x1 should shrink a lot."""
    pscore = estimate_propensity(synthetic_df, ["x1", "x2"], "treat")
    trimmed, ps = trim_common_support(synthetic_df, pscore, "treat")
    smd_before = abs(standardized_mean_diff(trimmed, "x1", "treat"))

    matches = nearest_neighbor_match(trimmed, ps, "treat", caliper=0.02)
    matched_ids = pd.unique(pd.concat([matches["treated_idx"], matches["control_idx"]]))
    matched_df = trimmed.loc[matched_ids]
    smd_after = abs(standardized_mean_diff(matched_df, "x1", "treat"))

    assert smd_after < smd_before
