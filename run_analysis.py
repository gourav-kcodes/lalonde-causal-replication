import json
import pandas as pd

from src.data_loader import load_experimental, load_observational, true_experimental_att
from src.data_loader import COVARIATES, TREATMENT, OUTCOME
from src.balance import balance_table
from src.propensity import estimate_propensity, trim_common_support, nearest_neighbor_match, att_from_matches
from src.estimators import naive_att, ipw_att, doubly_robust_att, bootstrap_ci
from src.visualize import love_plot, propensity_overlap_plot, results_comparison_plot

RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"


def main():
    exp = load_experimental()
    obs = load_observational()
    true_att = true_experimental_att(exp)
    print(f"True experimental ATT (benchmark): ${true_att:,.2f}\n")

    # --- 1. Naive estimate -------------------------------------------------
    naive = naive_att(obs, TREATMENT, OUTCOME)
    naive_ci = bootstrap_ci(lambda d: naive_att(d, TREATMENT, OUTCOME), obs)
    print(f"Naive ATT:          ${naive:,.2f}   95% CI {naive_ci}")

    balance_before = balance_table(obs, COVARIATES, TREATMENT)
    balance_before.to_csv(f"{RESULTS_DIR}/balance_before.csv", index=False)

    # --- 2. Propensity scores + common support ------------------------------
    pscore_full = estimate_propensity(obs, COVARIATES, TREATMENT)
    obs_trimmed, pscore = trim_common_support(obs, pscore_full, TREATMENT)
    print(f"\nDropped {len(obs) - len(obs_trimmed)} units outside common support "
          f"({len(obs_trimmed)} remain)")

    propensity_overlap_plot(obs_trimmed, pscore, TREATMENT, f"{FIGURES_DIR}/propensity_overlap.png")

    # --- 3. Propensity score matching ---------------------------------------
    matches = nearest_neighbor_match(obs_trimmed, pscore, TREATMENT, caliper=0.01)
    psm_att = att_from_matches(obs_trimmed, matches, OUTCOME)
    print(f"\nMatched {len(matches)} / {obs_trimmed[TREATMENT].sum()} treated units")
    print(f"PSM ATT:             ${psm_att:,.2f}")

    matched_ids = pd.unique(pd.concat([matches["treated_idx"], matches["control_idx"]]))
    matched_df = obs_trimmed.loc[matched_ids]
    balance_after = balance_table(matched_df, COVARIATES, TREATMENT)
    balance_after.to_csv(f"{RESULTS_DIR}/balance_after.csv", index=False)
    love_plot(balance_before, balance_after, f"{FIGURES_DIR}/love_plot.png")

    def psm_estimator(d):
        ps_full = estimate_propensity(d, COVARIATES, TREATMENT)
        d_trim, ps_trim = trim_common_support(d, ps_full, TREATMENT)
        m = nearest_neighbor_match(d_trim, ps_trim, TREATMENT, caliper=0.01)
        return att_from_matches(d_trim, m, OUTCOME)

    psm_ci = bootstrap_ci(psm_estimator, obs_trimmed, n_boot=200)
    print(f"PSM 95% CI:          {psm_ci}")

    # --- 4. IPW ---------------------------------------------------------------
    ipw = ipw_att(obs_trimmed, pscore, TREATMENT, OUTCOME)
    print(f"\nIPW ATT:             ${ipw:,.2f}")

    def ipw_estimator(d):
        ps_full = estimate_propensity(d, COVARIATES, TREATMENT)
        d_trim, ps_trim = trim_common_support(d, ps_full, TREATMENT)
        return ipw_att(d_trim, ps_trim, TREATMENT, OUTCOME)

    ipw_ci = bootstrap_ci(ipw_estimator, obs_trimmed, n_boot=200)
    print(f"IPW 95% CI:          {ipw_ci}")

    # --- 5. Doubly robust (AIPW) -----------------------------------------------
    dr = doubly_robust_att(obs_trimmed, pscore, COVARIATES, TREATMENT, OUTCOME)
    print(f"\nDoubly Robust ATT:   ${dr:,.2f}")

    def dr_estimator(d):
        ps_full = estimate_propensity(d, COVARIATES, TREATMENT)
        d_trim, ps_trim = trim_common_support(d, ps_full, TREATMENT)
        return doubly_robust_att(d_trim, ps_trim, COVARIATES, TREATMENT, OUTCOME)

    dr_ci = bootstrap_ci(dr_estimator, obs_trimmed, n_boot=200)
    print(f"DR 95% CI:           {dr_ci}")

    # --- 6. Save everything ------------------------------------------------
    results = {
        "True (experimental)": {"estimate": float(true_att), "ci": [float(true_att), float(true_att)]},
        "Naive": {"estimate": float(naive), "ci": [float(c) for c in naive_ci]},
        "PSM": {"estimate": float(psm_att), "ci": [float(c) for c in psm_ci]},
        "IPW": {"estimate": float(ipw), "ci": [float(c) for c in ipw_ci]},
        "Doubly Robust": {"estimate": float(dr), "ci": [float(c) for c in dr_ci]},
    }

    with open(f"{RESULTS_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(results).T.to_csv(f"{RESULTS_DIR}/results_table.csv")

    plot_results = {k: v for k, v in results.items() if k != "True (experimental)"}
    results_comparison_plot(plot_results, true_att, f"{FIGURES_DIR}/results_comparison.png")

    print("\nDone. Results saved to results/, figures saved to results/figures/")


if __name__ == "__main__":
    main()
