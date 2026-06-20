import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10


def love_plot(balance_before: pd.DataFrame, balance_after: pd.DataFrame, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    before = balance_before.set_index("covariate")["smd"]
    after = balance_after.set_index("covariate")["smd"].reindex(before.index)

    y_pos = range(len(before))
    ax.scatter(before.values, y_pos, color="#d62728", label="Before matching", zorder=3)
    ax.scatter(after.values, y_pos, color="#2ca02c", label="After matching", zorder=3)

    for y, (b, a) in enumerate(zip(before.values, after.values)):
        ax.plot([b, a], [y, y], color="gray", linewidth=0.8, zorder=1)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.1, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(-0.1, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(before.index)
    ax.set_xlabel("Standardized Mean Difference")
    ax.set_title("Covariate Balance: Before vs. After Matching")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def propensity_overlap_plot(df: pd.DataFrame, pscore: pd.Series, treatment_col: str, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    treated_ps = pscore[df[treatment_col] == 1]
    control_ps = pscore[df[treatment_col] == 0]

    ax.hist(control_ps, bins=40, alpha=0.6, label="Control (CPS)", color="#1f77b4", density=True)
    ax.hist(treated_ps, bins=40, alpha=0.6, label="Treated (NSW)", color="#ff7f0e", density=True)

    ax.set_xlabel("Estimated Propensity Score")
    ax.set_ylabel("Density")
    ax.set_title("Propensity Score Distribution by Group")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def results_comparison_plot(results: dict, true_att: float, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    labels = list(results.keys())
    values = [results[k]["estimate"] for k in labels]
    lowers = [results[k]["estimate"] - results[k]["ci"][0] for k in labels]
    uppers = [results[k]["ci"][1] - results[k]["estimate"] for k in labels]

    ax.barh(labels, values, xerr=[lowers, uppers], color="#4c72b0", capsize=4)
    ax.axvline(true_att, color="#d62728", linestyle="--", label=f"True experimental ATT (${true_att:,.0f})")
    ax.axvline(0, color="black", linewidth=0.8)

    ax.set_xlabel("Estimated ATT ($)")
    ax.set_title("Causal Estimates vs. the True Experimental Benchmark")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
