import pandas as pd
from causaldata import nsw_mixtape, cps_mixtape

COVARIATES = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]
OUTCOME = "re78"
TREATMENT = "treat"


def load_experimental() -> pd.DataFrame:
    """NSW treated + NSW experimental control. Randomized, so a simple
    difference in means is already an unbiased ATT estimate."""
    df = nsw_mixtape.load_pandas().data
    return df[[TREATMENT, *COVARIATES, OUTCOME]].copy()


def load_observational() -> pd.DataFrame:
    nsw = nsw_mixtape.load_pandas().data
    cps = cps_mixtape.load_pandas().data

    treated = nsw[nsw[TREATMENT] == 1]
    df = pd.concat([treated, cps], ignore_index=True)
    return df[[TREATMENT, *COVARIATES, OUTCOME]].copy()


def true_experimental_att(df_exp: pd.DataFrame) -> float:
    """The benchmark every causal estimator in this project is judged against."""
    treated = df_exp[df_exp[TREATMENT] == 1][OUTCOME]
    control = df_exp[df_exp[TREATMENT] == 0][OUTCOME]
    return treated.mean() - control.mean()


if __name__ == "__main__":
    exp = load_experimental()
    obs = load_observational()
    print(f"Experimental sample: {len(exp)} rows ({exp[TREATMENT].sum()} treated)")
    print(f"Observational sample: {len(obs)} rows ({obs[TREATMENT].sum()} treated)")
    print(f"True experimental ATT: ${true_experimental_att(exp):,.2f}")
