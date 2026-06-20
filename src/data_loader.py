"""
Data loading for the LaLonde (1986) causal inference replication.

Two comparison groups are used against the same treated group (NSW job
trainees), which is the whole point of the design:

1. Experimental control (NSW): randomized, gives the "true" benchmark ATT.
2. Observational control (CPS-1): a large non-experimental comparison group,
   used to simulate what an analyst sees when there's no randomization.

Source: Dehejia & Wahba (1999) reformatted data, distributed via the
`causaldata` package (Cunningham, "Causal Inference: The Mixtape").
"""

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
    """NSW treated + CPS-1 non-experimental comparison group. This is the
    dataset an analyst would actually have in a real observational study —
    no randomization, so confounding is a live problem."""
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
