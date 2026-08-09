Bayesian Online Change-Point Detection for Measles Surveillance

This repository implements a Bayesian Online Change-Point Detection (BOCD) framework for annual measles surveillance data.

The project extends a classical CUSUM-based change-point analysis toward sequential Bayesian predictive inference, providing a probabilistic representation of the time since the most recent change-point rather than a binary alarm.

Research motivation

Classical change-point methods such as CUSUM can identify unusual observations using a threshold-based decision rule. However, they do not directly provide a posterior distribution over possible change locations.

BOCD addresses this from a Bayesian perspective by maintaining, at every time point, a posterior distribution over the run length:

«the number of observations since the most recent change-point.»

This provides a richer representation of uncertainty and allows quantities such as the posterior expected run length and MAP run length to be examined over time.

The project is motivated by broader research interests in:

- Bayesian predictive inference
- uncertainty-aware machine learning
- sequential statistical inference
- change-point detection
- structured and longitudinal data
- probabilistic machine learning

Data

The analysis uses annual measles surveillance counts for 2010–2025.

The data are stored separately in:

data/measles_annual_counts.csv

with the following structure:

year,cases
2010,63
2011,220
...
2025,2289

Keeping the data separate from the analysis code makes the workflow easier to inspect and reproduce.

Method

The implementation follows the Bayesian Online Change-Point Detection framework introduced by:

Adams, R. P., & MacKay, D. J. C. (2007).
Bayesian Online Changepoint Detection.
arXiv:0710.3742.

For the current implementation, the observed surveillance counts are transformed using:

x_t = log(cases_t)

A conjugate Normal-Gamma predictive model is then used for the transformed observations.

After marginalization, the predictive distribution is Student-t.

At each time point, the algorithm updates:

P(r_t | x_1, ..., x_t)

where "r_t" denotes the current run length.

The implementation performs the recursive calculations in log space to improve numerical stability.

Main outputs

The analysis produces several posterior summaries.

Expected run length

E[r_t | x_1:t]

The posterior expected time since the most recent change-point.

A sharp decrease can indicate increasing posterior support for a recently initiated regime.

MAP run length

The most probable run length under the posterior distribution:

argmax_r P(r_t | x_1:t)

Run-length-zero posterior

The posterior probability associated with a run length of zero is also retained.

Because the current implementation uses a constant hazard function, this quantity should not be interpreted as an independent data-driven change-point score. The full run-length posterior and its summaries are therefore the primary diagnostics.

Project structure

bayesian-online-change-point-detection/
│
├── bocd.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── data/
    └── measles_annual_counts.csv

Installation

Clone the repository and install the required Python packages:

pip install -r requirements.txt

Running the analysis

Run:

python bocd.py

The script loads the measles surveillance data, applies the log transformation, runs BOCD, and prints the posterior run-length summaries.

Prior specification

The current implementation uses a weakly informative Normal-Gamma prior.

The prior mean and variance are initialized using only the first five observations of the series.

This avoids constructing the prior using the complete time series, including future observations.

The prior expected run length is controlled through the hazard parameter:

lam = 6.0

Sensitivity to this assumption will be examined in a subsequent analysis.

Limitations

The current implementation is intentionally a methodological starting point rather than a final disease-surveillance model.

Important limitations include:

1. The surveillance counts are modeled after a log transformation using a continuous Normal-Gamma predictive model rather than a count-specific likelihood.

2. The current hazard function is constant.

3. The dataset is relatively short, containing annual observations from 2010 through 2025.

4. The choice of prior and expected run length can influence posterior inference.

5. The current implementation does not yet include formal simulation-based calibration or false-positive evaluation.

These limitations motivate the planned extensions below.

Planned extensions

Future versions of the project will include:

- comparison with the existing CUSUM implementation
- side-by-side visualization of CUSUM and BOCD results
- simulation-based validation
- false-positive analysis
- sensitivity analysis for the hazard parameter
- investigation of alternative predictive models
- exploration of hierarchical change-point structures
- evaluation on additional surveillance series

A particularly interesting extension is a hierarchical BOCD formulation in which multiple surveillance regions are modeled jointly. This would move the project toward structured, non-exchangeable sequential data and provide a closer connection to modern Bayesian predictive modeling.

Relation to the CUSUM project

This repository is designed as a Bayesian extension of the following project:

"cusum-change-point-detection"

The CUSUM implementation provides a classical threshold-based perspective, while the present repository investigates a sequential Bayesian formulation of the same general change-point problem.

The two approaches are therefore complementary rather than interchangeable:

CUSUM
  │
  └── threshold-based anomaly detection

BOCD
  │
  └── posterior distribution over run length
      │
      ├── expected run length
      └── MAP run length

Reproducibility

The project keeps the data, analysis code, and dependencies explicitly separated.

The intended workflow is:

measles_annual_counts.csv
          │
          ▼
      preprocessing
          │
          ▼
    Normal-Gamma model
          │
          ▼
         BOCD
          │
          ▼
 posterior run-length distribution
          │
          ├── expected run length
          └── MAP run length

The repository is intended as a reproducible research project and as a methodological demonstration of Bayesian sequential inference applied to real-world surveillance data.
    Expected CSV columns
    --------------------
    year
    cases
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        "year",
        "cases",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df = (
        df[
            ["year", "cases"]
        ]
        .copy()
        .sort_values("year")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            "The data file is empty."
        )

    if df["year"].duplicated().any():
        raise ValueError(
            "Duplicate years detected."
        )

    if df["year"].isna().any():
        raise ValueError(
            "Missing year values detected."
        )

    if df["cases"].isna().any():
        raise ValueError(
            "Missing case counts detected."
        )

    if not np.all(
        np.isfinite(
            df["cases"].to_numpy(
                dtype=float
            )
        )
    ):
        raise ValueError(
            "Case counts contain non-finite values."
        )

    if (
        df["cases"] <= 0
    ).any():
        raise ValueError(
            "All case counts must be positive because "
            "the analysis uses log(cases)."
        )

    return df


def preprocess_counts(
    cases: np.ndarray,
) -> np.ndarray:
    """
    Apply the same transformation used in the CUSUM project.

    x_t = log(cases_t)

    The current surveillance series contains only positive counts,
    so the ordinary logarithm is used rather than log1p.
    """

    cases = np.asarray(
        cases,
        dtype=float,
    )

    if cases.ndim != 1:
        raise ValueError(
            "cases must be one-dimensional."
        )

    if len(cases) == 0:
        raise ValueError(
            "cases cannot be empty."
        )

    if not np.all(
        np.isfinite(cases)
    ):
        raise ValueError(
            "cases contains non-finite values."
        )

    if np.any(
        cases <= 0
    ):
        raise ValueError(
            "All case counts must be positive "
            "before log transformation."
        )

    return np.log(cases)


# ---------------------------------------------------------------------
# Normal-Gamma predictive model
# ---------------------------------------------------------------------

class NormalGammaPredictive:
    """
    Normal-Gamma conjugate predictive model.

    Model
    -----
        x | mu, tau ~ Normal(mu, 1/tau)

        tau ~ Gamma(alpha, beta)

    beta is parameterized as the Gamma rate.

    After marginalizing over mu and tau, the predictive distribution
    is Student-t.
    """

    def __init__(
        self,
        mu0: float,
        kappa0: float = 1.0,
        alpha0: float = 2.0,
        beta0: float = 1.0,
    ) -> None:

        if not np.isfinite(mu0):
            raise ValueError(
                "mu0 must be finite."
            )

        if kappa0 <= 0:
            raise ValueError(
                "kappa0 must be positive."
            )

        if alpha0 <= 0:
            raise ValueError(
                "alpha0 must be positive."
            )

        if beta0 <= 0:
            raise ValueError(
                "beta0 must be positive."
            )

        self.mu0 = float(mu0)
        self.kappa0 = float(kappa0)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)

        self.mu = np.array(
            [self.mu0],
            dtype=float,
        )

        self.kappa = np.array(
            [self.kappa0],
            dtype=float,
        )

        self.alpha = np.array(
            [self.alpha0],
            dtype=float,
        )

        self.beta = np.array(
            [self.beta0],
            dtype=float,
        )

    def log_pred_prob(
        self,
        x: float,
    ) -> np.ndarray:
        """
        Calculate the log Student-t predictive density for x under
        every currently active run-length hypothesis.
        """

        df = 2.0 * self.alpha

        scale = np.sqrt(
            self.beta
            * (
                self.kappa + 1.0
            )
            / (
                self.alpha
                * self.kappa
            )
        )

        y = (
            x - self.mu
        ) / scale

        return (
            gammaln(
                (df + 1.0) / 2.0
            )
            - gammaln(
                df / 2.0
            )
            - 0.5
            * np.log(
                df * np.pi
            )
            - np.log(scale)
            - (
                (df + 1.0) / 2.0
            )
            * np.log1p(
                (y ** 2) / df
            )
        )

    def update(
        self,
        x: float,
    ) -> None:
        """
        Update all active Normal-Gamma hypotheses after observing x.

        A new run-length-zero hypothesis is initialized from the prior.
        Existing hypotheses are updated with the new observation.
        """

        new_mu = (
            self.kappa * self.mu
            + x
        ) / (
            self.kappa + 1.0
        )

        new_kappa = (
            self.kappa + 1.0
        )

        new_alpha = (
            self.alpha + 0.5
        )

        new_beta = (
            self.beta
            + (
                self.kappa
                * (
                    x - self.mu
                ) ** 2
            )
            / (
                2.0
                * (
                    self.kappa + 1.0
                )
            )
        )

        self.mu = np.concatenate(
            (
                np.array(
                    [self.mu0]
                ),
                new_mu,
            )
        )

        self.kappa = np.concatenate(
            (
                np.array(
                    [self.kappa0]
                ),
                new_kappa,
            )
        )

        self.alpha = np.concatenate(
            (
                np.array(
                    [self.alpha0]
                ),
                new_alpha,
            )
        )

        self.beta = np.concatenate(
            (
                np.array(
                    [self.beta0]
                ),
                new_beta,
            )
        )


# ---------------------------------------------------------------------
# Hazard function
# ---------------------------------------------------------------------

def constant_hazard(
    run_length: np.ndarray,
    lam: float,
) -> np.ndarray:
    """
    Constant hazard function.

    h(r) = 1 / lambda

    lambda represents the prior expected run length between
    change-points.
    """

    if lam <= 1.0:
        raise ValueError(
            "lam must be greater than 1."
        )

    return np.full(
        len(run_length),
        1.0 / lam,
        dtype=float,
    )


# ---------------------------------------------------------------------
# Bayesian Online Change-Point Detection
# ---------------------------------------------------------------------

def bocd(
    data: np.ndarray,
    model: NormalGammaPredictive,
    lam: float,
) -> tuple[
    np.ndarray,
    NormalGammaPredictive,
]:
    """
    Run Bayesian Online Change-Point Detection.

    Parameters
    ----------
    data:
        One-dimensional transformed observations.

    model:
        Normal-Gamma predictive model.

    lam:
        Prior expected run length between change-points.

    Returns
    -------
    R:
        Posterior run-length matrix.

        R[t, r] =
            P(run length = r | x_1, ..., x_t)

    model:
        Updated predictive model.
    """

    data = np.asarray(
        data,
        dtype=float,
    )

    if data.ndim != 1:
        raise ValueError(
            "data must be one-dimensional."
        )

    if len(data) == 0:
        raise ValueError(
            "data cannot be empty."
        )

    if not np.all(
        np.isfinite(data)
    ):
        raise ValueError(
            "data contains non-finite values."
        )

    T = len(data)

    R = np.zeros(
        (
            T + 1,
            T + 1,
        ),
        dtype=float,
    )

    R[0, 0] = 1.0

    # Work in log space for numerical stability.
    log_R_previous = np.array(
        [0.0],
        dtype=float,
    )

    for t in range(
        1,
        T + 1,
    ):

        x = data[t - 1]

        log_pred = (
            model.log_pred_prob(x)
        )

        run_lengths = np.arange(t)

        hazard = constant_hazard(
            run_lengths,
            lam,
        )

        log_hazard = np.log(
            hazard
        )

        log_survival = np.log1p(
            -hazard
        )

        # Growth probabilities:
        # existing run continues.
        log_growth = (
            log_R_previous
            + log_pred
            + log_survival
        )

        # Change-point probability:
        # a new run starts at r = 0.
        log_cp = logsumexp(
            log_R_previous
            + log_pred
            + log_hazard
        )

        log_R_current = np.full(
            t + 1,
            -np.inf,
            dtype=float,
        )

        log_R_current[0] = (
            log_cp
        )

        log_R_current[1:] = (
            log_growth
        )

        # Normalize posterior probabilities.
        log_R_current -= (
            logsumexp(
                log_R_current
            )
        )

        R[
            t,
            : t + 1
        ] = np.exp(
            log_R_current
        )

        model.update(x)

        log_R_previous = (
            log_R_current
        )

    return R, model


# ---------------------------------------------------------------------
# Posterior summaries
# ---------------------------------------------------------------------

def expected_run_length(
    R: np.ndarray,
) -> np.ndarray:
    """
    Posterior expected run length.

    E[r_t | x_1:t]
    """

    run_lengths = np.arange(
        R.shape[1]
    )

    return (
        R[1:]
        * run_lengths
    ).sum(
        axis=1
    )


def map_run_length(
    R: np.ndarray,
) -> np.ndarray:
    """
    Posterior MAP run length at each time point.
    """

    return np.argmax(
        R[1:],
        axis=1,
    )


def changepoint_probability(
    R: np.ndarray,
) -> np.ndarray:
    """
    Posterior probability that the current run length is zero.

    Important
    ---------
    With a constant hazard, this quantity is constrained by the
    prior hazard rate and should not be interpreted as a standalone
    data-driven change-point score.

    The primary BOCD diagnostics are:

        - full run-length posterior
        - expected run length
        - MAP run length
    """

    return R[
        1:,
        0
    ]


# ---------------------------------------------------------------------
# Prior construction
# ---------------------------------------------------------------------

def build_default_model(
    x: np.ndarray,
) -> NormalGammaPredictive:
    """
    Construct a weakly informative Normal-Gamma prior.

    The prior mean and variance are initialized using only the first
    five observations, rather than the full time series.

    This avoids using future observations to construct the prior.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    if len(x) == 0:
        raise ValueError(
            "x cannot be empty."
        )

    baseline_n = min(
        5,
        len(x),
    )

    baseline = x[
        :baseline_n
    ]

    mu0 = float(
        np.mean(baseline)
    )

    if baseline_n > 1:

        variance = float(
            np.var(
                baseline,
                ddof=1,
            )
        )

    else:

        variance = 1.0

    variance = max(
        variance,
        0.05,
    )

    alpha0 = 2.0

    beta0 = (
        alpha0
        * variance
    )

    return NormalGammaPredictive(
        mu0=mu0,
        kappa0=1.0,
        alpha0=alpha0,
        beta0=beta0,
    )


# ---------------------------------------------------------------------
# Complete measles analysis
# ---------------------------------------------------------------------

def run_analysis(
    lam: float = 6.0,
    data_path: str | Path = DATA_PATH,
) -> pd.DataFrame:
    """
    Run BOCD on the annual measles surveillance series.

    Returns a tidy DataFrame containing:

        year
        cases
        log_cases
        expected_run_length
        map_run_length
        posterior_run_length_0
    """

    df = load_measles_data(
        data_path
    )

    cases = df[
        "cases"
    ].to_numpy(
        dtype=float
    )

    x = preprocess_counts(
        cases
    )

    model = build_default_model(
        x
    )

    R, _ = bocd(
        data=x,
        model=model,
        lam=lam,
    )

    return pd.DataFrame(
        {
            "year":
                df[
                    "year"
                ].to_numpy(),

            "cases":
                cases.astype(int),

            "log_cases":
                x,

            "expected_run_length":
                expected_run_length(R),

            "map_run_length":
                map_run_length(R),

            "posterior_run_length_0":
                changepoint_probability(R),
        }
    )


# ---------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------

if __name__ == "__main__":

    LAM = 6.0

    results = run_analysis(
        lam=LAM
    )

    print(
        "\nBayesian Online "
        "Change-Point Detection\n"
    )

    print(
        f"Prior expected run length "
        f"(lambda): {LAM}\n"
    )

    print(
        results.to_string(
            index=False
        )
    )
