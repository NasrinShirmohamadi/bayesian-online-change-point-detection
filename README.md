Bayesian Online Change-Point Detection for Measles Surveillance

A reproducible Bayesian extension of a classical CUSUM change-point detection project, applying Bayesian Online Change-Point Detection (BOCD) to annual measles surveillance counts.

The project implements the framework of Adams and MacKay (2007) and reframes sequential change-point detection in terms of a posterior distribution over the run length, rather than a binary threshold-based alarm.

Motivation

The companion project,

"CUSUM Change-Point Detection" (https://github.com/NasrinShirmohamadi/cusum-change-point-detection),

uses a classical cumulative-sum statistic to identify anomalous observations in annual measles surveillance data.

CUSUM provides a useful threshold-based detection mechanism, but its output is essentially a binary decision:

«Has the cumulative statistic crossed the detection threshold?»

BOCD provides a complementary Bayesian perspective.

At each time point, BOCD maintains a posterior distribution

[
P(r_t \mid x_{1:t}),
]

where r_t is the time since the most recent change-point.

This provides information about uncertainty in the current regime and allows the inferred run length to evolve sequentially as new observations arrive.

Research question

The central methodological question is:

«How does Bayesian posterior run-length inference complement classical threshold-based CUSUM detection in sequential infectious-disease surveillance?»

The project is intended as a methodological bridge between classical statistical process monitoring and modern predictive Bayesian inference.

Data

The example uses annual measles surveillance counts for 2010–2025.

Year| Reported cases
2010| 63
2011| 220
2012| 55
2013| 187
2014| 667
2015| 188
2016| 86
2017| 120
2018| 375
2019| 1274
2020| 13
2021| 49
2022| 121
2023| 59
2024| 285
2025| 2289

The observations are log-transformed before fitting the continuous Normal-Gamma predictive model used by BOCD.

Method

1. Bayesian Online Change-Point Detection

The implementation follows:

«Adams, R. P. & MacKay, D. J. C. (2007). Bayesian Online Changepoint Detection. arXiv:0710.3742.»

For each observation, BOCD updates the posterior distribution over possible run lengths.

The implementation uses a conjugate Normal-Gamma predictive model, giving a Student-t predictive distribution after marginalization.

The core quantities are:

- posterior probability of each run length;
- expected run length;
- MAP run length.

The constant hazard function is

[
H(r)=\frac{1}{\lambda},
]

where \lambda is the prior expected run length.

The default comparison uses:

lambda = 6

Sensitivity analysis evaluates:

lambda = 4, 6, 8, 10

2. CUSUM

The comparison reproduces the main settings of the companion CUSUM project:

Transformation: log(cases)
Baseline: median
Scale: sample standard deviation
k: 0.3
h: 1.5
Reset after threshold crossing: no

CUSUM produces a binary alarm when either the positive or negative cumulative statistic exceeds the specified threshold.

CUSUM vs BOCD

The two methods answer related but different questions.

CUSUM| BOCD
Classical sequential detector| Bayesian sequential detector
Threshold-based| Posterior-based
Binary alarm| Distribution over run lengths
Uses cumulative statistic| Uses sequential Bayesian updating
Detection threshold h| Prior hazard 1/\lambda
Limited uncertainty representation| Explicit posterior uncertainty

The purpose of this project is not to claim that BOCD universally outperforms CUSUM.

Instead, the comparison illustrates how Bayesian sequential inference can provide richer information about uncertainty and regime duration.

Repository structure

bayesian-online-change-point-detection/
│
├── bocd.py
├── compare_cusum_bocd.py
├── sensitivity_analysis.py
├── plot_comparison.py
├── requirements.txt
├── README.md
│
├── data/
│   └── measles_annual_counts.csv
│
└── figures/
    └── cusum_vs_bocd.png

Installation

Clone the repository and install the required dependencies:

pip install -r requirements.txt

Run BOCD

python bocd.py

Compare CUSUM and BOCD

python compare_cusum_bocd.py

The comparison produces year-by-year summaries including:

- reported cases;
- log-transformed observations;
- CUSUM statistics;
- CUSUM alarm;
- BOCD expected run length;
- BOCD MAP run length.

Sensitivity analysis

To examine the effect of the prior expected run length:

python sensitivity_analysis.py

The analysis evaluates:

lambda = 4
lambda = 6
lambda = 8
lambda = 10

This is important because Bayesian change-point inference depends on prior assumptions about the expected duration of a regime.

Visualization

Generate the main comparison figure with:

python plot_comparison.py

The resulting figure contains:

1. Annual measles surveillance counts with CUSUM alarms.
2. BOCD expected run length over time.

Important interpretation note

With a constant hazard, the posterior probability of run length zero is tied directly to the hazard specification and therefore should not be interpreted as an independent data-driven change-point score.

For this reason, the main BOCD visualization focuses on the posterior expected run length and MAP run length.

The project also does not claim that the BOCD output is calibrated in a formal statistical sense. Calibration assessment would require additional simulation or posterior predictive validation.

Limitations

Several extensions remain possible.

Count-specific observation models

The current BOCD implementation applies a Normal-Gamma predictive model to log-transformed counts.

A natural extension would be to develop a count-specific predictive model, such as a Poisson-Gamma or Negative-Binomial formulation.

Hierarchical surveillance models

The current analysis treats the annual series as a single sequence.

A more realistic surveillance model could introduce hierarchical structure across:

- geographic regions;
- demographic groups;
- surveillance systems;
- disease subtypes.

Non-constant hazard functions

The current implementation uses a constant hazard.

Future work could investigate hazard functions informed by:

- epidemiological knowledge;
- seasonal structure;
- intervention periods;
- covariates;
- hierarchical information.

Predictive Bayesian extensions

A further research direction is to investigate how sequential Bayesian change-point inference relates to modern predictive-Bayesian approaches, including uncertainty-aware prediction under structured, non-exchangeable data.

Relation to my research interests

This project builds directly on my background in biostatistics, Bayesian modeling, longitudinal data analysis, and statistical machine learning.

My master's research involved a Bayesian finite-mixture generalized linear mixed model fitted using MCMC for detecting anomalous outbreak years in infectious-disease surveillance data.

The present project extends that statistical perspective from retrospective Bayesian modeling toward sequential predictive inference and uncertainty-aware machine learning.

It is also aligned with my research interest in knowledge-driven and uncertainty-aware machine learning, particularly Bayesian inference for structured biomedical data.

References

Adams, R. P., & MacKay, D. J. C. (2007).

Bayesian Online Changepoint Detection.

arXiv:0710.3742.

https://arxiv.org/abs/0710.3742- change-point detection
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
