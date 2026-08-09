"""
Bayesian Online Change-Point Detection (BOCD)
=============================================

Bayesian Online Change-Point Detection for annual measles surveillance
counts, following Adams & MacKay (2007).

This repository extends a classical CUSUM change-point analysis of
the same measles surveillance series with a Bayesian sequential
predictive framework.

Reference
---------
Adams, R. P., & MacKay, D. J. C. (2007).
Bayesian Online Changepoint Detection.
arXiv:0710.3742
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.special import gammaln, logsumexp


# ---------------------------------------------------------------------
# Measles surveillance data
# ---------------------------------------------------------------------

DATA = {
    2010: 63,
    2011: 220,
    2012: 55,
    2013: 187,
    2014: 667,
    2015: 188,
    2016: 86,
    2017: 120,
    2018: 375,
    2019: 1274,
    2020: 13,
    2021: 49,
    2022: 121,
    2023: 59,
    2024: 285,
    2025: 2289,
}


def load_measles_data() -> pd.DataFrame:
    """Load and validate the annual measles surveillance series."""

    df = (
        pd.Series(DATA, name="cases")
        .sort_index()
        .rename_axis("year")
        .reset_index()
    )

    if df["year"].duplicated().any():
        raise ValueError("Duplicate years detected.")

    if df["cases"].isna().any():
        raise ValueError("Missing case counts detected.")

    if (df["cases"] <= 0).any():
        raise ValueError(
            "All case counts must be positive because "
            "the analysis uses log(cases)."
        )

    return df


def preprocess_counts(
    cases: np.ndarray,
) -> np.ndarray:
    """
    Apply the same transformation used by the CUSUM project.

    x_t = log(cases_t)
    """

    cases = np.asarray(
        cases,
        dtype=float,
    )

    if cases.ndim != 1:
        raise ValueError(
            "cases must be one-dimensional."
        )

    if not np.all(np.isfinite(cases)):
        raise ValueError(
            "cases contains non-finite values."
        )

    if np.any(cases <= 0):
        raise ValueError(
            "cases must be positive before log transformation."
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

    Marginalizing mu and tau gives a Student-t predictive distribution.
    """

    def __init__(
        self,
        mu0: float,
        kappa0: float = 1.0,
        alpha0: float = 2.0,
        beta0: float = 1.0,
    ):

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
        """Log Student-t predictive density for x."""

        df = 2.0 * self.alpha

        scale = np.sqrt(
            self.beta
            * (self.kappa + 1.0)
            / (self.alpha * self.kappa)
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
            - 0.5 * np.log(
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
        """Update the Normal-Gamma parameters after observing x."""

        new_mu = (
            self.kappa * self.mu + x
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
                * (x - self.mu) ** 2
            )
            / (
                2.0
                * (self.kappa + 1.0)
            )
        )

        self.mu = np.concatenate(
            (
                [self.mu0],
                new_mu,
            )
        )

        self.kappa = np.concatenate(
            (
                [self.kappa0],
                new_kappa,
            )
        )

        self.alpha = np.concatenate(
            (
                [self.alpha0],
                new_alpha,
            )
        )

        self.beta = np.concatenate(
            (
                [self.beta0],
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
# BOCD recursion
# ---------------------------------------------------------------------

def bocd(
    data: np.ndarray,
    model: NormalGammaPredictive,
    lam: float,
) -> tuple[np.ndarray, NormalGammaPredictive]:
    """
    Run Bayesian Online Change-Point Detection.

    Returns
    -------
    R:
        Run-length posterior matrix.

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

    if not np.all(np.isfinite(data)):
        raise ValueError(
            "data contains non-finite values."
        )

    T = len(data)

    R = np.zeros(
        (T + 1, T + 1),
        dtype=float,
    )

    R[0, 0] = 1.0

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

        # Growth probabilities.
        log_growth = (
            log_R_previous
            + log_pred
            + log_survival
        )

        # Change-point probability.
        log_cp = logsumexp(
            log_R_previous
            + log_pred
            + log_hazard
        )

        log_R_current = np.full(
            t + 1,
            -np.inf,
        )

        log_R_current[0] = log_cp

        log_R_current[1:] = log_growth

        # Normalize posterior.
        log_R_current -= logsumexp(
            log_R_current
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
    """Posterior expected run length."""

    run_lengths = np.arange(
        R.shape[1]
    )

    return (
        R[1:]
        * run_lengths
    ).sum(axis=1)


def map_run_length(
    R: np.ndarray,
) -> np.ndarray:
    """Posterior MAP run length."""

    return np.argmax(
        R[1:],
        axis=1,
    )


def changepoint_probability(
    R: np.ndarray,
) -> np.ndarray:
    """
    Posterior probability of run length zero.

    Under a constant hazard, this should NOT be interpreted as a
    standalone data-driven change-point score.

    The primary diagnostics are the full run-length posterior,
    expected run length, and MAP run length.
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

    Only the first five observations are used to initialize the prior.
    """

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

    variance = float(
        np.var(
            baseline,
            ddof=1,
        )
    )

    variance = max(
        variance,
        0.05,
    )

    alpha0 = 2.0

    beta0 = (
        alpha0 * variance
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
    lam: float,
) -> pd.DataFrame:
    """
    Run BOCD on the 2010-2025 measles surveillance series.
    """

    df = load_measles_data()

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
                df["year"].to_numpy(),

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
        f"\nBOCD analysis "
        f"(lambda = {LAM})\n"
    )

    print(
        results.to_string(
            index=False
        )
    )
