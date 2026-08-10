"""
CUSUM vs Bayesian Online Change-Point Detection
================================================

Comparison of the classical CUSUM detector from the companion
repository with Bayesian Online Change-Point Detection (BOCD).

Both methods are applied to the same annual measles surveillance
series after log transformation.

CUSUM
-----
The implementation follows the companion CUSUM repository:

    - log-transformed case counts
    - median baseline
    - standard deviation scaling
    - k = 0.3
    - h = 1.5
    - no reset after threshold crossing

BOCD
----
    - log-transformed case counts
    - Normal-Gamma predictive model
    - constant hazard
    - posterior distribution over run length

The purpose is methodological comparison rather than declaring one
method universally superior to the other.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bocd import (
    build_default_model,
    bocd,
    expected_run_length,
    load_measles_data,
    map_run_length,
    preprocess_counts,
)


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "measles_annual_counts.csv"
)


# ---------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------

def cusum_detector(
    values: np.ndarray,
    k: float = 0.3,
    h: float = 1.5,
) -> pd.DataFrame:
    """
    Apply the CUSUM procedure used in the companion repository.

    Parameters
    ----------
    values:
        One-dimensional log-transformed observations.

    k:
        Reference/slack parameter.

    h:
        Decision threshold.

    Returns
    -------
    DataFrame
        Contains the positive and negative cumulative statistics
        and the corresponding binary alarm flag.

    Notes
    -----
    The baseline is the median of the observations.

    Importantly, the cumulative statistics are NOT reset after
    crossing the threshold, matching the companion implementation.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "values must be one-dimensional."
        )

    if len(values) == 0:
        raise ValueError(
            "values cannot be empty."
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "values contain non-finite values."
        )

    if k < 0:
        raise ValueError(
            "k must be non-negative."
        )

    if h <= 0:
        raise ValueError(
            "h must be positive."
        )

    # Same baseline specification as the companion CUSUM project.
    baseline = float(
        np.median(values)
    )

    # Same scale calculation.
    sigma = float(
        np.std(
            values,
            ddof=1,
        )
    )

    if sigma <= 0:
        raise ValueError(
            "The observations must have non-zero variance."
        )

    standardized = (
        values - baseline
    ) / sigma

    positive = np.zeros(
        len(values),
        dtype=float,
    )

    negative = np.zeros(
        len(values),
        dtype=float,
    )

    alarms = np.zeros(
        len(values),
        dtype=int,
    )

    for i, value in enumerate(
        standardized
    ):

        previous_positive = (
            positive[i - 1]
            if i > 0
            else 0.0
        )

        previous_negative = (
            negative[i - 1]
            if i > 0
            else 0.0
        )

        positive[i] = max(
            0.0,
            previous_positive
            + value
            - k,
        )

        negative[i] = min(
            0.0,
            previous_negative
            + value
            + k,
        )

        # Match the companion repository:
        # threshold crossing does NOT reset the statistics.
        if (
            positive[i] > h
            or abs(negative[i]) > h
        ):
            alarms[i] = 1

    return pd.DataFrame(
        {
            "cusum_positive":
                positive,

            "cusum_negative":
                negative,

            "cusum_alarm":
                alarms,
        }
    )


# ---------------------------------------------------------------------
# BOCD
# ---------------------------------------------------------------------

def run_bocd(
    log_cases: np.ndarray,
    lam: float = 6.0,
) -> pd.DataFrame:
    """
    Run BOCD and return posterior run-length summaries.
    """

    model = build_default_model(
        log_cases
    )

    R, _ = bocd(
        data=log_cases,
        model=model,
        lam=lam,
    )

    return pd.DataFrame(
        {
            "bocd_expected_run_length":
                expected_run_length(R),

            "bocd_map_run_length":
                map_run_length(R),

            # Retained for completeness.
            # With constant hazard this is not a
            # data-driven change-point score.
            "bocd_run_length_0":
                R[
                    1:,
                    0
                ],
        }
    )


# ---------------------------------------------------------------------
# Complete comparison
# ---------------------------------------------------------------------

def run_comparison(
    data_path: str | Path = DATA_PATH,
    cusum_k: float = 0.3,
    cusum_h: float = 1.5,
    bocd_lam: float = 6.0,
) -> pd.DataFrame:
    """
    Run CUSUM and BOCD on the same measles surveillance series.
    """

    df = load_measles_data(
        data_path
    )

    cases = df[
        "cases"
    ].to_numpy(
        dtype=float
    )

    # Both methods operate on the same transformed observations.
    log_cases = preprocess_counts(
        cases
    )

    # ---------------------------------------------------------------
    # CUSUM
    # ---------------------------------------------------------------

    cusum_results = cusum_detector(
        log_cases,
        k=cusum_k,
        h=cusum_h,
    )

    # ---------------------------------------------------------------
    # BOCD
    # ---------------------------------------------------------------

    bocd_results = run_bocd(
        log_cases,
        lam=bocd_lam,
    )

    # ---------------------------------------------------------------
    # Combine results
    # ---------------------------------------------------------------

    comparison = df.copy()

    comparison[
        "log_cases"
    ] = log_cases

    comparison[
        "cusum_positive"
    ] = cusum_results[
        "cusum_positive"
    ]

    comparison[
        "cusum_negative"
    ] = cusum_results[
        "cusum_negative"
    ]

    comparison[
        "cusum_alarm"
    ] = cusum_results[
        "cusum_alarm"
    ]

    comparison[
        "bocd_expected_run_length"
    ] = bocd_results[
        "bocd_expected_run_length"
    ]

    comparison[
        "bocd_map_run_length"
    ] = bocd_results[
        "bocd_map_run_length"
    ]

    comparison[
        "bocd_run_length_0"
    ] = bocd_results[
        "bocd_run_length_0"
    ]

    return comparison


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":

    results = run_comparison()

    print(
        "\nCUSUM vs Bayesian Online "
        "Change-Point Detection\n"
    )

    print(
        "CUSUM parameters: "
        "k=0.3, h=1.5"
    )

    print(
        "CUSUM baseline: median"
    )

    print(
        "CUSUM reset after alarm: no"
    )

    print(
        "BOCD prior expected "
        "run length: lambda=6.0\n"
    )

    print(
        results.to_string(
            index=False
        )
    )

    print(
        "\nCUSUM alarm years:"
    )

    alarm_years = results.loc[
        results["cusum_alarm"] == 1,
        "year",
    ].tolist()

    print(
        alarm_years
        if alarm_years
        else "None"
    )

    print(
        "\nYears with the smallest "
        "BOCD expected run length:"
    )

    print(
        results[
            [
                "year",
                "bocd_expected_run_length",
                "bocd_map_run_length",
            ]
        ]
        .sort_values(
            "bocd_expected_run_length"
        )
        .head(5)
        .to_string(
            index=False
        )
)
