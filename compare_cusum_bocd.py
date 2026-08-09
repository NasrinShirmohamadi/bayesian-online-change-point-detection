"""
Comparison of CUSUM and Bayesian Online Change-Point Detection
================================================================

This module provides a common analysis framework for comparing:

1. Classical CUSUM-style change detection
2. Bayesian Online Change-Point Detection (BOCD)

The two methods are applied to the same annual measles surveillance
series.

The goal is methodological comparison rather than treating the two
methods as interchangeable detectors.

CUSUM produces a threshold-based signal, whereas BOCD produces a
posterior distribution over run length.
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

PROJECT_ROOT = (
    Path(__file__).resolve().parent
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "measles_annual_counts.csv"
)


# ---------------------------------------------------------------------
# CUSUM implementation
# ---------------------------------------------------------------------

def cusum_detector(
    values: np.ndarray,
    reference: float | None = None,
    threshold: float = 2.0,
    drift: float = 0.0,
) -> np.ndarray:
    """
    Simple two-sided standardized CUSUM detector.

    Parameters
    ----------
    values:
        One-dimensional observations.

    reference:
        Reference mean. If None, the mean of the observations is used.

    threshold:
        Decision threshold for the cumulative statistic.

    drift:
        Allowance parameter controlling sensitivity.

    Returns
    -------
    alarms:
        Binary array indicating whether the CUSUM threshold was crossed.

    Notes
    -----
    This function is intended as a transparent comparison implementation.
    The final comparison should use the exact CUSUM specification from the
    original CUSUM repository once its implementation is incorporated.
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

    if threshold <= 0:
        raise ValueError(
            "threshold must be positive."
        )

    if reference is None:
        reference = float(
            np.mean(values)
        )

    scale = float(
        np.std(
            values,
            ddof=1,
        )
    )

    if scale <= 0:
        raise ValueError(
            "The observations must have non-zero variance."
        )

    standardized = (
        values - reference
    ) / scale

    positive = 0.0
    negative = 0.0

    alarms = np.zeros(
        len(values),
        dtype=int,
    )

    for i, value in enumerate(
        standardized
    ):

        positive = max(
            0.0,
            positive
            + value
            - drift,
        )

        negative = min(
            0.0,
            negative
            + value
            + drift,
        )

        if (
            positive >= threshold
            or
            abs(negative) >= threshold
        ):

            alarms[i] = 1

            # Reset after an alarm.
            positive = 0.0
            negative = 0.0

    return alarms


# ---------------------------------------------------------------------
# BOCD analysis
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
            "expected_run_length":
                expected_run_length(R),

            "map_run_length":
                map_run_length(R),

            "posterior_run_length_0":
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
    lam: float = 6.0,
    cusum_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Run CUSUM and BOCD on the same measles surveillance series.

    Returns
    -------
    DataFrame containing the original data and diagnostics from both
    approaches.
    """

    df = load_measles_data(
        data_path
    )

    cases = df[
        "cases"
    ].to_numpy(
        dtype=float
    )

    log_cases = preprocess_counts(
        cases
    )

    # CUSUM is applied to the same transformed series so that both
    # methods operate on the same basic scale.
    cusum_alarms = cusum_detector(
        log_cases,
        threshold=cusum_threshold,
    )

    bocd_results = run_bocd(
        log_cases,
        lam=lam,
    )

    comparison = df.copy()

    comparison[
        "log_cases"
    ] = log_cases

    comparison[
        "cusum_alarm"
    ] = cusum_alarms

    comparison[
        "bocd_expected_run_length"
    ] = bocd_results[
        "expected_run_length"
    ]

    comparison[
        "bocd_map_run_length"
    ] = bocd_results[
        "map_run_length"
    ]

    comparison[
        "bocd_run_length_0"
    ] = bocd_results[
        "posterior_run_length_0"
    ]

    return comparison


# ---------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------

if __name__ == "__main__":

    results = run_comparison(
        lam=6.0,
        cusum_threshold=2.0,
    )

    print(
        "\nCUSUM vs BOCD comparison\n"
    )

    print(
        results.to_string(
            index=False
        )
    )
