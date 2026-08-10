"""
Sensitivity Analysis for Bayesian Online Change-Point Detection
================================================================

Examines how BOCD posterior run-length summaries change under
different prior expected run lengths (lambda).

The analysis uses the same annual measles surveillance data and
the same Normal-Gamma predictive model as bocd.py.

The purpose is to assess whether the main BOCD findings are
sensitive to the choice of the constant hazard parameter.
"""

from __future__ import annotations

from pathlib import Path

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

DATA_PATH = PROJECT_ROOT / "data" / "measles_annual_counts.csv"


# ---------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------


def run_sensitivity_analysis(
    lambdas: list[float] | None = None,
    data_path: str | Path = DATA_PATH,
) -> pd.DataFrame:
    """
    Run BOCD for multiple values of lambda.

    Parameters
    ----------
    lambdas:
        Prior expected run lengths to evaluate.

    data_path:
        Path to the measles surveillance CSV file.

    Returns
    -------
    DataFrame
        One row per year and lambda value.
    """

    if lambdas is None:
        lambdas = [
            4.0,
            6.0,
            8.0,
            10.0,
        ]

    df = load_measles_data(data_path)

    cases = df["cases"].to_numpy(dtype=float)

    log_cases = preprocess_counts(cases)

    all_results = []

    for lam in lambdas:

        model = build_default_model(log_cases)

        R, _ = bocd(
            data=log_cases,
            model=model,
            lam=lam,
        )

        expected = expected_run_length(R)

        map_length = map_run_length(R)

        result = pd.DataFrame(
            {
                "year": df["year"].to_numpy(),
                "cases": cases.astype(int),
                "lambda": lam,
                "expected_run_length": expected,
                "map_run_length": map_length,
            }
        )

        all_results.append(result)

    return pd.concat(
        all_results,
        ignore_index=True,
    )


# ---------------------------------------------------------------------
# Identify years with smallest expected run length
# ---------------------------------------------------------------------


def summarize_sensitivity(
    results: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Identify the years with the smallest expected run length
    for each lambda value.
    """

    summaries = []

    for lam, group in results.groupby("lambda"):

        selected = group.sort_values("expected_run_length").head(top_n).copy()

        selected["rank"] = range(
            1,
            len(selected) + 1,
        )

        summaries.append(
            selected[
                [
                    "lambda",
                    "rank",
                    "year",
                    "cases",
                    "expected_run_length",
                    "map_run_length",
                ]
            ]
        )

    return pd.concat(
        summaries,
        ignore_index=True,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":

    results = run_sensitivity_analysis()

    summary = summarize_sensitivity(
        results,
        top_n=5,
    )

    print("\nBOCD sensitivity analysis\n")

    print("Prior expected run lengths:")

    print(sorted(results["lambda"].unique()))

    print("\nYears with the smallest " "expected run length:\n")

    print(summary.to_string(index=False))
