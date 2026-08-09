"""
Visualization of CUSUM and BOCD
================================

Creates a reproducible figure comparing:

1. Annual measles surveillance counts.
2. CUSUM threshold-based alarms.
3. BOCD expected run length.

The analysis uses the same data and model implementations as
compare_cusum_bocd.py.

The figure is intended for the project README and for documenting
the methodological difference between classical CUSUM detection
and Bayesian posterior run-length inference.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from compare_cusum_bocd import run_comparison


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

FIGURE_DIR = (
    PROJECT_ROOT
    / "figures"
)

FIGURE_PATH = (
    FIGURE_DIR
    / "cusum_vs_bocd.png"
)


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def create_comparison_plot():
    """
    Generate and save the CUSUM vs BOCD comparison figure.
    """

    results = run_comparison()

    years = results[
        "year"
    ].to_numpy()

    cases = results[
        "cases"
    ].to_numpy()

    cusum_alarm = results[
        "cusum_alarm"
    ].to_numpy()

    expected_run_length = results[
        "bocd_expected_run_length"
    ].to_numpy()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
    )

    # ---------------------------------------------------------------
    # Panel 1: Surveillance counts
    # ---------------------------------------------------------------

    ax1 = axes[0]

    ax1.plot(
        years,
        cases,
        marker="o",
        linewidth=1.8,
        label="Reported measles cases",
    )

    alarm_mask = (
        cusum_alarm == 1
    )

    if alarm_mask.any():

        ax1.scatter(
            years[alarm_mask],
            cases[alarm_mask],
            s=70,
            marker="x",
            linewidths=2,
            label="CUSUM alarm",
        )

    ax1.set_ylabel(
        "Reported cases"
    )

    ax1.set_title(
        "Annual measles surveillance counts"
    )

    ax1.legend(
        loc="upper left"
    )

    ax1.grid(
        alpha=0.25
    )

    # ---------------------------------------------------------------
    # Panel 2: BOCD expected run length
    # ---------------------------------------------------------------

    ax2 = axes[1]

    ax2.plot(
        years,
        expected_run_length,
        marker="o",
        linewidth=1.8,
        label="BOCD expected run length",
    )

    ax2.set_xlabel(
        "Year"
    )

    ax2.set_ylabel(
        "Expected run length"
    )

    ax2.set_title(
        "Bayesian Online Change-Point Detection"
    )

    ax2.legend(
        loc="upper left"
    )

    ax2.grid(
        alpha=0.25
    )

    # ---------------------------------------------------------------
    # Formatting
    # ---------------------------------------------------------------

    ax2.set_xticks(
        years
    )

    fig.suptitle(
        "CUSUM vs Bayesian Online Change-Point Detection",
        fontsize=14,
    )

    fig.tight_layout()

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved figure to: {FIGURE_PATH}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    create_comparison_plot()
