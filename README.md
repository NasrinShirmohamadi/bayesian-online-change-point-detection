# Bayesian Online Change-Point Detection for Measles Surveillance

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A reproducible Bayesian extension of a classical CUSUM change-point
detection project, applying **Bayesian Online Change-Point Detection
(BOCD)** to annual U.S. measles surveillance counts.

The project implements the framework of Adams & MacKay (2007) and
reframes sequential change-point detection as inference over a
**posterior distribution of the run length**, rather than a binary,
threshold-based alarm.

## Motivation

The companion project, [`cusum-change-point-detection`](https://github.com/NasrinShirmohamadi/cusum-change-point-detection),
uses a classical cumulative-sum statistic to flag anomalous
observations in annual measles surveillance data. CUSUM is a useful
threshold-based detector, but its output is essentially a binary
decision: *has the cumulative statistic crossed the detection
threshold?*

BOCD offers a complementary, Bayesian perspective. At each time step
it maintains a full posterior distribution

```
P(r_t | x_1:t)
```

where `r_t` is the run length — the time elapsed since the most
recent change-point. This captures uncertainty about the current
regime and lets that belief evolve sequentially as new observations
arrive.

**Research question:** How does Bayesian posterior run-length
inference complement classical threshold-based CUSUM detection in
sequential infectious-disease surveillance?

This project is intended as a methodological bridge between classical
statistical process monitoring and modern predictive Bayesian
inference.

## Data

Annual U.S. measles case counts, 2010–2025 (source: CDC).

| Year | Cases | Year | Cases |
|------|-------|------|-------|
| 2010 | 63    | 2018 | 375   |
| 2011 | 220   | 2019 | 1274  |
| 2012 | 55    | 2020 | 13    |
| 2013 | 187   | 2021 | 49    |
| 2014 | 667   | 2022 | 121   |
| 2015 | 188   | 2023 | 59    |
| 2016 | 86    | 2024 | 285   |
| 2017 | 120   | 2025 | 2289  |

Stored separately in [`data/measles_annual_counts.csv`](data/measles_annual_counts.csv),
keeping data and analysis code decoupled for easier inspection and
reproducibility. Observations are log-transformed (`x_t = log(cases_t)`)
before fitting the Normal-Gamma predictive model used by BOCD.

## Method

### 1. Bayesian Online Change-Point Detection

Implementation follows Adams, R. P. & MacKay, D. J. C. (2007),
*Bayesian Online Changepoint Detection*, [arXiv:0710.3742](https://arxiv.org/abs/0710.3742).

For each observation, BOCD updates the posterior over possible run
lengths using a conjugate **Normal-Gamma** predictive model, which
gives a Student-t predictive distribution after marginalization. The
recursion is carried out in log space for numerical stability.

Core outputs:

- **Full run-length posterior** — `P(r_t | x_1:t)`
- **Expected run length** — `E[r_t | x_1:t]`, the primary
  data-dependent diagnostic; it grows within a stable regime and
  drops sharply (with a typical one-step detection lag) after a
  genuine change-point
- **MAP run length** — `argmax_r P(r_t | x_1:t)`

The hazard function is constant, `H(r) = 1/λ`, where `λ` is the prior
expected run length between change-points. The default comparison
uses `λ = 6`; sensitivity analysis evaluates `λ ∈ {4, 6, 8, 10}`.

> **Interpretation note.** With a constant hazard, the posterior
> probability of run length zero is fixed by the hazard specification
> itself and is *not* an independent, data-driven change-point score.
> For this reason, expected run length and MAP run length — not raw
> `P(r_t = 0)` — are the primary diagnostics used throughout this
> project.

### 2. CUSUM (companion method)

Reproduces the settings of the companion CUSUM project for direct
comparison:

| Setting | Value |
|---|---|
| Transformation | `log(cases)` |
| Baseline | median |
| Scale | sample standard deviation |
| `k` | 0.3 |
| `h` | 1.5 |
| Reset after threshold crossing | no |

CUSUM raises a binary alarm when either the positive or negative
cumulative statistic exceeds the threshold.

### CUSUM vs. BOCD

| | CUSUM | BOCD |
|---|---|---|
| Type | Classical sequential detector | Bayesian sequential detector |
| Mechanism | Threshold-based | Posterior-based |
| Output | Binary alarm | Distribution over run lengths |
| Update rule | Cumulative statistic | Sequential Bayesian updating |
| Tuning | Detection threshold `h` | Prior hazard `1/λ` |
| Uncertainty | Not represented | Explicit posterior uncertainty |

The goal is not to show that BOCD universally outperforms CUSUM, but
to illustrate how Bayesian sequential inference provides richer
information about uncertainty and regime duration.

## Repository structure

```
bayesian-online-change-point-detection/
├── bocd.py                    # Core BOCD implementation
├── compare_cusum_bocd.py      # Side-by-side CUSUM vs. BOCD comparison
├── sensitivity_analysis.py    # Sensitivity of results to the hazard prior λ
├── plot_comparison.py         # Generates the main comparison figure
├── requirements.txt
├── README.md
├── data/
│   └── measles_annual_counts.csv
└── figures/
    └── cusum_vs_bocd.png
```

## Installation

```bash
git clone https://github.com/NasrinShirmohamadi/bayesian-online-change-point-detection.git
cd bayesian-online-change-point-detection
pip install -r requirements.txt
```

## Usage

**Run BOCD:**

```bash
python bocd.py
```

**Compare CUSUM and BOCD:**

```bash
python compare_cusum_bocd.py
```

Produces a year-by-year table with reported cases, log-transformed
values, CUSUM statistics and alarm flags, and BOCD expected/MAP run
length.

**Sensitivity analysis** (effect of the prior expected run length):

```bash
python sensitivity_analysis.py
```

Evaluates `λ ∈ {4, 6, 8, 10}` — important because Bayesian
change-point inference depends on this prior assumption about regime
duration.

**Generate the comparison figure:**

```bash
python plot_comparison.py
```

Produces `figures/cusum_vs_bocd.png`, showing (1) annual case counts
with CUSUM alarms, and (2) BOCD expected run length over time.

## Limitations

This is intentionally a methodological starting point, not a final
disease-surveillance model:

1. Counts are modeled via a continuous Normal-Gamma predictive model
   on log-transformed data, rather than a count-specific likelihood
   (e.g. Poisson-Gamma or Negative-Binomial).
2. The hazard function is constant; a data- or knowledge-informed
   hazard (seasonality, intervention periods, covariates) is not yet
   implemented.
3. The series is short — 16 annual observations (2010–2025).
4. Results are sensitive to the choice of prior and expected run
   length (see `sensitivity_analysis.py`).
5. No formal simulation-based calibration or false-positive analysis
   has yet been performed.

## Planned extensions

- Count-specific (Poisson-Gamma / Negative-Binomial) predictive model
- Non-constant, informed hazard functions
- Simulation-based calibration and false-positive analysis
- **Hierarchical BOCD** across multiple surveillance regions,
  demographic groups, or disease subtypes — moving the project
  toward structured, non-exchangeable sequential data and closer to
  modern Bayesian predictive modeling
- Investigation of how sequential Bayesian change-point inference
  relates to broader predictive-Bayesian approaches for
  uncertainty-aware prediction under structured, non-exchangeable
  data

## Relation to my research interests

This project builds directly on my background in biostatistics,
Bayesian modeling, longitudinal data analysis, and statistical
machine learning. My M.Sc. research developed a Bayesian
finite-mixture generalized linear mixed model, fitted via MCMC, to
detect anomalous outbreak years in infectious-disease surveillance
data. This project extends that retrospective Bayesian perspective
toward sequential, predictive inference and uncertainty-aware machine
learning — aligned with my broader research interest in knowledge-
driven, uncertainty-aware machine learning for structured biomedical
data.

## Reference

Adams, R. P., & MacKay, D. J. C. (2007). *Bayesian Online Changepoint
Detection*. [arXiv:0710.3742](https://arxiv.org/abs/0710.3742)

## License

MIT — see [LICENSE](LICENSE).
