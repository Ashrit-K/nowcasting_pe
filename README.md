# Nowcasting Private Equity Fund Values

A Python implementation of the methodology from **"Nowcasting Net Asset Values: The Case of Private Equity"** by Brown, Ghysels, and Gredil (*The Review of Financial Studies*, 2023).

## The Problem

Private equity funds only report their values once per quarter — and even those numbers are smoothed and delayed. Between reports, investors are essentially flying blind. If you own a stake in a PE fund and the public stock market drops 15% this week, what is your fund actually worth right now? The quarterly report won't tell you for months.

## The Approach

This project implements a statistical model that produces **weekly estimates of a PE fund's true value** by stitching together three sources of information:

- **Quarterly NAV reports** — the official (but stale and smoothed) fund valuations
- **Cash flow events** — distributions paid out and capital called in, which happen irregularly
- **Public market returns** — weekly returns from a comparable public asset (like a sector ETF), which are always available

The core idea is a **Kalman filter** — a well-established algorithm that optimally combines noisy, irregularly-arriving signals to track a hidden quantity over time. Here, the hidden quantity is the fund's true return each week.

The model effectively "unsmooths" the quarterly reports: instead of a value that barely changes quarter to quarter, you get a realistic, more volatile weekly series that reflects what the fund is likely worth in real time.

## How It's Validated

Everything runs on **simulated data** where we know the true answer. We generate a synthetic PE fund with realistic behavior — market exposure, idiosyncratic risk, quarterly reporting, smoothing, cash flows — and then see how well the model can recover the true weekly values from only the information an investor would actually observe.

This makes it possible to measure exactly how much better the model does compared to simpler approaches (like just scaling the last reported NAV by public market returns).

## What's in the Notebook

The main deliverable is an interactive Jupyter notebook (`notebooks/nowcasting_pe_nav.ipynb`) that walks through the full methodology step by step:

1. **Why this matters** — motivation and context for the PE valuation problem
2. **Simulating a fund** — generating realistic synthetic PE fund data over ~11 years
3. **A simple baseline** — estimating values by scaling reported NAVs with public returns
4. **The state space model** — how the problem is cast into a form the Kalman filter can solve
5. **Running the Kalman filter** — producing weekly return estimates, comparing to the truth
6. **Estimating parameters** — learning the model's unknowns (market beta, smoothing degree, etc.) from data
7. **Handling sparse data** — a technique for when you don't have enough observations
8. **Measuring performance** — how close are the nowcasts to reality?
9. **Monte Carlo experiments** — repeating the exercise across many simulated funds to test robustness
10. **Sensitivity analysis** — which parameters matter most for nowcast quality?
11. **Glossary** — plain-language definitions of key concepts with links for further reading

Each section includes visualizations that compare the model's estimates against the known ground truth.

## Project Structure

```
nowcasting_pe/
├── src/                        # All model code
│   ├── simulation.py           # Generate synthetic PE fund data
│   ├── ssm.py                  # Build the state space model matrices
│   ├── kalman.py               # Kalman filter and smoother
│   ├── naive.py                # Simple baseline for comparison
│   ├── estimation.py           # Parameter estimation (grid search + optimization)
│   ├── metrics.py              # Performance measurement
│   ├── garch.py                # Time-varying volatility estimation
│   └── visualization.py        # All plotting functions
├── notebooks/
│   └── nowcasting_pe_nav.ipynb # Main walkthrough notebook
├── docs/plans/                 # Design and implementation notes
└── pyproject.toml              # Dependencies and project config
```

## Getting Started

**Requirements:** Python 3.13+ and [uv](https://docs.astral.sh/uv/) (a fast Python package manager).

```bash
uv venv
source .venv/bin/activate
uv pip install -e "."
```

Then open the notebook:

```bash
jupyter notebook notebooks/nowcasting_pe_nav.ipynb
```

## Key Model Parameters

| Parameter | What It Controls | Default |
|-----------|-----------------|---------|
| Beta | How much the fund moves with the public market | 1.19 |
| Alpha | The fund's excess return above the market | 0.0 |
| Lambda | How heavily the fund manager smooths reported NAVs (0 = no smoothing, 1 = fully stale) | 0.90 |
| F | Scale of the fund's idiosyncratic (non-market) risk | 2.0 |
| Beta_c | How closely the comparable public asset tracks the same market factor | 0.85 |

## Dependencies

numpy, scipy, pandas, matplotlib, arch, jupyter

## Reference

Brown, G. W., Ghysels, E., & Gredil, O. R. (2023). Nowcasting Net Asset Values: The Case of Private Equity. *The Review of Financial Studies*, 36(3), 945-986. [Paper on SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3507873)
