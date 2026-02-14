# Nowcasting PE NAVs

A pedagogical Python implementation of **"Nowcasting Net Asset Values: The Case of Private Equity"** by Brown, Ghysels, and Gredil (*The Review of Financial Studies*, Vol 36(3), March 2023, pp. 945-986).

## What This Does

Private equity funds report NAVs only quarterly, with smoothing and lag. This project implements the paper's state space model (SSM) that combines sparse quarterly NAVs, irregular fund distributions, and weekly comparable public asset returns to estimate latent fund values at weekly frequency using a Kalman filter.

Everything runs on **simulated data** with known parameters, so the methodology can be validated against ground truth.

## Project Structure

```
nowcasting_pe/
├── src/
│   ├── garch.py            # GARCH(1,1) for idiosyncratic volatility h_t
│   ├── simulation.py       # Synthetic PE fund DGP (eqs 1-9)
│   ├── naive.py            # Rc-interpolated NAV baseline (eq A.10)
│   ├── ssm.py              # SSM matrix construction (eqs A.4-A.6)
│   ├── kalman.py           # Kalman filter (A.2) and smoother (A.3)
│   ├── estimation.py       # Profile likelihood grid + MLE + EM iterations
│   ├── metrics.py          # PME-based RMSE metrics (Section 2.3)
│   └── visualization.py    # Plotting functions for all figures
├── notebooks/
│   └── nowcasting_pe_nav.ipynb   # Main pedagogical notebook
├── docs/plans/                    # Design and implementation plans
├── pyproject.toml
└── README.md
```

## Notebook Sections

1. **Introduction & Motivation** — The PE valuation problem
2. **Data Generating Process** — Simulate a fund (eqs 1-9), visualize all series
3. **Naive Nowcast** — Rc-interpolated NAVs as baseline
4. **State Space Model** — Observation/transition equations, matrix construction
5. **Kalman Filter & Smoother** — Forward/backward recursions, return extraction
6. **Parameter Estimation** — Profile likelihood grid, MLE, EM iterations
7. **Partial Imputation** — Peer-fund parameter anchoring
8. **Performance Metrics** — PME-based in-sample, OOS, and hybrid RMSE
9. **Monte Carlo Experiments** — Panel simulation, parameter recovery
10. **Sensitivity Analysis** — Effect of key parameters on nowcast quality
11. **Glossary & Appendix** — Definitions with links for SSM, Kalman filter, GARCH, MLE, PME, etc.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv venv
source .venv/bin/activate
uv pip install -e "."
```

## Usage

```bash
jupyter notebook notebooks/nowcasting_pe_nav.ipynb
```

Or run modules directly:

```python
from src.simulation import FundParams, simulate_fund
from src.kalman import kalman_filter, extract_returns

params = FundParams(beta=1.19, lam=0.90, seed=42)
fund = simulate_fund(params)

kf = kalman_filter(
    T=fund.T, alpha=params.alpha, beta=params.beta,
    beta_c=params.beta_c, psi=params.psi,
    F=params.F, Fc=params.Fc, lam=params.lam, delta=params.delta,
    sigma_n=params.sigma_n, sigma_d=params.sigma_d,
    Rm=fund.rm, rc_log=fund.rc_log, h_t=fund.h_t,
    D=fund.D, C=fund.C, NAV_reported=fund.NAV_reported,
    is_dist_week=fund.is_dist_week, is_nav_week=fund.is_nav_week,
    lambda_t=fund.lambda_t, delta_t=fund.delta_t,
    m_hat=fund.m_t,
)

r_weekly, r_cumulative = extract_returns(kf, use_smoothed=False)
```

## Key Parameters

| Parameter | Symbol | Description | Paper Default |
|-----------|--------|-------------|---------------|
| `alpha` | $\alpha$ | Weekly excess return | 0.0 |
| `beta` | $\beta$ | Market risk exposure | 1.19 |
| `F` | $F$ | Fund idiosyncratic vol scale | 2.0 |
| `lam` | $\lambda$ | NAV smoothing parameter | 0.90 |
| `sigma_n` | $\sigma_n$ | NAV reporting noise | 0.05 |
| `delta` | $\delta$ | Distribution density trend | 0.03 |
| `beta_c` | $\beta_c$ | Comparable asset loading | 0.85 |
| `Fc` | $F_c$ | Comparable idiosyncratic vol | 1.0 |

## Dependencies

numpy, scipy, pandas, matplotlib, arch (GARCH), jupyter, ipykernel

## Reference

Brown, G. W., Ghysels, E., & Gredil, O. R. (2023). Nowcasting Net Asset Values: The Case of Private Equity. *The Review of Financial Studies*, 36(3), 945-986. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3507873)
