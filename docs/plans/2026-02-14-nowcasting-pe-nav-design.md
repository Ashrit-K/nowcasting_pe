# Design: Nowcasting PE NAV — Pedagogical Prototype

**Paper:** "Nowcasting Net Asset Values: The Case of Private Equity"
Brown, Ghysels, Gredil — *The Review of Financial Studies*, Vol 36(3), March 2023, pp. 945-986.
SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3507873

**Date:** 2026-02-14

---

## Goal

Build a **pedagogical Jupyter notebook** backed by clean Python modules that:
- Teaches the core concepts of the paper in an intuitive, step-by-step manner
- Runs entirely on simulated data (no real fund data required)
- Covers the full methodology: simulation, naive nowcast, SSM, Kalman filter/smoother, parameter estimation, partial imputation, Monte Carlo experiments
- Includes a glossary of advanced concepts with links

## Environment

- Python 3.13, managed via `uv`
- Virtual environment via `uv venv`
- Git initialized from the start, meaningful commits at material checkpoints
- Code heavily commented for readability

## Project Structure

```
nowcasting_pe/
├── src/
│   ├── __init__.py
│   ├── simulation.py      # Synthetic PE fund DGP (eqs 1-9)
│   ├── ssm.py             # SSM matrix construction (eqs A.4-A.6)
│   ├── kalman.py           # Kalman filter (A.2) and smoother (A.3)
│   ├── naive.py            # Naive Rc-interpolated nowcast (eq A.10)
│   ├── estimation.py       # Profile likelihood + MLE + EM iterations
│   ├── metrics.py          # PME, RMSE metrics (in-sample, OOS, hybrid)
│   ├── garch.py            # GARCH(1,1) for h_t proxy
│   └── visualization.py    # All plotting functions
├── notebooks/
│   └── nowcasting_pe_nav.ipynb
├── docs/
│   └── plans/
├── pyproject.toml
└── paper.pdf
```

## Dependencies

numpy, scipy, pandas, matplotlib, arch (GARCH), jupyter, ipykernel

## Notebook Structure

### Table of Contents (navigable links)

1. **Introduction & Motivation** — The PE valuation problem; why NAVs are stale and smoothed
2. **The Data Generating Process** — Simulate a fund: returns (eq 1), comparable asset (eq 2), values (eq 3), cumulative returns (eq 4), NAV smoothing (eqs 5-6), distributions (eq 7), functional forms (eqs 8-9)
3. **The Naive Nowcast** — Rc-interpolated NAVs (eq A.10) as baseline benchmark
4. **State Space Model Formulation** — Observation equation, transition equation, SSM matrices (eqs A.4-A.6), handling missing data
5. **Kalman Filter & Smoother** — Forward recursion (eq A.2), backward recursion (eq A.3), Kalman gain, filtered vs smoothed returns
6. **Parameter Estimation** — Profile likelihood 15x15 grid for (α,β), MLE for (δ,λ,F,σn,σd), penalized likelihood, EM-like iterations for mapping function m_t
7. **Partial Imputation** — Peer-fund parameter imputation, β-anchoring, reduced parameter set
8. **Nowcasting Performance Metrics** — PME-based metrics, in-sample/OOS/hybrid RMSE, improvement rates
9. **Monte Carlo Experiments** — Panel of simulated funds, parameter recovery, nowcast performance under misspecification
10. **Sensitivity Analysis** — Effect of β, λ, F on nowcasts; crisis-period behavior
11. **Glossary & Appendix** — Brief explanations + Wikipedia links for: State Space Models, Kalman Filter, GARCH, MLE, Profile Likelihood, EM Algorithm, NAV Smoothing, PME, Nowcasting, MIDAS, Penalized Likelihood

### Notebook Pattern

Each section follows:
1. Markdown cell(s) explaining the concept in plain language with equations
2. Code cell(s) importing from `src/` and running the analysis
3. Visualization cell(s) showing results
4. Markdown cell with interpretation/takeaways

## Core Model Specification

### Fund Return Process (eq 1)
```
Rt = (α + β(Rmt - 1) + 1) * exp(ηt),  ηt ~ N(0, F² * ht)
```

### Comparable Asset (eq 2)
```
Rct = exp(rt * βc + ψ + ηct),  ηct ~ N(0, Fc² * ht)
```

### True Asset Values (eq 3)
```
Vt = Vt-1 * Rt - Dt + Ct,  V0 = C0 - D0 > 0
```

### Cumulative Returns & Mapping (eq 4)
```
R0:t = ∏τ=1..t Rτ ≡ Vt * Mt
```

### NAV Smoothing (eqs 5-6)
```
r̄0:t = (1 - λ(·)t) * r0:t + λ(·)t * r̄0:t-1
NAVt = exp(r̄0:t - mt + nt),  nt ~ N(0, σn²)
```

### Distribution Rule (eq 7)
```
Dt = δ(·)t * (Vt + Dt) * exp(dt),  dt ~ N(0, σd²)
```

### Functional Forms (eqs 8-9)
```
λ(·)t = λ * (1 - wt)
δ(·)t = min(0.99, δ * ty)
```

### SSM Matrices (eqs A.4-A.6)
- State vector: st = [rt, r0:t-1, r̄0:t-2, 1]'
- Observation vector: yt = [dt-1, navt-2, rct]' (with missing data)
- Regressor vector: xt = [m̂t-1, m̂t-2, 0]'
- Full Z, G, H, V, Q matrices as specified in the appendix

### Parameter Vector θ (10 parameters)
β, α, βc, ψ, F, Fc, λ, δ, σn, σd

### Estimation Procedure
1. Step 1: 15×15 (α,β) profile grid → for each: MLE of (δ,λ,F,σn,σd) with penalized log-likelihood
2. Step 2: Fix (α,β) at optimum → iteratively estimate remaining 8 params + update mapping function mt until convergence

### Performance Metrics
- In-Sample RMSE: squared deviation of PME(θ,T)0:t from 1
- OOS RMSE: squared deviation of PME(θ,T)τ:T from 1
- Hybrid RMSE: squared deviation of PME(θ,T)0:t from 1 using OOS data
- Improvement rate: fraction of funds where SSM RMSE < naive RMSE

## Simulated Fund Parameters (matching paper's Figure 1)

- 562 weeks of operating history
- True α = 0 (arithmetic)
- True β = 1.19
- λ = 0.9
- σn = 0.05
- F ≈ 2.0
- δ calibrated to produce ~25 distributions over fund life
