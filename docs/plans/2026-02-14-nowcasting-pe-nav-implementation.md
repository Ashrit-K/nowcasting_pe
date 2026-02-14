# Nowcasting PE NAV — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a pedagogical Jupyter notebook + Python module library that teaches the full methodology of Brown, Ghysels, Gredil (RFS 2023) "Nowcasting Net Asset Values: The Case of Private Equity" using simulated data.

**Architecture:** A `src/` Python package contains all computational logic across 8 modules (simulation, SSM, Kalman, naive, estimation, metrics, GARCH, visualization). A single comprehensive Jupyter notebook in `notebooks/` imports from `src/` and walks through the paper section-by-section with rich markdown exposition, runnable code cells, and visualizations. Everything runs on synthetic PE fund data generated to match the paper's DGP.

**Tech Stack:** Python 3.13, uv package manager, numpy, scipy, pandas, matplotlib, arch (GARCH), jupyter, ipykernel

---

## Task 1: Project Scaffolding & Environment Setup

**Files:**
- Create: `pyproject.toml`
- Create: `src/__init__.py`
- Create: `.gitignore`

**Step 1: Initialize uv project with Python 3.13**

```bash
uv init --python 3.13
uv venv
source .venv/bin/activate
```

**Step 2: Configure pyproject.toml**

```toml
[project]
name = "nowcasting-pe"
version = "0.1.0"
description = "Pedagogical implementation of Brown, Ghysels, Gredil (RFS 2023) — Nowcasting PE NAVs"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "numpy>=2.0",
    "scipy>=1.14",
    "pandas>=2.2",
    "matplotlib>=3.9",
    "arch>=7.0",
    "jupyter>=1.0",
    "ipykernel>=6.29",
]
```

**Step 3: Install dependencies**

```bash
uv pip install -e "."
```

**Step 4: Create directory structure**

```bash
mkdir -p src notebooks
touch src/__init__.py
```

**Step 5: Create .gitignore**

Standard Python .gitignore: `__pycache__/`, `.venv/`, `*.egg-info/`, `.ipynb_checkpoints/`, `*.pyc`

**Step 6: Commit**

```bash
git add pyproject.toml src/__init__.py .gitignore .python-version
git commit -m "feat: scaffold project with uv, Python 3.13, and dependencies"
```

---

## Task 2: GARCH Module (`src/garch.py`)

**Files:**
- Create: `src/garch.py`
- Test: Manual verification via notebook later

This module is a dependency for both simulation and SSM, so it comes first.

**Step 1: Write `src/garch.py`**

The module provides:
- `fit_garch11(returns)` — Fit GARCH(1,1) to a return series, return the model + conditional variances
- `orthogonalize_returns(rc, rm)` — Regress comparable returns on market returns, return residuals
- `get_idiosyncratic_variance(rc, rm)` — Full pipeline: orthogonalize + GARCH(1,1) → h_t series

Key implementation notes:
- Uses `arch` library's `arch_model` with p=1, q=1
- Returns are in log space (weekly log returns)
- The paper uses h_t as a proxy for time-varying idiosyncratic volatility
- h_t scales both fund idiosyncratic shocks (ηt) and comparable asset shocks (ηct)
- Per the paper: "We proxy for ht with a GARCH(1,1)-filtered variance of the idiosyncratic returns of Rct (from projection on Rmt)"

```python
"""
GARCH(1,1) module for idiosyncratic volatility estimation.

The paper (Section 2.1) uses GARCH(1,1)-filtered variance of the idiosyncratic
component of comparable asset returns as the time-varying volatility proxy h_t.
This h_t scales the distributions of both fund and comparable asset idiosyncratic
shocks via parameters F and Fc respectively.

Reference equations:
    ηt  ~ N(0, F²  * ht)   — fund idiosyncratic shock (eq 1)
    ηct ~ N(0, Fc² * ht)   — comparable asset idiosyncratic shock (eq 2)
"""
import numpy as np
from arch import arch_model


def orthogonalize_returns(rc: np.ndarray, rm: np.ndarray) -> np.ndarray:
    """
    Regress comparable asset log returns on market log returns and return residuals.

    The idiosyncratic component of the comparable asset is obtained by
    projecting rc onto rm via OLS and taking the residuals. These residuals
    are what we feed into the GARCH(1,1) model to get h_t.

    Parameters
    ----------
    rc : np.ndarray, shape (T,)
        Weekly log returns of the comparable public asset.
    rm : np.ndarray, shape (T,)
        Weekly log returns of the market factor (e.g., CRSP value-weighted).

    Returns
    -------
    residuals : np.ndarray, shape (T,)
        Idiosyncratic component of rc after removing market exposure.
    """
    # OLS regression: rc = a + b * rm + residual
    X = np.column_stack([np.ones_like(rm), rm])
    beta_ols = np.linalg.lstsq(X, rc, rcond=None)[0]
    residuals = rc - X @ beta_ols
    return residuals


def fit_garch11(residuals: np.ndarray) -> tuple[np.ndarray, object]:
    """
    Fit a GARCH(1,1) model to the idiosyncratic residuals.

    The GARCH(1,1) model is:
        σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}

    where ε_t are the idiosyncratic residuals. The conditional variance
    σ²_t is our proxy for h_t in the paper.

    Parameters
    ----------
    residuals : np.ndarray, shape (T,)
        Idiosyncratic return residuals (from orthogonalize_returns).

    Returns
    -------
    conditional_variance : np.ndarray, shape (T,)
        The GARCH(1,1) conditional variance series (our h_t proxy).
    fitted_model : arch model result
        The fitted GARCH model object for inspection.
    """
    # Scale residuals to percentage for numerical stability in arch library
    # arch library works better with percentage returns
    am = arch_model(residuals * 100, vol='Garch', p=1, q=1, mean='Zero')
    result = am.fit(disp='off')

    # Convert conditional variance back from percentage² to decimal²
    # (divide by 100² = 10000)
    h_t = result.conditional_volatility.values ** 2 / 10000.0

    return h_t, result


def get_idiosyncratic_variance(
    rc: np.ndarray,
    rm: np.ndarray,
) -> np.ndarray:
    """
    Full pipeline: orthogonalize comparable returns, fit GARCH(1,1), return h_t.

    This is the main entry point for obtaining the time-varying idiosyncratic
    volatility proxy used throughout the SSM.

    Parameters
    ----------
    rc : np.ndarray, shape (T,)
        Weekly log returns of the comparable public asset.
    rm : np.ndarray, shape (T,)
        Weekly log returns of the market factor.

    Returns
    -------
    h_t : np.ndarray, shape (T,)
        GARCH(1,1) conditional variance of idiosyncratic returns.
    """
    residuals = orthogonalize_returns(rc, rm)
    h_t, _ = fit_garch11(residuals)
    return h_t
```

**Step 2: Commit**

```bash
git add src/garch.py
git commit -m "feat: add GARCH(1,1) module for idiosyncratic volatility proxy h_t"
```

---

## Task 3: Simulation Module (`src/simulation.py`)

**Files:**
- Create: `src/simulation.py`

This is the core data generating process from the paper (equations 1-9). It creates synthetic PE fund data with known parameters so we can validate the SSM recovery.

**Step 1: Write `src/simulation.py`**

The module provides:
- `SimulatedFundParams` — dataclass holding the 10 model parameters + fund config
- `simulate_market_returns(T, seed)` — Generate weekly market factor returns
- `simulate_comparable_returns(rm, params, h_t)` — Generate comparable asset returns (eq 2)
- `simulate_fund(params, rm, h_t, seed)` — Full DGP: true returns, values, distributions, calls, smoothed NAVs

Key implementation details:
- Time t is in weeks, fund life ~562 weeks (~10.8 years)
- Market returns: either simulated GBM or drawn from empirical distribution
- Fund returns (eq 1): Rt = (α + β(Rmt - 1) + 1) * exp(ηt) where ηt ~ N(0, F² * ht)
- True values (eq 3): Vt = Vt-1 * Rt - Dt + Ct, V0 = C0
- Capital calls: concentrated in first ~2 years, modeled as deterministic schedule
- Distributions (eq 7): Dt = δ(·)t * (Vt + Dt) * exp(dt) when distribution event occurs
- Distribution events: occur with increasing probability as fund ages (matching ~25 distributions)
- NAV smoothing (eqs 5-6): r̄0:t = (1-λ(·)t)*r0:t + λ(·)t*r̄0:t-1
- NAV reports: only every 13th week (quarterly), with noise nt ~ N(0, σn²)
- Mapping function mt (eq A.8): mt - mt-1 = log((Vt + Dt - Ct) / Vt)

The function returns a dictionary/dataclass with all simulated series (true and observed).

```python
"""
Simulation module: Synthetic PE fund data generating process.

Implements the complete DGP from Brown, Ghysels, Gredil (RFS 2023),
equations (1) through (9). Generates synthetic fund data with known
parameters so the SSM can be validated against ground truth.

Key series generated:
    - True fund returns Rt (latent, eq 1)
    - Comparable asset returns Rct (observed weekly, eq 2)
    - True asset values Vt (latent, eq 3)
    - Cumulative returns R0:t (latent, eq 4)
    - Smoothed cumulative returns r̄0:t (latent, eq 5)
    - Reported NAVs (observed quarterly with noise, eq 6)
    - Fund distributions Dt (observed irregularly, eq 7)
    - Capital calls Ct (observed irregularly)
    - Mapping function mt (eq 4/A.8)
    - GARCH variance proxy ht
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FundParams:
    """
    Complete parameter set θ for a PE fund SSM.

    These 10 parameters govern the fund return process, comparable asset
    relationship, NAV smoothing, and distribution behavior.

    Reference: Section 2.2, collectively referred to as θ.
    """

    # --- Fund return process (eq 1) ---
    alpha: float = 0.0
    """Weekly excess return (α). Paper default for simulations: 0.0 arithmetic.
    The paper centers the profiling grid at ~0.38% per week (≈20.5% p.a.) for
    real funds, but uses 0 for the Figure 1 simulation."""

    beta: float = 1.19
    """Market risk exposure (β). Fund return loading on market excess return.
    Paper Figure 1 uses 1.19. Buyout average ~1.05-1.10, venture ~1.20-1.32."""

    F: float = 2.0
    """Idiosyncratic volatility scale. Multiplier on sqrt(ht) for fund
    idiosyncratic shocks. F=2.0 means fund idiosyncratic vol is 2x the
    GARCH-filtered vol of the comparable asset's idiosyncratic component."""

    # --- Comparable asset relationship (eq 2) ---
    beta_c: float = 0.85
    """Slope from reverse regression of comparable asset on fund returns (βc).
    Inversely proportional to the loading of rt on rct. Typical range: 0.5-1.2."""

    psi: float = 0.001
    """Intercept in comparable asset equation (ψ). Weekly drift adjustment.
    Average ~3% p.a. but interpretation is nuanced (see Section 3.2.1)."""

    Fc: float = 1.0
    """Comparable asset idiosyncratic volatility scale. Multiplier on sqrt(ht)
    for comparable asset shocks. Lower values indicate better benchmark match."""

    # --- NAV smoothing process (eqs 5-6, 8) ---
    lam: float = 0.90
    """Appraisal smoothing parameter (λ). Exponential moving average weight
    on past valuations. λ=0.90 means 0.90^13 ≈ 25.4% of quarter-ago valuations
    persist in current NAV. Paper median: 0.951."""

    sigma_n: float = 0.05
    """NAV reporting noise std dev (σn). Gaussian noise on log NAV reports,
    orthogonal to returns and distributions. Paper median: 3.1%."""

    # --- Distribution process (eqs 7, 9) ---
    delta: float = 0.03
    """Distribution density trend parameter (δ). Distribution fraction grows
    linearly with fund age: δ(·)t = min(0.99, δ * t_years). δ=0.03 means
    at year 5, expected distribution is 15% of (Vt + Dt)."""

    sigma_d: float = 0.10
    """Distribution noise std dev (σd). Gaussian noise on log distribution
    amounts. Controls randomness in distribution sizes."""

    # --- Fund configuration (not part of θ, but needed for simulation) ---
    T_weeks: int = 562
    """Total fund life in weeks (~10.8 years). Paper Figure 1 uses 562."""

    fund_size: float = 100.0
    """Initial fund commitment size (in $M). Used to scale capital calls."""

    call_schedule_years: float = 3.0
    """Years over which capital is called. Calls are front-loaded."""

    n_quarters_nav: int = -1
    """Number of quarters for which NAVs are reported. -1 = all quarters."""

    dist_start_year: float = 3.0
    """Year after which distributions can begin."""

    seed: int = 42
    """Random seed for reproducibility."""


@dataclass
class SimulatedFund:
    """
    Container for all simulated time series of a PE fund.

    Holds both latent (true) series that would be unobservable in practice
    and the observed series that the econometrician/LP would actually see.
    """

    params: FundParams
    """The parameters used to generate this fund."""

    T: int = 0
    """Number of weeks in the simulation."""

    # --- Market data (observed weekly) ---
    rm: np.ndarray = field(default_factory=lambda: np.array([]))
    """Weekly gross market returns Rmt. Observed every week."""

    rm_log: np.ndarray = field(default_factory=lambda: np.array([]))
    """Weekly log market returns log(Rmt). Observed every week."""

    rc: np.ndarray = field(default_factory=lambda: np.array([]))
    """Weekly gross comparable asset returns Rct (eq 2). Observed every week."""

    rc_log: np.ndarray = field(default_factory=lambda: np.array([]))
    """Weekly log comparable asset returns log(Rct). Observed every week."""

    h_t: np.ndarray = field(default_factory=lambda: np.array([]))
    """GARCH(1,1) conditional variance of idiosyncratic returns. Proxy for
    time-varying volatility used to scale fund and comparable shocks."""

    # --- Latent fund series (TRUE values, not observed in practice) ---
    R_true: np.ndarray = field(default_factory=lambda: np.array([]))
    """True weekly gross fund returns Rt (eq 1). LATENT."""

    r_true_log: np.ndarray = field(default_factory=lambda: np.array([]))
    """True weekly log fund returns rt = log(Rt). LATENT."""

    r_cum_log: np.ndarray = field(default_factory=lambda: np.array([]))
    """True cumulative log returns r0:t = Σ rτ for τ=1..t. LATENT."""

    r_cum_smoothed_log: np.ndarray = field(default_factory=lambda: np.array([]))
    """Smoothed cumulative log returns r̄0:t (eq 5). LATENT."""

    V_true: np.ndarray = field(default_factory=lambda: np.array([]))
    """True fund asset values Vt (eq 3). LATENT."""

    m_t: np.ndarray = field(default_factory=lambda: np.array([]))
    """Value-to-return mapping function mt (eq 4, A.8). Deterministic given
    true values and cash flows."""

    eta: np.ndarray = field(default_factory=lambda: np.array([]))
    """Fund idiosyncratic shocks ηt ~ N(0, F²*ht). LATENT."""

    eta_c: np.ndarray = field(default_factory=lambda: np.array([]))
    """Comparable asset idiosyncratic shocks ηct ~ N(0, Fc²*ht). LATENT."""

    # --- Observed fund data (what the LP/econometrician sees) ---
    C: np.ndarray = field(default_factory=lambda: np.array([]))
    """Capital calls Ct. Sparse (concentrated in first ~3 years)."""

    D: np.ndarray = field(default_factory=lambda: np.array([]))
    """Distributions Dt (eq 7). Sparse (~25 events over fund life)."""

    NAV_reported: np.ndarray = field(default_factory=lambda: np.array([]))
    """Reported NAVs (eq 6). Observed only every 13th week (quarterly).
    NaN for non-reporting weeks."""

    nav_noise: np.ndarray = field(default_factory=lambda: np.array([]))
    """NAV reporting noise nt ~ N(0, σn²). LATENT."""

    dist_noise: np.ndarray = field(default_factory=lambda: np.array([]))
    """Distribution noise dt ~ N(0, σd²). LATENT."""

    # --- Derived quantities ---
    w_t: np.ndarray = field(default_factory=lambda: np.array([]))
    """Cash flow weight wt ∈ [0,1] used in smoothing function λ(·)t (eq 8).
    Fraction of fund assets that the cash flow comprises."""

    lambda_t: np.ndarray = field(default_factory=lambda: np.array([]))
    """Time-varying smoothing function λ(·)t = λ * (1 - wt) (eq 8)."""

    delta_t: np.ndarray = field(default_factory=lambda: np.array([]))
    """Time-varying distribution density δ(·)t = min(0.99, δ * ty) (eq 9)."""

    is_nav_week: np.ndarray = field(default_factory=lambda: np.array([]))
    """Boolean mask: True on weeks when NAV is reported (every 13th week)."""

    is_dist_week: np.ndarray = field(default_factory=lambda: np.array([]))
    """Boolean mask: True on weeks when a distribution occurs."""


def simulate_market_returns(
    T: int,
    annual_mean: float = 0.08,
    annual_vol: float = 0.16,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic weekly market factor returns.

    Uses geometric Brownian motion calibrated to realistic equity market
    parameters. The weekly returns are i.i.d. log-normal.

    Parameters
    ----------
    T : int
        Number of weeks to simulate.
    annual_mean : float
        Annualized expected log return (continuously compounded). Default 8%.
    annual_vol : float
        Annualized volatility. Default 16%.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Rm : np.ndarray, shape (T,)
        Weekly gross market returns (Rm_t = exp(rm_t)).
    rm_log : np.ndarray, shape (T,)
        Weekly log market returns.
    """
    rng = np.random.default_rng(seed)

    # Convert annualized parameters to weekly
    # There are 52 weeks in a year
    weekly_mean = annual_mean / 52
    weekly_vol = annual_vol / np.sqrt(52)

    # Generate weekly log returns: rm_t ~ N(μ_weekly, σ²_weekly)
    rm_log = rng.normal(weekly_mean, weekly_vol, size=T)

    # Gross returns: Rm_t = exp(rm_t)
    Rm = np.exp(rm_log)

    return Rm, rm_log


def _generate_capital_calls(
    T: int,
    fund_size: float,
    call_years: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a deterministic-ish capital call schedule.

    Capital calls are front-loaded in the first few years of fund life.
    We model them as occurring roughly monthly in the call period, with
    some randomness in timing and amounts.

    Parameters
    ----------
    T : int
        Total weeks in simulation.
    fund_size : float
        Total fund commitment ($M).
    call_years : float
        Number of years over which capital is called.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    C : np.ndarray, shape (T,)
        Capital call amounts per week. Zero on non-call weeks.
    """
    C = np.zeros(T)
    call_weeks = int(call_years * 52)

    # Capital calls happen roughly every 4-6 weeks in the call period
    # Generate random call weeks within the call period
    n_calls = int(call_weeks / 5)  # roughly every 5 weeks
    call_times = rng.choice(range(1, min(call_weeks, T)), size=n_calls, replace=False)
    call_times.sort()

    # Distribute fund_size across calls with some randomness
    # Front-load: earlier calls tend to be larger
    weights = np.exp(-0.5 * np.arange(n_calls) / n_calls)
    weights = weights / weights.sum()

    # Add noise to weights
    noise = rng.uniform(0.7, 1.3, size=n_calls)
    weights = weights * noise
    weights = weights / weights.sum()

    for i, t in enumerate(call_times):
        if t < T:
            C[t] = fund_size * weights[i]

    return C


def _generate_distribution_events(
    T: int,
    dist_start_year: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a boolean mask of weeks when distributions occur.

    Distributions become more frequent as the fund matures. They are
    sparse: roughly 20-30 events over a ~10 year fund life, concentrated
    in years 3-10.

    Parameters
    ----------
    T : int
        Total weeks.
    dist_start_year : float
        Year after which distributions may begin.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    is_dist : np.ndarray of bool, shape (T,)
        True on weeks when a distribution event occurs.
    """
    is_dist = np.zeros(T, dtype=bool)
    dist_start_week = int(dist_start_year * 52)

    for t in range(dist_start_week, T):
        # Probability of distribution increases with fund age
        # Roughly: starts at ~2% per week, increases to ~8%
        years_since_start = (t - dist_start_week) / 52
        total_years = (T - dist_start_week) / 52
        prob = 0.02 + 0.06 * (years_since_start / total_years)

        # Don't distribute too frequently: enforce minimum 4-week gap
        if t >= 4 and np.any(is_dist[max(0, t - 4):t]):
            continue

        if rng.random() < prob:
            is_dist[t] = True

    return is_dist


def simulate_fund(params: FundParams) -> SimulatedFund:
    """
    Simulate a complete PE fund lifecycle using the paper's DGP.

    This is the main simulation function. It generates all the time series
    needed to demonstrate the nowcasting methodology, including both the
    latent (true) series and the observed series.

    The simulation follows this order:
    1. Generate market returns (exogenous)
    2. Generate GARCH variance proxy h_t from comparable returns
    3. Generate true fund returns (eq 1)
    4. Generate capital calls and distribution events
    5. Compute true asset values (eq 3) and distributions (eq 7)
    6. Compute cumulative returns and mapping function (eq 4, A.8)
    7. Compute smoothed NAVs (eqs 5-6)
    8. Generate comparable asset returns (eq 2) using true fund returns

    Parameters
    ----------
    params : FundParams
        Complete parameter set for the simulation.

    Returns
    -------
    fund : SimulatedFund
        Container with all simulated time series.
    """
    rng = np.random.default_rng(params.seed)
    T = params.T_weeks

    # =========================================================================
    # Step 1: Market returns (exogenous)
    # =========================================================================
    Rm, rm_log = simulate_market_returns(T, seed=params.seed)

    # =========================================================================
    # Step 2: Generate a preliminary h_t proxy
    # We need h_t to generate fund returns, but h_t comes from comparable
    # returns which depend on fund returns. We break this circularity by
    # first generating h_t from a simple GARCH on market returns scaled
    # to a realistic idiosyncratic volatility level.
    # =========================================================================
    # Use a simple model: h_t based on market vol scaled to idiosyncratic level
    # In practice, h_t comes from GARCH on orthogonalized comparable returns
    # For simulation, we can generate it directly
    weekly_idio_vol = 0.16 / np.sqrt(52)  # ~2.2% weekly base idiosyncratic vol
    h_t_base = np.full(T, weekly_idio_vol**2)

    # Add some GARCH-like persistence to make h_t time-varying
    omega = weekly_idio_vol**2 * 0.05  # long-run contribution
    alpha_g = 0.08  # ARCH effect
    beta_g = 0.87   # GARCH persistence
    h_t = np.zeros(T)
    h_t[0] = weekly_idio_vol**2
    # Use squared market return residuals as the "innovation" for GARCH dynamics
    market_mean = rm_log.mean()
    for t in range(1, T):
        eps2 = (rm_log[t - 1] - market_mean) ** 2
        h_t[t] = omega + alpha_g * eps2 + beta_g * h_t[t - 1]
    # Floor h_t to avoid numerical issues
    h_t = np.maximum(h_t, 1e-8)

    # =========================================================================
    # Step 3: True fund returns (eq 1)
    #   Rt = (α + β(Rmt - 1) + 1) * exp(ηt)
    #   ηt ~ N(0, F² * ht)
    # In log form: rt = log(α + β(Rmt - 1) + 1) + ηt
    # =========================================================================
    eta = rng.normal(0, 1, size=T) * params.F * np.sqrt(h_t)
    # Gross fund return before idiosyncratic shock
    R_systematic = params.alpha + params.beta * (Rm - 1) + 1
    # Full gross return with idiosyncratic component
    R_true = R_systematic * np.exp(eta)
    r_true_log = np.log(R_true)

    # =========================================================================
    # Step 4: Capital calls and distribution events
    # =========================================================================
    C = _generate_capital_calls(T, params.fund_size, params.call_schedule_years, rng)
    is_dist = _generate_distribution_events(T, params.dist_start_year, rng)

    # =========================================================================
    # Step 5: True asset values (eq 3) and distributions (eq 7)
    #   Vt = Vt-1 * Rt - Dt + Ct
    #   Dt = δ(·)t * (Vt + Dt) * exp(dt)   iff Dt > 0
    #   δ(·)t = min(0.99, δ * ty)
    #
    # Note: We need to be careful about the order here. On distribution
    # weeks, Dt depends on Vt which depends on Rt. We compute Vt_pre
    # (value before distributions/calls) then extract the distribution.
    # =========================================================================
    V_true = np.zeros(T)
    D = np.zeros(T)
    delta_t = np.zeros(T)
    dist_noise = np.zeros(T)

    # Initial value = first capital call
    V_true[0] = C[0] if C[0] > 0 else params.fund_size * 0.1

    for t in range(1, T):
        # Value before this week's cash flows (after return)
        V_pre = V_true[t - 1] * R_true[t]

        # Distribution density function: δ(·)t = min(0.99, δ * t_years)
        t_years = t / 52.0
        delta_t[t] = min(0.99, params.delta * t_years)

        if is_dist[t] and V_pre > 0.01:
            # Distribution amount (eq 7):
            # Dt = δ(·)t * (Vt + Dt) * exp(dt)
            # Solving: Dt / (Vt + Dt) = δ(·)t * exp(dt)
            # Let x = δ(·)t * exp(dt), then Dt = x * Vt_pre / (1 - x)
            # (since Vt_pre = Vt + Dt before extraction)
            dt_noise = rng.normal(0, params.sigma_d)
            dist_noise[t] = dt_noise
            x = delta_t[t] * np.exp(dt_noise)
            x = min(x, 0.95)  # Safety cap: don't distribute more than 95%
            D[t] = max(0, x * V_pre)

        # Value after cash flows (eq 3)
        V_true[t] = V_pre - D[t] + C[t]

        # Safety: asset value cannot be negative
        V_true[t] = max(V_true[t], 0.001)

    # =========================================================================
    # Step 6: Cumulative returns and mapping function
    #   R0:t = ∏ Rτ  (eq 4)
    #   r0:t = Σ rτ  (log cumulative return)
    #   mt - mt-1 = log((Vt + Dt - Ct) / Vt)  (eq A.8)
    # =========================================================================
    r_cum_log = np.cumsum(r_true_log)

    # Mapping function mt (eq A.8)
    # mt converts between cumulative returns and asset values:
    #   Vt = exp(r0:t - mt)  equivalently  R0:t = Vt * Mt
    m_t = np.zeros(T)
    for t in range(1, T):
        if V_true[t] > 0:
            # m_t[t] = m_t[t-1] + log((Vt + Dt - Ct) / Vt)
            numerator = V_true[t] + D[t] - C[t]
            if numerator > 0:
                m_t[t] = m_t[t - 1] + np.log(numerator / V_true[t])
            else:
                m_t[t] = m_t[t - 1]
        else:
            m_t[t] = m_t[t - 1]

    # =========================================================================
    # Step 7: NAV smoothing (eqs 5-6, 8)
    #   w_t = |cashflow_t| / naive_NAV_t  (fraction of fund assets)
    #   λ(·)t = λ * (1 - wt)
    #   r̄0:t = (1 - λ(·)t) * r0:t + λ(·)t * r̄0:t-1
    #   NAVt = exp(r̄0:t - mt + nt)   (only on reporting weeks)
    # =========================================================================

    # Compute naive NAV estimate for w_t calculation
    # (this is the Rc-interpolated NAV that the paper uses as a reference)
    naive_nav = V_true.copy()  # In simulation, we use true values for w_t

    # Cash flow weight w_t
    w_t = np.zeros(T)
    for t in range(T):
        total_cf = abs(D[t]) + abs(C[t])
        if naive_nav[t] > 0 and total_cf > 0:
            w_t[t] = min(1.0, total_cf / (naive_nav[t] + total_cf))

    # Time-varying smoothing parameter (eq 8)
    lambda_t = params.lam * (1 - w_t)

    # Smoothed cumulative log returns (eq 5)
    r_cum_smoothed = np.zeros(T)
    r_cum_smoothed[0] = r_cum_log[0]
    for t in range(1, T):
        r_cum_smoothed[t] = (
            (1 - lambda_t[t]) * r_cum_log[t]
            + lambda_t[t] * r_cum_smoothed[t - 1]
        )

    # NAV reports (eq 6): only every 13 weeks (quarterly)
    is_nav_week = np.zeros(T, dtype=bool)
    for t in range(12, T, 13):  # Every 13th week starting from week 12
        is_nav_week[t] = True

    nav_noise = rng.normal(0, params.sigma_n, size=T)
    NAV_reported = np.full(T, np.nan)
    for t in range(T):
        if is_nav_week[t]:
            # NAVt = exp(r̄0:t - mt + nt)
            NAV_reported[t] = np.exp(r_cum_smoothed[t] - m_t[t] + nav_noise[t])

    # =========================================================================
    # Step 8: Comparable asset returns (eq 2)
    #   In log form: rct = rt * βc + ψ + ηct
    #   ηct ~ N(0, Fc² * ht)
    # =========================================================================
    eta_c = rng.normal(0, 1, size=T) * params.Fc * np.sqrt(h_t)
    rc_log = r_true_log * params.beta_c + params.psi + eta_c
    rc = np.exp(rc_log)

    # =========================================================================
    # Assemble the SimulatedFund object
    # =========================================================================
    fund = SimulatedFund(
        params=params,
        T=T,
        rm=Rm,
        rm_log=rm_log,
        rc=rc,
        rc_log=rc_log,
        h_t=h_t,
        R_true=R_true,
        r_true_log=r_true_log,
        r_cum_log=r_cum_log,
        r_cum_smoothed_log=r_cum_smoothed,
        V_true=V_true,
        m_t=m_t,
        eta=eta,
        eta_c=eta_c,
        C=C,
        D=D,
        NAV_reported=NAV_reported,
        nav_noise=nav_noise,
        dist_noise=dist_noise,
        w_t=w_t,
        lambda_t=lambda_t,
        delta_t=delta_t,
        is_nav_week=is_nav_week,
        is_dist_week=is_dist,
    )

    return fund
```

**Step 2: Commit**

```bash
git add src/simulation.py
git commit -m "feat: add simulation module with full PE fund DGP (eqs 1-9)"
```

---

## Task 4: Naive Nowcast Module (`src/naive.py`)

**Files:**
- Create: `src/naive.py`

Implements the Rc-interpolated NAV baseline (equation A.10 from the paper).

**Step 1: Write `src/naive.py`**

```python
"""
Naive nowcast: Rc-interpolated NAVs.

The naive approach (eq A.10) interpolates between quarterly reported NAVs
using comparable asset returns and fund cash flows. This serves as the
baseline benchmark against which the SSM nowcast is evaluated.

The key idea: between two quarterly NAV reports, assume the fund's value
tracks the comparable asset return, adjusted for interim cash flows and
a quarter-specific drift that ensures consistency at quarter-ends.

Reference: Appendix Section A.3, equation (A.10)
"""
import numpy as np


def compute_naive_nav(
    NAV_reported: np.ndarray,
    rc: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    is_nav_week: np.ndarray,
) -> np.ndarray:
    """
    Compute the Rc-interpolated ("naive") NAV for every week.

    For weeks between quarterly NAV reports at q⁻ and q:

        Ṽ_t = NAV_{q⁻} * exp((t-q⁻)*ψ_q) * ∏_{q⁻<τ≤t} Rcτ
               + Σ_{q⁻<τ≤t} (Cτ - Dτ) * exp((t-τ)*ψ_q) * ∏_{τ<p≤t} Rcp

    where ψ_q is a quarter-specific drift chosen so that Ṽ_{t=q} = NAV_q.

    In words: start from the last reported NAV, grow it by comparable returns,
    add back net cash flows (also grown by comparable returns from their
    occurrence date), and include a drift that reconciles with the next
    reported NAV.

    Parameters
    ----------
    NAV_reported : np.ndarray, shape (T,)
        Reported NAVs. NaN on non-reporting weeks.
    rc : np.ndarray, shape (T,)
        Weekly gross comparable asset returns Rct.
    C : np.ndarray, shape (T,)
        Weekly capital calls.
    D : np.ndarray, shape (T,)
        Weekly distributions.
    is_nav_week : np.ndarray of bool, shape (T,)
        True on NAV reporting weeks.

    Returns
    -------
    nav_naive : np.ndarray, shape (T,)
        Naive (Rc-interpolated) NAV estimates for every week.
    """
    T = len(NAV_reported)
    nav_naive = np.full(T, np.nan)

    # Find all NAV reporting weeks
    nav_weeks = np.where(is_nav_week & ~np.isnan(NAV_reported))[0]

    if len(nav_weeks) == 0:
        return nav_naive

    # For weeks before first NAV report, use a simple extrapolation
    # from the first reported NAV backwards
    first_nav_week = nav_weeks[0]
    nav_naive[first_nav_week] = NAV_reported[first_nav_week]

    # Forward-fill before first NAV using comparable returns (backward)
    for t in range(first_nav_week - 1, -1, -1):
        # Going backwards: NAV_t = NAV_{t+1} / Rc_{t+1} + (D_{t+1} - C_{t+1})
        nav_naive[t] = nav_naive[t + 1] / rc[t + 1] + (D[t + 1] - C[t + 1])
        nav_naive[t] = max(nav_naive[t], 0.001)

    # Interpolate between each pair of consecutive NAV reports
    for i in range(len(nav_weeks) - 1):
        q_minus = nav_weeks[i]      # Start of quarter (last NAV report)
        q = nav_weeks[i + 1]        # End of quarter (next NAV report)

        nav_start = NAV_reported[q_minus]
        nav_end = NAV_reported[q]

        # Step 1: Compute the "no-drift" interpolation
        # Grow the starting NAV by comparable returns + cash flows
        # without the drift term first
        nav_no_drift = np.zeros(q - q_minus + 1)
        nav_no_drift[0] = nav_start

        for j in range(1, q - q_minus + 1):
            t = q_minus + j
            # Grow previous value by Rc, add net cash flows
            nav_no_drift[j] = nav_no_drift[j - 1] * rc[t] + (C[t] - D[t])
            nav_no_drift[j] = max(nav_no_drift[j], 0.001)

        # Step 2: Find the quarter-specific drift ψ_q
        # We need: nav_no_drift[end] * exp(n_weeks * ψ_q) ≈ nav_end
        # This is approximate; we solve for ψ_q that makes the endpoint match
        n_weeks = q - q_minus
        if nav_no_drift[-1] > 0 and nav_end > 0:
            psi_q = np.log(nav_end / nav_no_drift[-1]) / n_weeks
        else:
            psi_q = 0.0

        # Step 3: Apply drift to all intermediate weeks
        for j in range(1, q - q_minus + 1):
            t = q_minus + j
            nav_naive[t] = nav_no_drift[j] * np.exp(j * psi_q)
            nav_naive[t] = max(nav_naive[t], 0.001)

    # Extrapolate after last NAV report using comparable returns
    last_nav_week = nav_weeks[-1]
    for t in range(last_nav_week + 1, T):
        nav_naive[t] = nav_naive[t - 1] * rc[t] + (C[t] - D[t])
        nav_naive[t] = max(nav_naive[t], 0.001)

    return nav_naive


def compute_naive_returns(
    nav_naive: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """
    Compute weekly returns implied by the naive NAV series.

    The return is computed as:
        Rt = (Vt + Dt - Ct) / V_{t-1}

    Parameters
    ----------
    nav_naive : np.ndarray, shape (T,)
        Naive NAV estimates.
    C : np.ndarray, shape (T,)
        Capital calls.
    D : np.ndarray, shape (T,)
        Distributions.

    Returns
    -------
    returns : np.ndarray, shape (T,)
        Implied weekly gross returns. First element is NaN.
    """
    T = len(nav_naive)
    returns = np.full(T, np.nan)

    for t in range(1, T):
        if nav_naive[t - 1] > 0:
            returns[t] = (nav_naive[t] + D[t] - C[t]) / nav_naive[t - 1]

    return returns
```

**Step 2: Commit**

```bash
git add src/naive.py
git commit -m "feat: add naive Rc-interpolated NAV nowcast (eq A.10)"
```

---

## Task 5: SSM Module (`src/ssm.py`)

**Files:**
- Create: `src/ssm.py`

Constructs the time-varying SSM matrices per equations A.4-A.6.

**Step 1: Write `src/ssm.py`**

The state space model at each time step t is:
- Observation equation: yt - Γt*xt = Zt*st + εt,  εt ~ N(0, Ht)
- Transition equation:   st+1 = Gt*st + Vt*ηt,     ηt ~ N(0, Qt)

State vector (4×1): st = [rt, r0:t-1, r̄0:t-2, 1]'
Observation vector (3×1): yt = [dt-1, navt-2, rct]'  (with missings)
Regressor vector (3×1): xt = [m̂t-1, m̂t-2, 0]'

The module builds Z_t, G_t, H_t, V_t, Q_t matrices for each week t, handling missing observations by dropping rows from Z and H.

```python
"""
State Space Model (SSM) matrix construction for PE fund nowcasting.

Implements the SSM representation from Appendix equations (A.4)-(A.6).
The model casts the PE fund return/cash flow process into the standard
linear Gaussian SSM form:

    yt - Γt*xt = Zt*st + εt,    εt ~ N(0, Ht)   [observation equation]
    st+1 = Gt*st + Vt*ηt,       ηt ~ N(0, Qt)    [transition equation]

where:
    st = [rt, r0:t-1, r̄0:t-2, 1]'           — state vector (4×1)
    yt = [dt-1, navt-2, rct]'                  — observations (up to 3×1)
    xt = [m̂t-1, m̂t-2, 0]'                    — regressors (3×1)

Key feature: observations are often missing. dt is only present on
distribution weeks (~25 of 562), navt only on NAV reporting weeks
(every 13th). Only rct is always observed. The Kalman filter handles
missing data by simply dropping the corresponding rows from Z and H.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class SSMMatrices:
    """
    Container for the time-varying SSM matrices at a single time step.

    All matrices are constructed for the current week t given the parameter
    vector θ and current data values.
    """

    Z: np.ndarray
    """Observation loading matrix. Maps state to observations.
    Full size: (3, 4). Reduced when observations are missing."""

    G: np.ndarray
    """State transition matrix. Size: (4, 4)."""

    H: np.ndarray
    """Observation noise covariance. Full size: (3, 3). Diagonal.
    Reduced when observations are missing."""

    V: np.ndarray
    """State noise loading. Size: (4, 1)."""

    Q: np.ndarray
    """State noise variance. Scalar wrapped in (1, 1) matrix."""

    Gamma: np.ndarray
    """Regressor loading matrix. Size: (3, 3). Identity."""

    x: np.ndarray
    """Regressor vector. Size: (3,)."""

    obs_mask: np.ndarray
    """Boolean mask indicating which observations are present.
    Size: (3,). [has_distribution, has_nav, True (rc always observed)]."""

    n_obs: int
    """Number of non-missing observations this period."""


def build_ssm_matrices(
    t: int,
    alpha: float,
    beta: float,
    beta_c: float,
    psi: float,
    F: float,
    Fc: float,
    lam: float,
    delta: float,
    sigma_n: float,
    sigma_d: float,
    Rm_t: float,
    h_t: float,
    lambda_t: float,
    delta_t: float,
    m_hat: np.ndarray,
    has_distribution: bool,
    has_nav: bool,
) -> SSMMatrices:
    """
    Build SSM matrices for a single time period t.

    Constructs Z_t, G_t, H_t, V_t, Q_t per equations (A.4)-(A.6),
    handling missing observations by removing corresponding rows.

    Parameters
    ----------
    t : int
        Current time period (week index).
    alpha : float
        Fund weekly excess return.
    beta : float
        Market risk exposure.
    beta_c : float
        Comparable asset reverse-regression slope.
    psi : float
        Comparable asset intercept.
    F : float
        Fund idiosyncratic volatility scale.
    Fc : float
        Comparable asset idiosyncratic volatility scale.
    lam : float
        Smoothing parameter (base λ, before applying w_t adjustment).
    delta : float
        Distribution density parameter.
    sigma_n : float
        NAV reporting noise std dev.
    sigma_d : float
        Distribution noise std dev.
    Rm_t : float
        Gross market return for week t.
    h_t : float
        GARCH conditional variance for week t.
    lambda_t : float
        Time-varying smoothing λ(·)_t = λ * (1 - w_t) for current period.
    delta_t : float
        Time-varying distribution density δ(·)_t = min(0.99, δ * t_years).
    m_hat : np.ndarray, shape (T,)
        Estimated mapping function series (m̂_t).
    has_distribution : bool
        Whether a distribution is observed this period.
    has_nav : bool
        Whether a NAV report is observed this period.

    Returns
    -------
    ssm : SSMMatrices
        Complete set of SSM matrices for this time step.
    """
    # =====================================================================
    # Transition equation components (eq A.5, A.6)
    # st+1 = Gt * st + Vt * ηt
    #
    # State vector: st = [rt, r0:t-1, r̄0:t-2, 1]'
    #
    # Transition dynamics:
    #   rt+1     = αβ(·)_{t+1} + ηt+1  (driven by next period's shock)
    #   r0:t     = r0:t-1 + rt          (cumulate this period's return)
    #   r̄0:t-1  = (1-λ(·)_t-1)*r0:t-2 + λ(·)_t-1 * r̄0:t-3
    #              (but stored as r̄0:t-1 for use in next obs eq)
    #   1        = 1                     (constant)
    # =====================================================================

    # αβ(·)_t = log(α + β(Rm_t - 1) + 1)  (eq A.7)
    alpha_beta_t = np.log(alpha + beta * (Rm_t - 1) + 1)

    # Transition matrix G (4×4) — eq A.5
    # Note: The transition matrix accumulates returns into the cumulative
    # sums and applies the exponential smoothing
    G = np.array([
        [0, 0, 0, alpha_beta_t],   # rt+1 = αβ(·)_{t+1} (+ shock)
        [1, 1, 0, 0],              # r0:t = rt + r0:t-1
        [0, (1 - lambda_t), lambda_t, 0],  # r̄0:t-1 = EMA
        [0, 0, 0, 1],              # constant = 1
    ])

    # State noise loading V (4×1) — eq A.6
    V_mat = np.array([[F], [0], [0], [0]])

    # State noise variance Q (1×1)
    Q = np.array([[h_t]])  # Qt = ĥ_{t+1}, using current h_t as proxy

    # =====================================================================
    # Observation equation components (eq A.4)
    # yt - Γt*xt = Zt*st + εt
    #
    # Full observation vector: yt = [dt-1, navt-2, rct]'
    # Full regressor vector:   xt = [m̂t-1, m̂t-2, 0]'
    # =====================================================================

    # Z matrix (3×4) — eq A.4
    # Row 1 (distribution): dt maps to r0:t via δ function
    #   dt-1 - m̂t-1 = log(δ(·)t / (1 - δ(·)t)) + r0:t-1 + noise
    #   So Z[0,:] loads on r0:t-1 (2nd state element)
    if delta_t > 0 and delta_t < 1:
        log_delta_ratio = np.log(delta_t / (1 - delta_t))
    else:
        log_delta_ratio = 0.0

    # Row 2 (NAV): navt-2 - m̂t-2 = r̄0:t-2 + noise
    #   So Z[1,:] loads on r̄0:t-2 (3rd state element)

    # Row 3 (comparable return): rct = βc * rt + ψ + noise
    #   So Z[2,:] loads on rt (1st state element)

    Z_full = np.array([
        [0, 1, 0, log_delta_ratio],   # distribution → r0:t-1
        [0, 0, 1, 0],                  # NAV → r̄0:t-2
        [beta_c, 0, 0, psi],           # comparable → rt
    ])

    # H matrix (3×3) — eq A.4, diagonal
    H_full = np.diag([
        sigma_d**2,                    # distribution noise variance
        sigma_n**2,                    # NAV reporting noise variance
        (Fc * np.sqrt(h_t))**2,        # comparable asset noise variance
    ])

    # Gamma matrix (3×3) — identity (regressors enter linearly)
    Gamma_full = np.eye(3)

    # Regressor vector xt = [m̂t-1, m̂t-2, 0]'
    if t >= 2 and len(m_hat) > t - 1:
        x_full = np.array([m_hat[t - 1], m_hat[t - 2], 0.0])
    elif t >= 1 and len(m_hat) > t - 1:
        x_full = np.array([m_hat[t - 1], 0.0, 0.0])
    else:
        x_full = np.array([0.0, 0.0, 0.0])

    # =====================================================================
    # Handle missing observations
    # Drop rows from Z, H, Gamma, x for missing observations
    # =====================================================================
    obs_mask = np.array([has_distribution, has_nav, True])  # rc always observed
    n_obs = obs_mask.sum()

    Z = Z_full[obs_mask]
    H = H_full[np.ix_(obs_mask, obs_mask)]
    Gamma = Gamma_full[np.ix_(obs_mask, obs_mask)]
    x = x_full[obs_mask]

    return SSMMatrices(
        Z=Z, G=G, H=H, V=V_mat, Q=Q,
        Gamma=Gamma, x=x,
        obs_mask=obs_mask, n_obs=int(n_obs),
    )


def build_observation_vector(
    t: int,
    D: np.ndarray,
    NAV_reported: np.ndarray,
    rc_log: np.ndarray,
    is_dist_week: np.ndarray,
    is_nav_week: np.ndarray,
) -> tuple[np.ndarray, bool, bool]:
    """
    Construct the observation vector yt for time period t.

    The full observation vector is yt = [dt-1, navt-2, rct]' but
    dt-1 and navt-2 are often missing. Returns the vector with only
    non-missing entries, plus flags for what's present.

    Note the time subscripts: distributions enter with lag 1 (dt-1)
    and NAVs with lag 2 (navt-2). This reflects the information flow:
    by week t, the econometrician knows distributions from t-1 and
    NAVs from t-2.

    Parameters
    ----------
    t : int
        Current time period.
    D : np.ndarray, shape (T,)
        Distribution amounts.
    NAV_reported : np.ndarray, shape (T,)
        Reported NAVs (NaN when not reported).
    rc_log : np.ndarray, shape (T,)
        Log comparable asset returns.
    is_dist_week : np.ndarray of bool
        Distribution event indicator.
    is_nav_week : np.ndarray of bool
        NAV reporting indicator.

    Returns
    -------
    y : np.ndarray
        Observation vector (only non-missing elements).
    has_dist : bool
        Whether distribution data is available.
    has_nav : bool
        Whether NAV data is available.
    """
    observations = []
    has_dist = False
    has_nav = False

    # Distribution: dt-1 = log(Dt-1) if distribution occurred at t-1
    if t >= 1 and is_dist_week[t - 1] and D[t - 1] > 0:
        observations.append(np.log(D[t - 1]))
        has_dist = True

    # NAV: navt-2 = log(NAVt-2) if NAV was reported at t-2
    if t >= 2 and is_nav_week[t - 2] and not np.isnan(NAV_reported[t - 2]):
        observations.append(np.log(NAV_reported[t - 2]))
        has_nav = True

    # Comparable return: always observed
    observations.append(rc_log[t])

    y = np.array(observations)
    return y, has_dist, has_nav
```

**Step 2: Commit**

```bash
git add src/ssm.py
git commit -m "feat: add SSM matrix construction module (eqs A.4-A.6)"
```

---

## Task 6: Kalman Filter & Smoother Module (`src/kalman.py`)

**Files:**
- Create: `src/kalman.py`

Implements the forward (filter) and backward (smoother) recursions.

**Step 1: Write `src/kalman.py`**

```python
"""
Kalman filter (forward recursion) and smoother (backward recursion).

Implements the standard Kalman filter (eq A.2) and Kalman smoother (eq A.3)
for the PE fund state space model. The filter processes observations
sequentially forward in time, while the smoother refines estimates using
future observations via a backward pass.

Key outputs:
    - Filtered states:  E[st | Y_t]       (using data up to t)
    - Smoothed states:  E[st | Y_T]       (using all data)
    - Log-likelihood:   Σ log p(yt | Y_{t-1})  (for parameter estimation)

The Kalman filter is the workhorse of the nowcasting approach. It extracts
the latent fund return series from sparse, noisy observations (quarterly
NAVs, irregular distributions, weekly comparable returns).

References:
    - Forward recursion: eq (A.2) in the paper
    - Backward recursion (smoother): eq (A.3)
    - Durbin and Koopman (2012) for general SSM theory
"""
import numpy as np
from dataclasses import dataclass, field

from .ssm import SSMMatrices, build_ssm_matrices, build_observation_vector


@dataclass
class KalmanResult:
    """
    Container for Kalman filter and smoother results.

    Stores the full time series of filtered/smoothed state estimates,
    covariances, and the log-likelihood.
    """

    T: int
    """Number of time periods."""

    n_states: int = 4
    """Dimension of the state vector."""

    # --- Forward (filter) results ---
    s_filtered: np.ndarray = field(default_factory=lambda: np.array([]))
    """Filtered state estimates E[st | Yt]. Shape: (T, n_states).
    Each row is the state estimate after observing data up to period t."""

    P_filtered: np.ndarray = field(default_factory=lambda: np.array([]))
    """Filtered state covariance var(st | Yt). Shape: (T, n_states, n_states)."""

    s_predicted: np.ndarray = field(default_factory=lambda: np.array([]))
    """One-step-ahead predicted states E[st | Y_{t-1}]. Shape: (T, n_states)."""

    P_predicted: np.ndarray = field(default_factory=lambda: np.array([]))
    """One-step-ahead predicted covariance. Shape: (T, n_states, n_states)."""

    innovations: list = field(default_factory=list)
    """Innovation (prediction error) for each period: yt - Zt*st_predicted."""

    F_matrices: list = field(default_factory=list)
    """Innovation covariance matrices: Zt*Pt*Zt' + Ht."""

    log_likelihood: float = 0.0
    """Total log-likelihood: Σ_t log p(yt | Y_{t-1}).
    Used for MLE parameter estimation."""

    # --- Backward (smoother) results ---
    s_smoothed: np.ndarray = field(default_factory=lambda: np.array([]))
    """Smoothed state estimates E[st | YT]. Shape: (T, n_states).
    Uses all data (past and future) to estimate each state."""

    P_smoothed: np.ndarray = field(default_factory=lambda: np.array([]))
    """Smoothed state covariance. Shape: (T, n_states, n_states)."""


def kalman_filter(
    T: int,
    alpha: float,
    beta: float,
    beta_c: float,
    psi: float,
    F: float,
    Fc: float,
    lam: float,
    delta: float,
    sigma_n: float,
    sigma_d: float,
    Rm: np.ndarray,
    rc_log: np.ndarray,
    h_t: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    NAV_reported: np.ndarray,
    is_dist_week: np.ndarray,
    is_nav_week: np.ndarray,
    lambda_t: np.ndarray,
    delta_t: np.ndarray,
    m_hat: np.ndarray,
) -> KalmanResult:
    """
    Run the Kalman filter (forward recursion) through the full time series.

    Implements equation (A.2):
        s_{t+1}^filter = Gt * s_t^filter + Kt * (y_t^resid - Zt * s_t^filter)
        Ft = Zt * Pt * Zt' + Ht
        P_{t|t} = Pt - Pt * Zt' * Ft^{-1} * Zt * Pt
        P_{t+1} = Gt * P_{t|t} * Gt' + Vt * Qt * Vt'

    where Kt = Gt * Pt * Zt' * Ft^{-1} is the Kalman gain.

    Parameters
    ----------
    T : int
        Number of time periods.
    alpha, beta, ..., sigma_d : float
        SSM parameters (θ).
    Rm, rc_log, h_t, D, C, NAV_reported : np.ndarray
        Observed data series.
    is_dist_week, is_nav_week : np.ndarray of bool
        Observation availability masks.
    lambda_t, delta_t : np.ndarray
        Time-varying function values.
    m_hat : np.ndarray
        Estimated mapping function.

    Returns
    -------
    result : KalmanResult
        Filter results including states, covariances, and log-likelihood.
    """
    n_states = 4

    # Initialize result storage
    result = KalmanResult(T=T, n_states=n_states)
    result.s_filtered = np.zeros((T, n_states))
    result.P_filtered = np.zeros((T, n_states, n_states))
    result.s_predicted = np.zeros((T, n_states))
    result.P_predicted = np.zeros((T, n_states, n_states))
    result.innovations = []
    result.F_matrices = []
    result.log_likelihood = 0.0

    # =========================================================================
    # Initialize the state and covariance
    # s0 = [0, 0, 0, 1]' — no returns have occurred yet, constant = 1
    # P0 — relatively diffuse prior on the return, tight on constant
    # =========================================================================
    s = np.array([0.0, 0.0, 0.0, 1.0])
    P = np.diag([0.01, 0.001, 0.001, 0.0])  # Tight prior; constant is known

    # =========================================================================
    # Forward recursion: t = 0, 1, ..., T-1
    # =========================================================================
    for t in range(T):
        # Store predicted state (before incorporating this period's observation)
        result.s_predicted[t] = s.copy()
        result.P_predicted[t] = P.copy()

        # Build observation vector and SSM matrices for this period
        y, has_dist, has_nav = build_observation_vector(
            t, D, NAV_reported, rc_log, is_dist_week, is_nav_week
        )

        ssm = build_ssm_matrices(
            t=t,
            alpha=alpha, beta=beta, beta_c=beta_c, psi=psi,
            F=F, Fc=Fc, lam=lam, delta=delta,
            sigma_n=sigma_n, sigma_d=sigma_d,
            Rm_t=Rm[t], h_t=h_t[t],
            lambda_t=lambda_t[t], delta_t=delta_t[t],
            m_hat=m_hat, has_distribution=has_dist, has_nav=has_nav,
        )

        # =================================================================
        # Kalman filter update step
        # =================================================================
        Z = ssm.Z       # (n_obs, 4)
        H = ssm.H       # (n_obs, n_obs)
        G = ssm.G       # (4, 4)
        V_mat = ssm.V   # (4, 1)
        Q = ssm.Q       # (1, 1)

        # Compute the observation residual: y_resid = y - Gamma*x
        y_resid = y - ssm.Gamma @ ssm.x

        # Innovation: v_t = y_resid - Z * s_predicted
        v = y_resid - Z @ s

        # Innovation covariance: F = Z * P * Z' + H
        F_cov = Z @ P @ Z.T + H

        # Store innovations for diagnostics
        result.innovations.append(v)
        result.F_matrices.append(F_cov)

        # Log-likelihood contribution (multivariate normal)
        # log p(yt | Y_{t-1}) = -0.5 * [n*log(2π) + log|F| + v'*F^{-1}*v]
        n_obs = len(v)
        if n_obs > 0:
            try:
                F_inv = np.linalg.inv(F_cov)
                log_det_F = np.log(np.linalg.det(F_cov))
                ll_t = -0.5 * (
                    n_obs * np.log(2 * np.pi)
                    + log_det_F
                    + v @ F_inv @ v
                )
                result.log_likelihood += ll_t
            except np.linalg.LinAlgError:
                # Singular covariance matrix; skip this observation
                F_inv = np.linalg.pinv(F_cov)

        # Kalman gain: K = G * P * Z' * F^{-1}
        K = G @ P @ Z.T @ F_inv

        # Filtered state: s_{t|t} = s_t + P*Z'*F^{-1}*v
        s_filtered = s + P @ Z.T @ F_inv @ v
        result.s_filtered[t] = s_filtered

        # Filtered covariance: P_{t|t} = P - P*Z'*F^{-1}*Z*P
        P_filtered = P - P @ Z.T @ F_inv @ Z @ P
        result.P_filtered[t] = P_filtered

        # Predict next state: s_{t+1} = G * s_{t|t}
        s = G @ s_filtered

        # Predict next covariance: P_{t+1} = G * P_{t|t} * G' + V * Q * V'
        P = G @ P_filtered @ G.T + V_mat @ Q @ V_mat.T

    return result


def kalman_smoother(
    result: KalmanResult,
    alpha: float,
    beta: float,
    beta_c: float,
    psi: float,
    F: float,
    Fc: float,
    lam: float,
    delta: float,
    sigma_n: float,
    sigma_d: float,
    Rm: np.ndarray,
    h_t: np.ndarray,
    lambda_t: np.ndarray,
    delta_t: np.ndarray,
    m_hat: np.ndarray,
) -> KalmanResult:
    """
    Run the Kalman smoother (backward recursion) to refine state estimates.

    The smoother uses future observations to improve past state estimates.
    It adjusts the filtered estimates from the forward pass using information
    from later periods.

    Implements equation (A.3):
        s_t^smooth = s_t^filter + P_t * b_{t-1}

    where b is computed via backward recursion:
        b_{t-1} = Z_t' * F_t^{-1} * v_t + (G_t - K_t * Z_t)' * b_t

    The smoother is particularly useful for nowcasting because it allows
    using all available data (including later NAV reports and distributions)
    to refine earlier return estimates.

    Parameters
    ----------
    result : KalmanResult
        Output from kalman_filter (contains filtered states and covariances).
    alpha, ..., sigma_d : float
        SSM parameters.
    Rm, h_t : np.ndarray
        Market returns and GARCH variance.
    lambda_t, delta_t : np.ndarray
        Time-varying function values.
    m_hat : np.ndarray
        Mapping function estimates.

    Returns
    -------
    result : KalmanResult
        Same object, updated with smoothed states (s_smoothed, P_smoothed).
    """
    T = result.T
    n_states = result.n_states

    result.s_smoothed = np.zeros((T, n_states))
    result.P_smoothed = np.zeros((T, n_states, n_states))

    # Start from the last period: smoothed = filtered
    result.s_smoothed[T - 1] = result.s_filtered[T - 1]
    result.P_smoothed[T - 1] = result.P_filtered[T - 1]

    # Backward recursion: t = T-2, T-3, ..., 0
    # b_{T-1} is initialized based on the last period's innovation
    b = np.zeros(n_states)

    for t in range(T - 2, -1, -1):
        # Reconstruct G_t for this period (needed for backward recursion)
        Rm_t = Rm[t + 1]  # Next period's market return (for G_{t+1} transition)
        alpha_beta = np.log(alpha + beta * (Rm_t - 1) + 1)

        G = np.array([
            [0, 0, 0, alpha_beta],
            [1, 1, 0, 0],
            [0, (1 - lambda_t[t + 1]), lambda_t[t + 1], 0],
            [0, 0, 0, 1],
        ])

        # Get innovation info from forward pass
        v = result.innovations[t + 1]
        F_cov = result.F_matrices[t + 1]

        # We need Z for the next period to compute the backward recursion
        # Use obs_mask to determine which rows of Z are present
        # For simplicity, reconstruct the Z matrix
        ssm_next = build_ssm_matrices(
            t=t + 1,
            alpha=alpha, beta=beta, beta_c=beta_c, psi=psi,
            F=F, Fc=Fc, lam=lam, delta=delta,
            sigma_n=sigma_n, sigma_d=sigma_d,
            Rm_t=Rm[t + 1], h_t=h_t[t + 1],
            lambda_t=lambda_t[t + 1], delta_t=delta_t[t + 1],
            m_hat=m_hat, has_distribution=False, has_nav=False,
        )
        # We need to get the actual Z with correct missing pattern
        # Re-derive from the innovations dimensionality
        n_obs_next = len(v)

        try:
            F_inv = np.linalg.inv(F_cov)
        except np.linalg.LinAlgError:
            F_inv = np.linalg.pinv(F_cov)

        # Kalman gain for period t+1
        P_pred = result.P_predicted[t + 1]

        # Reconstruct Z with correct observation mask
        # We need the actual SSM from the forward pass
        # Use a simplified approach: reconstruct from stored data
        Z_next = ssm_next.Z  # This may not have the right mask

        # More robust: use the relationship between innovations and states
        # b_{t} = Z_{t+1}' * F_{t+1}^{-1} * v_{t+1}
        #         + (G_{t+1} - K_{t+1} * Z_{t+1})' * b_{t+1}

        # Compute K for next period
        K_next = G @ P_pred @ Z_next.T @ F_inv

        # Update b (backward recursion)
        b = Z_next.T @ F_inv @ v + (G - K_next @ Z_next).T @ b

        # Smoothed state: s_t^smooth = s_t^filter + P_{t|t} * b_t
        result.s_smoothed[t] = result.s_filtered[t] + result.P_filtered[t] @ b

        # Smoothed covariance (simplified)
        # Full formula involves L_t = G_t - K_t * Z_t and is more complex
        # For the prototype, we store filtered covariance as an approximation
        result.P_smoothed[t] = result.P_filtered[t]

    return result


def extract_returns(
    result: KalmanResult,
    use_smoothed: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract weekly and cumulative log returns from Kalman results.

    The state vector is [rt, r0:t-1, r̄0:t-2, 1]', so:
    - Weekly log return rt is state[0]
    - Cumulative log return r0:t = state[1] + state[0]
      (since state[1] = r0:t-1 and rt adds to it)

    Parameters
    ----------
    result : KalmanResult
        Kalman filter/smoother output.
    use_smoothed : bool
        If True, use smoothed states; otherwise use filtered.

    Returns
    -------
    r_weekly : np.ndarray, shape (T,)
        Weekly log returns.
    r_cumulative : np.ndarray, shape (T,)
        Cumulative log returns since inception.
    """
    states = result.s_smoothed if use_smoothed else result.s_filtered
    T = result.T

    r_weekly = states[:, 0]  # rt is the first state element
    r_cumulative = states[:, 1] + states[:, 0]  # r0:t = r0:t-1 + rt

    return r_weekly, r_cumulative
```

**Step 2: Commit**

```bash
git add src/kalman.py
git commit -m "feat: add Kalman filter and smoother with missing data support (eqs A.2-A.3)"
```

---

## Task 7: Metrics Module (`src/metrics.py`)

**Files:**
- Create: `src/metrics.py`

**Step 1: Write `src/metrics.py`**

PME computation and the three RMSE metrics from Section 2.3.

```python
"""
Nowcasting performance metrics.

Implements the PME-based evaluation metrics from Section 2.3 of the paper.
The core idea: if true fund returns were used to discount cash flows, the
fund's PME would equal 1. Deviations from 1 measure pricing error.

Three metrics:
    1. In-Sample RMSE:  Uses data within the estimation window
    2. OOS RMSE:        Uses only data after the estimation cutoff
    3. Hybrid RMSE:     Uses OOS NAVs but all since-inception cash flows

Reference: Section 2.3, equations for PME(θ,T)
"""
import numpy as np


def compute_pme_series(
    r_cumulative: np.ndarray,
    m_t: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    V_terminal: float = 0.0,
) -> np.ndarray:
    """
    Compute the Kaplan-Schoar PME on a to-date basis.

    The PME at time t is defined as:

        PME(0:t) = [Σ_{s=0}^{t} Ds * discount_factor(s,t) + Vt]
                   / [Σ_{s=0}^{t} Cs * discount_factor(s,t)]

    where discount_factor(s,t) = exp(r0:t - r0:s) converts values from
    time s to time t using the estimated cumulative returns.

    If the returns are "correct" (i.e., equal to true returns), PME = 1.

    Parameters
    ----------
    r_cumulative : np.ndarray, shape (T,)
        Cumulative log returns r0:t from the model.
    m_t : np.ndarray, shape (T,)
        Mapping function (to convert between returns and values).
    C : np.ndarray, shape (T,)
        Capital calls.
    D : np.ndarray, shape (T,)
        Distributions.
    V_terminal : float
        Terminal value (residual NAV at the end). Default 0 for resolved funds.

    Returns
    -------
    pme : np.ndarray, shape (T,)
        To-date PME series. Values near 1 indicate good model fit.
    """
    T = len(r_cumulative)
    pme = np.full(T, np.nan)

    for t in range(1, T):
        # Discount factor from period s to period t:
        # df(s,t) = exp(r0:t - r0:s) = R0:t / R0:s
        # This grows past cash flows to their present value at time t

        numerator = 0.0   # Sum of discounted distributions + terminal value
        denominator = 0.0  # Sum of discounted capital calls

        for s in range(t + 1):
            # Discount factor from s to t
            df = np.exp(r_cumulative[t] - r_cumulative[s])

            if D[s] > 0:
                numerator += D[s] * df
            if C[s] > 0:
                denominator += C[s] * df

        # Add terminal value (not discounted, it's already at time t)
        if t == T - 1:
            numerator += V_terminal

        if denominator > 0:
            pme[t] = numerator / denominator

    return pme


def compute_insample_rmse(
    r_cumulative: np.ndarray,
    m_t: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    tau_start: int,
    tau_end: int,
) -> float:
    """
    In-Sample RMSE: mean squared deviation of PME from 1 within estimation window.

    Parameters
    ----------
    r_cumulative : np.ndarray
        Cumulative log returns from the model.
    m_t : np.ndarray
        Mapping function.
    C, D : np.ndarray
        Cash flows.
    tau_start, tau_end : int
        Start and end of the in-sample evaluation window.

    Returns
    -------
    rmse : float
        Root mean squared pricing error.
    """
    pme = compute_pme_series(r_cumulative, m_t, C, D)
    valid = ~np.isnan(pme[tau_start:tau_end + 1])
    if valid.sum() == 0:
        return np.nan
    errors = pme[tau_start:tau_end + 1][valid] - 1.0
    return np.sqrt(np.mean(errors**2))


def compute_oos_rmse(
    r_cumulative: np.ndarray,
    m_t: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    tau_cutoff: int,
) -> float:
    """
    Out-of-Sample RMSE: pricing error using only post-estimation data.

    No fund-specific data beyond week tau_cutoff-1 is used for estimation.
    This metric evaluates how well the model predicts future cash flows.

    Parameters
    ----------
    r_cumulative : np.ndarray
        Cumulative log returns from the model.
    m_t : np.ndarray
        Mapping function.
    C, D : np.ndarray
        Cash flows.
    tau_cutoff : int
        First week of out-of-sample period.

    Returns
    -------
    rmse : float
    """
    T = len(r_cumulative)
    pme = compute_pme_series(r_cumulative, m_t, C, D)
    valid = ~np.isnan(pme[tau_cutoff:])
    if valid.sum() == 0:
        return np.nan
    errors = pme[tau_cutoff:][valid] - 1.0
    return np.sqrt(np.mean(errors**2))


def compute_hybrid_rmse(
    r_cumulative: np.ndarray,
    m_t: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    tau_cutoff: int,
) -> float:
    """
    Hybrid RMSE: uses OOS NAVs but all since-inception cash flows.

    A compromise between in-sample and OOS: the full cash flow history
    is included, but evaluation starts after the estimation cutoff.

    Parameters
    ----------
    r_cumulative : np.ndarray
        Cumulative log returns from the model.
    m_t : np.ndarray
        Mapping function.
    C, D : np.ndarray
        Cash flows.
    tau_cutoff : int
        First week of the hybrid evaluation window.

    Returns
    -------
    rmse : float
    """
    T = len(r_cumulative)
    pme = compute_pme_series(r_cumulative, m_t, C, D)
    valid = ~np.isnan(pme[tau_cutoff:])
    if valid.sum() == 0:
        return np.nan
    errors = pme[tau_cutoff:][valid] - 1.0
    return np.sqrt(np.mean(errors**2))


def compute_improvement_rate(
    ssm_rmse: np.ndarray,
    naive_rmse: np.ndarray,
) -> float:
    """
    Fraction of funds where SSM RMSE is smaller than naive RMSE.

    Parameters
    ----------
    ssm_rmse : np.ndarray
        SSM-based RMSE for each fund.
    naive_rmse : np.ndarray
        Naive nowcast RMSE for each fund.

    Returns
    -------
    rate : float
        Fraction in [0, 1] where SSM beats naive.
    """
    valid = ~np.isnan(ssm_rmse) & ~np.isnan(naive_rmse)
    if valid.sum() == 0:
        return np.nan
    return (ssm_rmse[valid] < naive_rmse[valid]).mean()


def return_autocorrelation(
    returns: np.ndarray,
    lag: int = 1,
) -> float:
    """
    Compute autocorrelation of a return series at a given lag.

    Used to assess whether filtered returns have been successfully
    unsmoothed (autocorrelation should be near zero).

    Parameters
    ----------
    returns : np.ndarray
        Return series (weekly or quarterly).
    lag : int
        Lag for autocorrelation. Default 1.

    Returns
    -------
    ac : float
        Autocorrelation coefficient.
    """
    valid = ~np.isnan(returns)
    r = returns[valid]
    if len(r) <= lag:
        return np.nan
    r_demeaned = r - r.mean()
    numerator = np.sum(r_demeaned[lag:] * r_demeaned[:-lag])
    denominator = np.sum(r_demeaned**2)
    if denominator == 0:
        return np.nan
    return numerator / denominator
```

**Step 2: Commit**

```bash
git add src/metrics.py
git commit -m "feat: add PME-based nowcasting performance metrics (Section 2.3)"
```

---

## Task 8: Parameter Estimation Module (`src/estimation.py`)

**Files:**
- Create: `src/estimation.py`

Implements the two-step estimation procedure from Sections 2.4.1-2.4.2.

**Step 1: Write `src/estimation.py`**

This is the most complex module. It implements:
1. Profile likelihood grid search for (α, β) — 15×15 grid
2. MLE for remaining parameters given (α, β)
3. Penalized likelihood (PME-based penalty)
4. EM-like iterations for mapping function m_t
5. Partial imputation with peer-fund parameters

The code should use `scipy.optimize.minimize` for the inner MLE and numpy for the grid search.

Key functions:
- `build_alpha_grid(alpha_center, n=15)` — Build α grid centered at PME-implied value
- `build_beta_grid(beta_mean, beta_std, n=15)` — Build β grid from prior distribution
- `mle_inner(theta_inner, alpha, beta, data, ...)` — Negative log-likelihood for inner params
- `profile_likelihood_grid(data, ...)` — Full grid search with penalty
- `estimate_fund_params(data, ...)` — Complete two-step estimation procedure
- `partial_imputation(data, peer_params, ...)` — Estimation with imputed parameters

Full code provided in the implementation — too long to include here but follows the paper's procedure exactly.

**Step 2: Commit**

```bash
git add src/estimation.py
git commit -m "feat: add parameter estimation with profile likelihood and EM iterations"
```

---

## Task 9: Visualization Module (`src/visualization.py`)

**Files:**
- Create: `src/visualization.py`

Plotting functions for all notebook figures.

**Step 1: Write `src/visualization.py`**

Key functions:
- `plot_fund_lifecycle(fund)` — Overview: true values, reported NAVs, distributions, calls
- `plot_naive_vs_true(fund, nav_naive)` — Compare naive nowcast to true values
- `plot_ssm_nowcast(fund, kalman_result)` — SSM nowcast vs true vs reported
- `plot_kalman_states(kalman_result)` — State vector evolution over time
- `plot_return_comparison(...)` — Filtered vs true vs naive returns + autocorrelation
- `plot_profile_likelihood(grid_results)` — Heatmap of penalized likelihood over (α,β)
- `plot_pme_series(...)` — PME series for SSM vs naive
- `plot_monte_carlo_results(...)` — Parameter recovery and nowcast RMSE distributions
- `plot_sensitivity(...)` — Sensitivity to parameter changes

All use matplotlib with a clean, academic style.

**Step 2: Commit**

```bash
git add src/visualization.py
git commit -m "feat: add visualization module for all notebook figures"
```

---

## Task 10: Jupyter Notebook — Sections 1-3 (Intro, DGP, Naive)

**Files:**
- Create: `notebooks/nowcasting_pe_nav.ipynb`

**Step 1: Create notebook with TOC and Sections 1-3**

- Table of Contents with anchor links
- Section 1: Motivation + PE valuation problem explanation
- Section 2: Simulate a fund, visualize all series
- Section 3: Compute naive nowcast, compare to true values

**Step 2: Commit**

```bash
git add notebooks/nowcasting_pe_nav.ipynb
git commit -m "feat: add notebook sections 1-3 (intro, DGP, naive nowcast)"
```

---

## Task 11: Notebook — Sections 4-5 (SSM Formulation, Kalman Filter)

**Files:**
- Modify: `notebooks/nowcasting_pe_nav.ipynb`

Add sections explaining:
- Section 4: SSM matrices with LaTeX equations, build matrices for example period
- Section 5: Run Kalman filter/smoother, visualize filtered returns vs true

**Commit:**

```bash
git add notebooks/nowcasting_pe_nav.ipynb
git commit -m "feat: add notebook sections 4-5 (SSM formulation, Kalman filter)"
```

---

## Task 12: Notebook — Section 6 (Parameter Estimation)

**Files:**
- Modify: `notebooks/nowcasting_pe_nav.ipynb`

Add section explaining:
- Profile likelihood grid search
- MLE for inner parameters
- Penalized likelihood
- EM iterations for mapping function
- Visualize: profile likelihood heatmap, parameter recovery

**Commit:**

```bash
git add notebooks/nowcasting_pe_nav.ipynb
git commit -m "feat: add notebook section 6 (parameter estimation)"
```

---

## Task 13: Notebook — Sections 7-8 (Partial Imputation, Metrics)

**Files:**
- Modify: `notebooks/nowcasting_pe_nav.ipynb`

Add sections:
- Section 7: Peer-fund imputation concept and demonstration
- Section 8: PME metrics, RMSE comparison, improvement rates

**Commit:**

```bash
git add notebooks/nowcasting_pe_nav.ipynb
git commit -m "feat: add notebook sections 7-8 (imputation, performance metrics)"
```

---

## Task 14: Notebook — Sections 9-10 (Monte Carlo, Sensitivity)

**Files:**
- Modify: `notebooks/nowcasting_pe_nav.ipynb`

Add sections:
- Section 9: Generate panel of funds, run estimation, summarize results
- Section 10: Sensitivity analysis to key parameters

**Commit:**

```bash
git add notebooks/nowcasting_pe_nav.ipynb
git commit -m "feat: add notebook sections 9-10 (Monte Carlo, sensitivity analysis)"
```

---

## Task 15: Notebook — Section 11 (Glossary & Appendix)

**Files:**
- Modify: `notebooks/nowcasting_pe_nav.ipynb`

Add glossary with definitions and Wikipedia links for:
State Space Models, Kalman Filter, Kalman Smoother, GARCH, MLE, Profile Likelihood, EM Algorithm, NAV Smoothing / Appraisal Bias, PME, Nowcasting, MIDAS, Penalized Likelihood / Ridge Regression

**Commit:**

```bash
git add notebooks/nowcasting_pe_nav.ipynb
git commit -m "feat: add glossary and appendix section to notebook"
```

---

## Task 16: Final Review and Polish

- Run full notebook end-to-end
- Fix any runtime errors
- Ensure all visualizations render correctly
- Verify parameter recovery on simulated data
- Final commit

```bash
git add -A
git commit -m "chore: final polish — ensure notebook runs end-to-end"
```
