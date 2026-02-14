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
    """Weekly excess return (α). Paper default for simulations: 0.0 arithmetic."""

    beta: float = 1.19
    """Market risk exposure (β). Fund return loading on market excess return."""

    F: float = 2.0
    """Idiosyncratic volatility scale. Multiplier on sqrt(ht) for fund shocks."""

    # --- Comparable asset relationship (eq 2) ---
    beta_c: float = 0.85
    """Slope from reverse regression of comparable asset on fund returns (βc)."""

    psi: float = 0.001
    """Intercept in comparable asset equation (ψ). Weekly drift adjustment."""

    Fc: float = 1.0
    """Comparable asset idiosyncratic volatility scale."""

    # --- NAV smoothing process (eqs 5-6, 8) ---
    lam: float = 0.90
    """Appraisal smoothing parameter (λ)."""

    sigma_n: float = 0.05
    """NAV reporting noise std dev (σn)."""

    # --- Distribution process (eqs 7, 9) ---
    delta: float = 0.03
    """Distribution density trend parameter (δ)."""

    sigma_d: float = 0.10
    """Distribution noise std dev (σd)."""

    # --- Fund configuration (not part of θ, but needed for simulation) ---
    T_weeks: int = 562
    """Total fund life in weeks (~10.8 years). Paper Figure 1 uses 562."""

    fund_size: float = 100.0
    """Initial fund commitment size (in $M)."""

    call_schedule_years: float = 3.0
    """Years over which capital is called."""

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
    """Weekly gross market returns Rmt."""

    rm_log: np.ndarray = field(default_factory=lambda: np.array([]))
    """Weekly log market returns log(Rmt)."""

    rc: np.ndarray = field(default_factory=lambda: np.array([]))
    """Weekly gross comparable asset returns Rct (eq 2)."""

    rc_log: np.ndarray = field(default_factory=lambda: np.array([]))
    """Weekly log comparable asset returns log(Rct)."""

    h_t: np.ndarray = field(default_factory=lambda: np.array([]))
    """GARCH(1,1) conditional variance proxy for time-varying volatility."""

    # --- Latent fund series (TRUE values, not observed in practice) ---
    R_true: np.ndarray = field(default_factory=lambda: np.array([]))
    """True weekly gross fund returns Rt (eq 1). LATENT."""

    r_true_log: np.ndarray = field(default_factory=lambda: np.array([]))
    """True weekly log fund returns rt = log(Rt). LATENT."""

    r_cum_log: np.ndarray = field(default_factory=lambda: np.array([]))
    """True cumulative log returns r0:t = Σ rτ. LATENT."""

    r_cum_smoothed_log: np.ndarray = field(default_factory=lambda: np.array([]))
    """Smoothed cumulative log returns r̄0:t (eq 5). LATENT."""

    V_true: np.ndarray = field(default_factory=lambda: np.array([]))
    """True fund asset values Vt (eq 3). LATENT."""

    m_t: np.ndarray = field(default_factory=lambda: np.array([]))
    """Value-to-return mapping function mt (eq 4, A.8)."""

    eta: np.ndarray = field(default_factory=lambda: np.array([]))
    """Fund idiosyncratic shocks ηt ~ N(0, F²*ht). LATENT."""

    eta_c: np.ndarray = field(default_factory=lambda: np.array([]))
    """Comparable asset idiosyncratic shocks ηct ~ N(0, Fc²*ht). LATENT."""

    # --- Observed fund data ---
    C: np.ndarray = field(default_factory=lambda: np.array([]))
    """Capital calls Ct. Sparse (concentrated in first ~3 years)."""

    D: np.ndarray = field(default_factory=lambda: np.array([]))
    """Distributions Dt (eq 7). Sparse (~25 events over fund life)."""

    NAV_reported: np.ndarray = field(default_factory=lambda: np.array([]))
    """Reported NAVs (eq 6). NaN for non-reporting weeks."""

    nav_noise: np.ndarray = field(default_factory=lambda: np.array([]))
    """NAV reporting noise nt ~ N(0, σn²). LATENT."""

    dist_noise: np.ndarray = field(default_factory=lambda: np.array([]))
    """Distribution noise dt ~ N(0, σd²). LATENT."""

    # --- Derived quantities ---
    w_t: np.ndarray = field(default_factory=lambda: np.array([]))
    """Cash flow weight wt ∈ [0,1] for smoothing function λ(·)t (eq 8)."""

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

    weekly_mean = annual_mean / 52
    weekly_vol = annual_vol / np.sqrt(52)

    rm_log = rng.normal(weekly_mean, weekly_vol, size=T)
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
    """
    C = np.zeros(T)
    call_weeks = int(call_years * 52)

    n_calls = int(call_weeks / 5)
    call_times = rng.choice(range(1, min(call_weeks, T)), size=n_calls, replace=False)
    call_times.sort()

    # Front-load: earlier calls tend to be larger
    weights = np.exp(-0.5 * np.arange(n_calls) / n_calls)
    weights = weights / weights.sum()

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

    Distributions become more frequent as the fund matures (~20-30 events).
    """
    is_dist = np.zeros(T, dtype=bool)
    dist_start_week = int(dist_start_year * 52)

    for t in range(dist_start_week, T):
        years_since_start = (t - dist_start_week) / 52
        total_years = (T - dist_start_week) / 52
        prob = 0.02 + 0.06 * (years_since_start / total_years)

        if t >= 4 and np.any(is_dist[max(0, t - 4):t]):
            continue

        if rng.random() < prob:
            is_dist[t] = True

    return is_dist


def simulate_fund(params: FundParams) -> SimulatedFund:
    """
    Simulate a complete PE fund lifecycle using the paper's DGP.

    Generates all time series needed to demonstrate the nowcasting methodology,
    including both latent (true) and observed series.

    The simulation follows this order:
    1. Generate market returns (exogenous)
    2. Generate GARCH variance proxy h_t
    3. Generate true fund returns (eq 1)
    4. Generate capital calls and distribution events
    5. Compute true asset values (eq 3) and distributions (eq 7)
    6. Compute cumulative returns and mapping function (eq 4, A.8)
    7. Compute smoothed NAVs (eqs 5-6)
    8. Generate comparable asset returns (eq 2)

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
    # =========================================================================
    weekly_idio_vol = 0.16 / np.sqrt(52)
    omega = weekly_idio_vol**2 * 0.05
    alpha_g = 0.08
    beta_g = 0.87
    h_t = np.zeros(T)
    h_t[0] = weekly_idio_vol**2
    market_mean = rm_log.mean()
    for t in range(1, T):
        eps2 = (rm_log[t - 1] - market_mean) ** 2
        h_t[t] = omega + alpha_g * eps2 + beta_g * h_t[t - 1]
    h_t = np.maximum(h_t, 1e-8)

    # =========================================================================
    # Step 3: True fund returns (eq 1)
    #   Rt = (α + β(Rmt - 1) + 1) * exp(ηt)
    #   ηt ~ N(0, F² * ht)
    # =========================================================================
    eta = rng.normal(0, 1, size=T) * params.F * np.sqrt(h_t)
    R_systematic = params.alpha + params.beta * (Rm - 1) + 1
    R_true = R_systematic * np.exp(eta)
    r_true_log = np.log(R_true)

    # =========================================================================
    # Step 4: Capital calls and distribution events
    # =========================================================================
    C = _generate_capital_calls(T, params.fund_size, params.call_schedule_years, rng)
    is_dist = _generate_distribution_events(T, params.dist_start_year, rng)

    # =========================================================================
    # Step 5: True asset values (eq 3) and distributions (eq 7)
    # =========================================================================
    V_true = np.zeros(T)
    D = np.zeros(T)
    delta_t = np.zeros(T)
    dist_noise = np.zeros(T)

    V_true[0] = C[0] if C[0] > 0 else params.fund_size * 0.1

    for t in range(1, T):
        V_pre = V_true[t - 1] * R_true[t]
        t_years = t / 52.0
        delta_t[t] = min(0.99, params.delta * t_years)

        if is_dist[t] and V_pre > 0.01:
            dt_noise = rng.normal(0, params.sigma_d)
            dist_noise[t] = dt_noise
            x = delta_t[t] * np.exp(dt_noise)
            x = min(x, 0.95)
            D[t] = max(0, x * V_pre)

        V_true[t] = V_pre - D[t] + C[t]
        V_true[t] = max(V_true[t], 0.001)

    # =========================================================================
    # Step 6: Cumulative returns and mapping function
    # =========================================================================
    r_cum_log = np.cumsum(r_true_log)

    m_t = np.zeros(T)
    for t in range(1, T):
        if V_true[t] > 0:
            numerator = V_true[t] + D[t] - C[t]
            if numerator > 0:
                m_t[t] = m_t[t - 1] + np.log(numerator / V_true[t])
            else:
                m_t[t] = m_t[t - 1]
        else:
            m_t[t] = m_t[t - 1]

    # =========================================================================
    # Step 7: NAV smoothing (eqs 5-6, 8)
    # =========================================================================
    naive_nav = V_true.copy()

    w_t = np.zeros(T)
    for t in range(T):
        total_cf = abs(D[t]) + abs(C[t])
        if naive_nav[t] > 0 and total_cf > 0:
            w_t[t] = min(1.0, total_cf / (naive_nav[t] + total_cf))

    lambda_t = params.lam * (1 - w_t)

    r_cum_smoothed = np.zeros(T)
    r_cum_smoothed[0] = r_cum_log[0]
    for t in range(1, T):
        r_cum_smoothed[t] = (
            (1 - lambda_t[t]) * r_cum_log[t]
            + lambda_t[t] * r_cum_smoothed[t - 1]
        )

    is_nav_week = np.zeros(T, dtype=bool)
    for t in range(12, T, 13):
        is_nav_week[t] = True

    nav_noise = rng.normal(0, params.sigma_n, size=T)
    NAV_reported = np.full(T, np.nan)
    for t in range(T):
        if is_nav_week[t]:
            NAV_reported[t] = np.exp(r_cum_smoothed[t] - m_t[t] + nav_noise[t])

    # =========================================================================
    # Step 8: Comparable asset returns (eq 2)
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
