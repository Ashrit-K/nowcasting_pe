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

    nav_weeks = np.where(is_nav_week & ~np.isnan(NAV_reported))[0]

    if len(nav_weeks) == 0:
        return nav_naive

    first_nav_week = nav_weeks[0]
    nav_naive[first_nav_week] = NAV_reported[first_nav_week]

    # Forward-fill before first NAV using comparable returns (backward)
    for t in range(first_nav_week - 1, -1, -1):
        nav_naive[t] = nav_naive[t + 1] / rc[t + 1] + (D[t + 1] - C[t + 1])
        nav_naive[t] = max(nav_naive[t], 0.001)

    # Interpolate between each pair of consecutive NAV reports
    for i in range(len(nav_weeks) - 1):
        q_minus = nav_weeks[i]
        q = nav_weeks[i + 1]

        nav_start = NAV_reported[q_minus]
        nav_end = NAV_reported[q]

        # Step 1: Compute the "no-drift" interpolation
        nav_no_drift = np.zeros(q - q_minus + 1)
        nav_no_drift[0] = nav_start

        for j in range(1, q - q_minus + 1):
            t = q_minus + j
            nav_no_drift[j] = nav_no_drift[j - 1] * rc[t] + (C[t] - D[t])
            nav_no_drift[j] = max(nav_no_drift[j], 0.001)

        # Step 2: Find the quarter-specific drift ψ_q
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
