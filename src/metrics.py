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

    where discount_factor(s,t) = exp(r0:t - r0:s).

    If the returns are "correct" (equal to true returns), PME = 1.

    Parameters
    ----------
    r_cumulative : np.ndarray, shape (T,)
        Cumulative log returns r0:t from the model.
    m_t : np.ndarray, shape (T,)
        Mapping function.
    C : np.ndarray, shape (T,)
        Capital calls.
    D : np.ndarray, shape (T,)
        Distributions.
    V_terminal : float
        Terminal value (residual NAV at the end).

    Returns
    -------
    pme : np.ndarray, shape (T,)
        To-date PME series. Values near 1 indicate good model fit.
    """
    T = len(r_cumulative)
    pme = np.full(T, np.nan)

    for t in range(1, T):
        numerator = 0.0
        denominator = 0.0

        for s in range(t + 1):
            df = np.exp(r_cumulative[t] - r_cumulative[s])

            if D[s] > 0:
                numerator += D[s] * df
            if C[s] > 0:
                denominator += C[s] * df

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
