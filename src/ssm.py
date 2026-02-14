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
    """

    Z: np.ndarray
    """Observation loading matrix. Maps state to observations."""

    G: np.ndarray
    """State transition matrix. Size: (4, 4)."""

    H: np.ndarray
    """Observation noise covariance. Diagonal."""

    V: np.ndarray
    """State noise loading. Size: (4, 1)."""

    Q: np.ndarray
    """State noise variance. Size: (1, 1)."""

    Gamma: np.ndarray
    """Regressor loading matrix."""

    x: np.ndarray
    """Regressor vector."""

    obs_mask: np.ndarray
    """Boolean mask indicating which observations are present."""

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
    alpha, beta, beta_c, psi, F, Fc, lam, delta, sigma_n, sigma_d : float
        SSM parameters (θ).
    Rm_t : float
        Gross market return for week t.
    h_t : float
        GARCH conditional variance for week t.
    lambda_t : float
        Time-varying smoothing λ(·)_t for current period.
    delta_t : float
        Time-varying distribution density δ(·)_t.
    m_hat : np.ndarray, shape (T,)
        Estimated mapping function series.
    has_distribution : bool
        Whether a distribution is observed this period.
    has_nav : bool
        Whether a NAV report is observed this period.

    Returns
    -------
    ssm : SSMMatrices
        Complete set of SSM matrices for this time step.
    """
    # αβ(·)_t = log(α + β(Rm_t - 1) + 1)
    alpha_beta_t = np.log(alpha + beta * (Rm_t - 1) + 1)

    # Transition matrix G (4×4) — eq A.5
    G = np.array([
        [0, 0, 0, alpha_beta_t],
        [1, 1, 0, 0],
        [0, (1 - lambda_t), lambda_t, 0],
        [0, 0, 0, 1],
    ])

    # State noise loading V (4×1)
    V_mat = np.array([[F], [0], [0], [0]])

    # State noise variance Q (1×1)
    Q = np.array([[h_t]])

    # Observation matrices
    if delta_t > 0 and delta_t < 1:
        log_delta_ratio = np.log(delta_t / (1 - delta_t))
    else:
        log_delta_ratio = 0.0

    Z_full = np.array([
        [0, 1, 0, log_delta_ratio],
        [0, 0, 1, 0],
        [beta_c, 0, 0, psi],
    ])

    H_full = np.diag([
        sigma_d**2,
        sigma_n**2,
        (Fc * np.sqrt(h_t))**2,
    ])

    Gamma_full = np.eye(3)

    if t >= 2 and len(m_hat) > t - 1:
        x_full = np.array([m_hat[t - 1], m_hat[t - 2], 0.0])
    elif t >= 1 and len(m_hat) > t - 1:
        x_full = np.array([m_hat[t - 1], 0.0, 0.0])
    else:
        x_full = np.array([0.0, 0.0, 0.0])

    # Handle missing observations
    obs_mask = np.array([has_distribution, has_nav, True])
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
    dt-1 and navt-2 are often missing.

    Note the time subscripts: distributions enter with lag 1 (dt-1)
    and NAVs with lag 2 (navt-2).

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

    if t >= 1 and is_dist_week[t - 1] and D[t - 1] > 0:
        observations.append(np.log(D[t - 1]))
        has_dist = True

    if t >= 2 and is_nav_week[t - 2] and not np.isnan(NAV_reported[t - 2]):
        observations.append(np.log(NAV_reported[t - 2]))
        has_nav = True

    observations.append(rc_log[t])

    y = np.array(observations)
    return y, has_dist, has_nav
