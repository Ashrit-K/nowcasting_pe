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
    """

    T: int
    """Number of time periods."""

    n_states: int = 4
    """Dimension of the state vector."""

    # --- Forward (filter) results ---
    s_filtered: np.ndarray = field(default_factory=lambda: np.array([]))
    """Filtered state estimates E[st | Yt]. Shape: (T, n_states)."""

    P_filtered: np.ndarray = field(default_factory=lambda: np.array([]))
    """Filtered state covariance var(st | Yt). Shape: (T, n_states, n_states)."""

    s_predicted: np.ndarray = field(default_factory=lambda: np.array([]))
    """One-step-ahead predicted states E[st | Y_{t-1}]. Shape: (T, n_states)."""

    P_predicted: np.ndarray = field(default_factory=lambda: np.array([]))
    """One-step-ahead predicted covariance. Shape: (T, n_states, n_states)."""

    innovations: list = field(default_factory=list)
    """Innovation (prediction error) for each period."""

    F_matrices: list = field(default_factory=list)
    """Innovation covariance matrices."""

    log_likelihood: float = 0.0
    """Total log-likelihood: Σ_t log p(yt | Y_{t-1})."""

    # --- Backward (smoother) results ---
    s_smoothed: np.ndarray = field(default_factory=lambda: np.array([]))
    """Smoothed state estimates E[st | YT]. Shape: (T, n_states)."""

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

    Implements equation (A.2).

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

    result = KalmanResult(T=T, n_states=n_states)
    result.s_filtered = np.zeros((T, n_states))
    result.P_filtered = np.zeros((T, n_states, n_states))
    result.s_predicted = np.zeros((T, n_states))
    result.P_predicted = np.zeros((T, n_states, n_states))
    result.innovations = []
    result.F_matrices = []
    result.log_likelihood = 0.0

    # Initialize state and covariance
    s = np.array([0.0, 0.0, 0.0, 1.0])
    P = np.diag([0.01, 0.001, 0.001, 0.0])

    for t in range(T):
        result.s_predicted[t] = s.copy()
        result.P_predicted[t] = P.copy()

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

        Z = ssm.Z
        H = ssm.H
        G = ssm.G
        V_mat = ssm.V
        Q = ssm.Q

        y_resid = y - ssm.Gamma @ ssm.x
        v = y_resid - Z @ s
        F_cov = Z @ P @ Z.T + H

        result.innovations.append(v)
        result.F_matrices.append(F_cov)

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
                F_inv = np.linalg.pinv(F_cov)

        K = G @ P @ Z.T @ F_inv
        s_filtered = s + P @ Z.T @ F_inv @ v
        result.s_filtered[t] = s_filtered
        P_filtered = P - P @ Z.T @ F_inv @ Z @ P
        result.P_filtered[t] = P_filtered

        s = G @ s_filtered
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
    D: np.ndarray,
    NAV_reported: np.ndarray,
    is_dist_week: np.ndarray,
    is_nav_week: np.ndarray,
    lambda_t: np.ndarray,
    delta_t: np.ndarray,
    m_hat: np.ndarray,
) -> KalmanResult:
    """
    Run the Kalman smoother (backward recursion) to refine state estimates.

    Implements equation (A.3).

    Parameters
    ----------
    result : KalmanResult
        Output from kalman_filter.
    alpha, ..., sigma_d : float
        SSM parameters.
    Rm, h_t : np.ndarray
        Market returns and GARCH variance.
    D, NAV_reported : np.ndarray
        Observation data (needed to reconstruct observation masks).
    is_dist_week, is_nav_week : np.ndarray of bool
        Observation availability masks.
    lambda_t, delta_t : np.ndarray
        Time-varying function values.
    m_hat : np.ndarray
        Mapping function estimates.

    Returns
    -------
    result : KalmanResult
        Updated with smoothed states (s_smoothed, P_smoothed).
    """
    T = result.T
    n_states = result.n_states

    result.s_smoothed = np.zeros((T, n_states))
    result.P_smoothed = np.zeros((T, n_states, n_states))

    result.s_smoothed[T - 1] = result.s_filtered[T - 1]
    result.P_smoothed[T - 1] = result.P_filtered[T - 1]

    # Backward recursion
    b = np.zeros(n_states)

    for t in range(T - 2, -1, -1):
        Rm_t = Rm[t + 1]
        alpha_beta = np.log(alpha + beta * (Rm_t - 1) + 1)

        G = np.array([
            [0, 0, 0, alpha_beta],
            [1, 1, 0, 0],
            [0, (1 - lambda_t[t + 1]), lambda_t[t + 1], 0],
            [0, 0, 0, 1],
        ])

        v = result.innovations[t + 1]
        F_cov = result.F_matrices[t + 1]

        # Reconstruct the correct observation mask for period t+1
        _, has_dist_next, has_nav_next = build_observation_vector(
            t + 1, D, NAV_reported, np.zeros(T),  # rc_log not needed for mask
            is_dist_week, is_nav_week,
        )

        ssm_next = build_ssm_matrices(
            t=t + 1,
            alpha=alpha, beta=beta, beta_c=beta_c, psi=psi,
            F=F, Fc=Fc, lam=lam, delta=delta,
            sigma_n=sigma_n, sigma_d=sigma_d,
            Rm_t=Rm[t + 1], h_t=h_t[t + 1],
            lambda_t=lambda_t[t + 1], delta_t=delta_t[t + 1],
            m_hat=m_hat,
            has_distribution=has_dist_next, has_nav=has_nav_next,
        )

        try:
            F_inv = np.linalg.inv(F_cov)
        except np.linalg.LinAlgError:
            F_inv = np.linalg.pinv(F_cov)

        P_pred = result.P_predicted[t + 1]
        Z_next = ssm_next.Z

        K_next = G @ P_pred @ Z_next.T @ F_inv
        b = Z_next.T @ F_inv @ v + (G - K_next @ Z_next).T @ b

        result.s_smoothed[t] = result.s_filtered[t] + result.P_filtered[t] @ b
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

    r_weekly = states[:, 0]
    r_cumulative = states[:, 1] + states[:, 0]

    return r_weekly, r_cumulative
