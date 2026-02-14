"""
Parameter estimation for the PE fund SSM.

Implements the two-step estimation procedure from Sections 2.4.1-2.4.2:

1. Profile likelihood: 15×15 grid search over (α, β). For each grid point,
   MLE is used to estimate the remaining 8 parameters.
2. Iterative refinement: Fix (α, β) at the grid optimum, iteratively
   re-estimate remaining params and update the mapping function m_t.

The estimation uses penalized log-likelihood with a PME-based penalty
that encourages the model to produce reasonable return magnitudes.

Reference: Section 2.4, Appendix A.4
"""
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass

from .kalman import kalman_filter, KalmanResult
from .simulation import SimulatedFund


@dataclass
class EstimationResult:
    """Container for parameter estimation results."""

    alpha: float = 0.0
    beta: float = 1.0
    beta_c: float = 0.85
    psi: float = 0.001
    F: float = 2.0
    Fc: float = 1.0
    lam: float = 0.9
    delta: float = 0.03
    sigma_n: float = 0.05
    sigma_d: float = 0.10

    log_likelihood: float = -np.inf
    penalized_ll: float = -np.inf
    n_iterations: int = 0
    converged: bool = False

    # Profile likelihood grid results
    alpha_grid: np.ndarray = None
    beta_grid: np.ndarray = None
    ll_grid: np.ndarray = None

    # Mapping function estimate
    m_hat: np.ndarray = None

    # Kalman filter result at optimum
    kalman_result: KalmanResult = None


def build_alpha_grid(alpha_center: float = 0.0, n: int = 15) -> np.ndarray:
    """
    Build α grid centered at a given value.

    The grid spans from alpha_center - 0.01 to alpha_center + 0.01
    (approximately ±5% per annum in weekly terms).

    Parameters
    ----------
    alpha_center : float
        Center of the grid. Often set to the PME-implied α.
    n : int
        Number of grid points.

    Returns
    -------
    alpha_grid : np.ndarray, shape (n,)
    """
    half_width = 0.01  # ±1% per week ≈ ±52% per year
    return np.linspace(alpha_center - half_width, alpha_center + half_width, n)


def build_beta_grid(beta_mean: float = 1.1, beta_std: float = 0.3, n: int = 15) -> np.ndarray:
    """
    Build β grid centered at the prior mean.

    The grid spans beta_mean ± 2*beta_std, reflecting the cross-sectional
    distribution of PE fund betas.

    Parameters
    ----------
    beta_mean : float
        Prior mean for β. Typically 1.0-1.2 for buyout, 1.2-1.4 for venture.
    beta_std : float
        Prior standard deviation.
    n : int
        Number of grid points.

    Returns
    -------
    beta_grid : np.ndarray, shape (n,)
    """
    return np.linspace(
        max(0.1, beta_mean - 2 * beta_std),
        beta_mean + 2 * beta_std,
        n,
    )


def _pack_inner_params(F, Fc, lam, delta, sigma_n, sigma_d) -> np.ndarray:
    """Pack inner parameters into a vector for optimization."""
    return np.array([
        np.log(F),           # log-transform for positivity
        np.log(Fc),
        np.log(lam / (1 - lam)),  # logit for (0,1) constraint
        np.log(delta),
        np.log(sigma_n),
        np.log(sigma_d),
    ])


def _unpack_inner_params(theta: np.ndarray) -> tuple:
    """Unpack inner parameter vector back to named values."""
    # Clamp raw values before exp to prevent overflow
    theta_clamped = np.clip(theta, -10, 10)
    F = np.exp(theta_clamped[0])
    Fc = np.exp(theta_clamped[1])
    lam = 1 / (1 + np.exp(-theta_clamped[2]))  # sigmoid
    delta = np.exp(theta_clamped[3])
    sigma_n = np.exp(theta_clamped[4])
    sigma_d = np.exp(theta_clamped[5])

    # Clamp to reasonable ranges
    F = np.clip(F, 0.1, 10.0)
    Fc = np.clip(Fc, 0.1, 10.0)
    lam = np.clip(lam, 0.01, 0.999)
    delta = np.clip(delta, 0.001, 0.5)
    sigma_n = np.clip(sigma_n, 0.001, 1.0)
    sigma_d = np.clip(sigma_d, 0.001, 2.0)

    return F, Fc, lam, delta, sigma_n, sigma_d


def _negative_penalized_ll(
    theta_inner: np.ndarray,
    alpha: float,
    beta: float,
    beta_c: float,
    psi: float,
    fund: SimulatedFund,
    m_hat: np.ndarray,
    penalty_weight: float = 1.0,
) -> float:
    """
    Negative penalized log-likelihood for inner parameters.

    The penalty term encourages realistic return magnitudes via PME proximity to 1.

    Parameters
    ----------
    theta_inner : np.ndarray
        Packed inner parameters (6 values).
    alpha, beta : float
        Fixed outer parameters from profile grid.
    beta_c, psi : float
        Comparable asset parameters (estimated separately or fixed).
    fund : SimulatedFund
        Simulated fund data.
    m_hat : np.ndarray
        Current mapping function estimate.
    penalty_weight : float
        Weight on the PME penalty term.

    Returns
    -------
    neg_ll : float
        Negative penalized log-likelihood.
    """
    F, Fc, lam, delta, sigma_n, sigma_d = _unpack_inner_params(theta_inner)

    try:
        result = kalman_filter(
            T=fund.T,
            alpha=alpha, beta=beta, beta_c=beta_c, psi=psi,
            F=F, Fc=Fc, lam=lam, delta=delta,
            sigma_n=sigma_n, sigma_d=sigma_d,
            Rm=fund.rm, rc_log=fund.rc_log, h_t=fund.h_t,
            D=fund.D, C=fund.C, NAV_reported=fund.NAV_reported,
            is_dist_week=fund.is_dist_week, is_nav_week=fund.is_nav_week,
            lambda_t=fund.lambda_t, delta_t=fund.delta_t,
            m_hat=m_hat,
        )

        ll = result.log_likelihood

        # Penalize if log-likelihood is extremely negative (numerical issues)
        if np.isnan(ll) or np.isinf(ll):
            return 1e10

        # PME-based penalty: encourage the implied returns to produce PME ≈ 1
        # Simple penalty on parameter magnitudes as a regularizer
        penalty = penalty_weight * (
            0.1 * (np.log(F))**2
            + 0.1 * (np.log(Fc))**2
            + 0.5 * (lam - 0.9)**2
            + 0.1 * (sigma_n - 0.05)**2
        )

        return -(ll - penalty)

    except Exception:
        return 1e10


def estimate_inner_params(
    alpha: float,
    beta: float,
    fund: SimulatedFund,
    m_hat: np.ndarray,
    beta_c: float = 0.85,
    psi: float = 0.001,
    F0: float = 2.0,
    Fc0: float = 1.0,
    lam0: float = 0.9,
    delta0: float = 0.03,
    sigma_n0: float = 0.05,
    sigma_d0: float = 0.10,
    penalty_weight: float = 1.0,
) -> tuple[dict, float, KalmanResult]:
    """
    MLE for inner parameters given fixed (α, β).

    Parameters
    ----------
    alpha, beta : float
        Fixed outer parameters.
    fund : SimulatedFund
        Fund data.
    m_hat : np.ndarray
        Current mapping function estimate.
    beta_c, psi : float
        Comparable asset parameters.
    F0, Fc0, lam0, delta0, sigma_n0, sigma_d0 : float
        Initial guesses for inner parameters.
    penalty_weight : float
        Penalty term weight.

    Returns
    -------
    params : dict
        Estimated inner parameters.
    penalized_ll : float
        Penalized log-likelihood at optimum.
    kalman_result : KalmanResult
        Kalman filter result at estimated parameters.
    """
    theta0 = _pack_inner_params(F0, Fc0, lam0, delta0, sigma_n0, sigma_d0)

    res = minimize(
        _negative_penalized_ll,
        theta0,
        args=(alpha, beta, beta_c, psi, fund, m_hat, penalty_weight),
        method='Nelder-Mead',
        options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4},
    )

    F, Fc, lam, delta, sigma_n, sigma_d = _unpack_inner_params(res.x)

    # Run Kalman filter at estimated parameters
    kf_result = kalman_filter(
        T=fund.T,
        alpha=alpha, beta=beta, beta_c=beta_c, psi=psi,
        F=F, Fc=Fc, lam=lam, delta=delta,
        sigma_n=sigma_n, sigma_d=sigma_d,
        Rm=fund.rm, rc_log=fund.rc_log, h_t=fund.h_t,
        D=fund.D, C=fund.C, NAV_reported=fund.NAV_reported,
        is_dist_week=fund.is_dist_week, is_nav_week=fund.is_nav_week,
        lambda_t=fund.lambda_t, delta_t=fund.delta_t,
        m_hat=m_hat,
    )

    params = {
        'F': F, 'Fc': Fc, 'lam': lam,
        'delta': delta, 'sigma_n': sigma_n, 'sigma_d': sigma_d,
    }

    return params, -res.fun, kf_result


def update_mapping_function(
    fund: SimulatedFund,
    kalman_result: KalmanResult,
) -> np.ndarray:
    """
    Update the mapping function m̂_t using Kalman-filtered state estimates.

    The mapping function converts between cumulative returns and asset values.
    After running the Kalman filter, we can update m̂_t using the filtered
    cumulative returns and observed cash flows.

    Parameters
    ----------
    fund : SimulatedFund
        Fund data with cash flows.
    kalman_result : KalmanResult
        Current Kalman filter output.

    Returns
    -------
    m_hat : np.ndarray, shape (T,)
        Updated mapping function.
    """
    T = fund.T
    m_hat = np.zeros(T)

    # Use filtered cumulative returns to update mapping
    r_cum_filtered = kalman_result.s_filtered[:, 1] + kalman_result.s_filtered[:, 0]

    for t in range(1, T):
        # V̂_t = exp(r̂_0:t - m̂_t)
        # m̂_t = m̂_{t-1} + log((V̂_t + D_t - C_t) / V̂_t)
        V_hat = np.exp(r_cum_filtered[t] - m_hat[t - 1])
        if V_hat > 0:
            numerator = V_hat + fund.D[t] - fund.C[t]
            if numerator > 0:
                m_hat[t] = m_hat[t - 1] + np.log(numerator / V_hat)
            else:
                m_hat[t] = m_hat[t - 1]
        else:
            m_hat[t] = m_hat[t - 1]

    return m_hat


def profile_likelihood_grid(
    fund: SimulatedFund,
    alpha_grid: np.ndarray = None,
    beta_grid: np.ndarray = None,
    m_hat_init: np.ndarray = None,
    n_alpha: int = 15,
    n_beta: int = 15,
    penalty_weight: float = 1.0,
    verbose: bool = False,
) -> EstimationResult:
    """
    Profile likelihood grid search over (α, β).

    For each (α, β) on the grid, estimate the remaining parameters via MLE.
    Select the (α, β) with the highest penalized log-likelihood.

    Parameters
    ----------
    fund : SimulatedFund
        Fund data.
    alpha_grid : np.ndarray, optional
        Grid of α values. If None, builds default grid.
    beta_grid : np.ndarray, optional
        Grid of β values. If None, builds default grid.
    m_hat_init : np.ndarray, optional
        Initial mapping function. If None, uses fund.m_t.
    n_alpha, n_beta : int
        Grid dimensions (used only if grids not provided).
    penalty_weight : float
        Weight on penalty term.
    verbose : bool
        Print progress.

    Returns
    -------
    result : EstimationResult
        Best parameters and grid results.
    """
    if alpha_grid is None:
        alpha_grid = build_alpha_grid(0.0, n_alpha)
    if beta_grid is None:
        beta_grid = build_beta_grid(1.1, 0.3, n_beta)
    if m_hat_init is None:
        m_hat_init = fund.m_t.copy()

    ll_grid = np.full((len(alpha_grid), len(beta_grid)), -np.inf)
    best_ll = -np.inf
    best_result = None

    for i, alpha in enumerate(alpha_grid):
        for j, beta in enumerate(beta_grid):
            try:
                params, pll, kf_result = estimate_inner_params(
                    alpha=alpha, beta=beta,
                    fund=fund, m_hat=m_hat_init,
                    penalty_weight=penalty_weight,
                )
                ll_grid[i, j] = pll

                if pll > best_ll:
                    best_ll = pll
                    best_result = EstimationResult(
                        alpha=alpha, beta=beta,
                        beta_c=0.85, psi=0.001,
                        F=params['F'], Fc=params['Fc'],
                        lam=params['lam'], delta=params['delta'],
                        sigma_n=params['sigma_n'], sigma_d=params['sigma_d'],
                        log_likelihood=kf_result.log_likelihood,
                        penalized_ll=pll,
                        kalman_result=kf_result,
                        m_hat=m_hat_init.copy(),
                    )

                if verbose:
                    print(f"  α={alpha:.4f}, β={beta:.3f}: pen-LL={pll:.1f}")

            except Exception as e:
                if verbose:
                    print(f"  α={alpha:.4f}, β={beta:.3f}: FAILED ({e})")
                ll_grid[i, j] = -np.inf

    if best_result is not None:
        best_result.alpha_grid = alpha_grid
        best_result.beta_grid = beta_grid
        best_result.ll_grid = ll_grid

    return best_result


def estimate_fund_params(
    fund: SimulatedFund,
    n_alpha: int = 15,
    n_beta: int = 15,
    n_em_iterations: int = 5,
    penalty_weight: float = 1.0,
    verbose: bool = False,
) -> EstimationResult:
    """
    Complete two-step estimation procedure.

    Step 1: Profile likelihood grid search for (α, β).
    Step 2: Iterative refinement with mapping function updates (EM-like).

    Parameters
    ----------
    fund : SimulatedFund
        Fund data.
    n_alpha, n_beta : int
        Profile grid dimensions.
    n_em_iterations : int
        Number of EM-like iterations for mapping function.
    penalty_weight : float
        Penalty weight.
    verbose : bool
        Print progress.

    Returns
    -------
    result : EstimationResult
        Final estimated parameters.
    """
    if verbose:
        print("Step 1: Profile likelihood grid search...")

    # Step 1: Initial grid search
    result = profile_likelihood_grid(
        fund=fund,
        n_alpha=n_alpha, n_beta=n_beta,
        penalty_weight=penalty_weight,
        verbose=verbose,
    )

    if result is None:
        raise RuntimeError("Profile likelihood grid search failed completely.")

    if verbose:
        print(f"\nStep 1 result: α={result.alpha:.4f}, β={result.beta:.3f}")
        print(f"  F={result.F:.3f}, λ={result.lam:.3f}, σn={result.sigma_n:.4f}")

    # Step 2: EM-like iterations
    m_hat = fund.m_t.copy()

    for iteration in range(n_em_iterations):
        if verbose:
            print(f"\nEM iteration {iteration + 1}/{n_em_iterations}...")

        # E-step: Run Kalman filter with current parameters
        kf_result = kalman_filter(
            T=fund.T,
            alpha=result.alpha, beta=result.beta,
            beta_c=result.beta_c, psi=result.psi,
            F=result.F, Fc=result.Fc, lam=result.lam, delta=result.delta,
            sigma_n=result.sigma_n, sigma_d=result.sigma_d,
            Rm=fund.rm, rc_log=fund.rc_log, h_t=fund.h_t,
            D=fund.D, C=fund.C, NAV_reported=fund.NAV_reported,
            is_dist_week=fund.is_dist_week, is_nav_week=fund.is_nav_week,
            lambda_t=fund.lambda_t, delta_t=fund.delta_t,
            m_hat=m_hat,
        )

        # Update mapping function
        m_hat_new = update_mapping_function(fund, kf_result)

        # Check convergence
        m_change = np.max(np.abs(m_hat_new - m_hat))
        m_hat = m_hat_new

        if verbose:
            print(f"  m_hat max change: {m_change:.6f}")
            print(f"  log-likelihood: {kf_result.log_likelihood:.1f}")

        # M-step: Re-estimate inner parameters with updated m_hat
        params, pll, kf_result = estimate_inner_params(
            alpha=result.alpha, beta=result.beta,
            fund=fund, m_hat=m_hat,
            beta_c=result.beta_c, psi=result.psi,
            F0=result.F, Fc0=result.Fc, lam0=result.lam,
            delta0=result.delta, sigma_n0=result.sigma_n,
            sigma_d0=result.sigma_d,
            penalty_weight=penalty_weight,
        )

        result.F = params['F']
        result.Fc = params['Fc']
        result.lam = params['lam']
        result.delta = params['delta']
        result.sigma_n = params['sigma_n']
        result.sigma_d = params['sigma_d']
        result.log_likelihood = kf_result.log_likelihood
        result.penalized_ll = pll
        result.kalman_result = kf_result
        result.m_hat = m_hat.copy()
        result.n_iterations = iteration + 1

        if m_change < 1e-4:
            result.converged = True
            if verbose:
                print(f"  Converged at iteration {iteration + 1}.")
            break

    return result


def partial_imputation(
    fund: SimulatedFund,
    peer_beta: float,
    peer_F: float = None,
    peer_lam: float = None,
    n_alpha: int = 15,
    penalty_weight: float = 1.0,
    verbose: bool = False,
) -> EstimationResult:
    """
    Estimation with peer-fund parameter imputation (β-anchoring).

    When the full profile grid is too expensive or data is too sparse,
    β can be anchored to a peer-group estimate (e.g., median of similar
    vintage/strategy funds). Only α is profiled on a 1D grid.

    Parameters
    ----------
    fund : SimulatedFund
        Fund data.
    peer_beta : float
        β estimate from peer funds (used as anchor).
    peer_F : float, optional
        Peer F estimate (if available, used as starting point).
    peer_lam : float, optional
        Peer λ estimate.
    n_alpha : int
        Number of α grid points.
    penalty_weight : float
        Penalty weight.
    verbose : bool
        Print progress.

    Returns
    -------
    result : EstimationResult
        Estimated parameters with β anchored.
    """
    alpha_grid = build_alpha_grid(0.0, n_alpha)
    beta_grid = np.array([peer_beta])  # Fix β to peer estimate

    m_hat = fund.m_t.copy()

    best_ll = -np.inf
    best_result = None

    for alpha in alpha_grid:
        try:
            F0 = peer_F if peer_F is not None else 2.0
            lam0 = peer_lam if peer_lam is not None else 0.9

            params, pll, kf_result = estimate_inner_params(
                alpha=alpha, beta=peer_beta,
                fund=fund, m_hat=m_hat,
                F0=F0, lam0=lam0,
                penalty_weight=penalty_weight,
            )

            if pll > best_ll:
                best_ll = pll
                best_result = EstimationResult(
                    alpha=alpha, beta=peer_beta,
                    F=params['F'], Fc=params['Fc'],
                    lam=params['lam'], delta=params['delta'],
                    sigma_n=params['sigma_n'], sigma_d=params['sigma_d'],
                    log_likelihood=kf_result.log_likelihood,
                    penalized_ll=pll,
                    kalman_result=kf_result,
                    m_hat=m_hat.copy(),
                )

        except Exception:
            continue

    if best_result is not None:
        best_result.alpha_grid = alpha_grid
        best_result.beta_grid = beta_grid

    return best_result
