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
    am = arch_model(residuals * 100, vol='Garch', p=1, q=1, mean='Zero')
    result = am.fit(disp='off')

    # Convert conditional variance back from percentage² to decimal²
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
