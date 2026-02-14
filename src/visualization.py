"""
Visualization module for PE fund nowcasting.

All plotting functions for the pedagogical notebook. Uses matplotlib
with a clean academic style suitable for teaching.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from .simulation import SimulatedFund
from .kalman import KalmanResult

# Global style settings
STYLE = {
    'figure.figsize': (12, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 11,
}


def _apply_style():
    """Apply consistent plotting style."""
    plt.rcParams.update(STYLE)


def plot_fund_lifecycle(fund: SimulatedFund, figsize=(14, 10)):
    """
    Overview plot: true values, reported NAVs, distributions, capital calls.

    Creates a 4-panel figure showing the complete fund lifecycle.
    """
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    weeks = np.arange(fund.T)
    years = weeks / 52

    # Panel 1: True fund value vs reported NAVs
    ax = axes[0, 0]
    ax.plot(years, fund.V_true, 'b-', linewidth=1.5, label='True Value $V_t$')
    nav_weeks = np.where(~np.isnan(fund.NAV_reported))[0]
    ax.scatter(nav_weeks / 52, fund.NAV_reported[nav_weeks],
               c='red', s=30, zorder=5, label='Reported NAV')
    ax.set_xlabel('Year')
    ax.set_ylabel('Value ($M)')
    ax.set_title('Fund Value: True vs Reported NAV')
    ax.legend()

    # Panel 2: Cumulative returns (true vs smoothed)
    ax = axes[0, 1]
    ax.plot(years, fund.r_cum_log, 'b-', linewidth=1.5,
            label='True cumulative $r_{0:t}$')
    ax.plot(years, fund.r_cum_smoothed_log, 'r--', linewidth=1.5,
            label='Smoothed $\\bar{r}_{0:t}$')
    ax.set_xlabel('Year')
    ax.set_ylabel('Log cumulative return')
    ax.set_title('Cumulative Returns: True vs NAV-Smoothed')
    ax.legend()

    # Panel 3: Cash flows
    ax = axes[1, 0]
    dist_weeks = np.where(fund.D > 0)[0]
    call_weeks = np.where(fund.C > 0)[0]
    ax.bar(call_weeks / 52, fund.C[call_weeks], width=0.05,
           color='green', alpha=0.7, label='Capital Calls')
    ax.bar(dist_weeks / 52, -fund.D[dist_weeks], width=0.05,
           color='orange', alpha=0.7, label='Distributions (neg)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Cash Flow ($M)')
    ax.set_title('Fund Cash Flows')
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.5)

    # Panel 4: Weekly returns and GARCH volatility
    ax = axes[1, 1]
    ax.plot(years, fund.r_true_log, 'b-', alpha=0.4, linewidth=0.5,
            label='True weekly $r_t$')
    ax.plot(years, fund.params.F * np.sqrt(fund.h_t), 'r-', linewidth=1.5,
            label='$F \\cdot \\sqrt{h_t}$ (1σ band)')
    ax.plot(years, -fund.params.F * np.sqrt(fund.h_t), 'r-', linewidth=1.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Log return')
    ax.set_title('Weekly Returns and Volatility')
    ax.legend()

    plt.tight_layout()
    return fig, axes


def plot_naive_vs_true(
    fund: SimulatedFund,
    nav_naive: np.ndarray,
    figsize=(12, 5),
):
    """
    Compare naive (Rc-interpolated) nowcast to true values.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    years = np.arange(fund.T) / 52

    # Panel 1: Value comparison
    ax = axes[0]
    ax.plot(years, fund.V_true, 'b-', linewidth=1.5, label='True Value $V_t$')
    ax.plot(years, nav_naive, 'g--', linewidth=1.5, label='Naive Nowcast $\\tilde{V}_t$')
    nav_weeks = np.where(~np.isnan(fund.NAV_reported))[0]
    ax.scatter(nav_weeks / 52, fund.NAV_reported[nav_weeks],
               c='red', s=25, zorder=5, label='Reported NAV')
    ax.set_xlabel('Year')
    ax.set_ylabel('Value ($M)')
    ax.set_title('Naive Nowcast vs True Value')
    ax.legend()

    # Panel 2: Tracking error
    ax = axes[1]
    valid = ~np.isnan(nav_naive)
    tracking_error = np.full(fund.T, np.nan)
    tracking_error[valid] = (nav_naive[valid] - fund.V_true[valid]) / fund.V_true[valid]
    ax.plot(years, tracking_error * 100, 'g-', linewidth=1)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Error (%)')
    ax.set_title('Naive Nowcast Tracking Error')

    plt.tight_layout()
    return fig, axes


def plot_ssm_nowcast(
    fund: SimulatedFund,
    kalman_result: KalmanResult,
    nav_naive: np.ndarray = None,
    figsize=(14, 8),
):
    """
    SSM nowcast vs true vs reported NAVs.
    """
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    years = np.arange(fund.T) / 52

    # Extract filtered/smoothed returns
    r_filtered = kalman_result.s_filtered[:, 0]
    r_cum_filtered = kalman_result.s_filtered[:, 1] + kalman_result.s_filtered[:, 0]

    # Panel 1: Cumulative returns
    ax = axes[0, 0]
    ax.plot(years, fund.r_cum_log, 'b-', linewidth=1.5, alpha=0.7,
            label='True $r_{0:t}$')
    ax.plot(years, r_cum_filtered, 'r-', linewidth=1.5,
            label='Filtered $\\hat{r}_{0:t}$')
    if hasattr(kalman_result, 's_smoothed') and kalman_result.s_smoothed.size > 0:
        r_cum_smoothed = kalman_result.s_smoothed[:, 1] + kalman_result.s_smoothed[:, 0]
        ax.plot(years, r_cum_smoothed, 'g--', linewidth=1.5,
                label='Smoothed $\\hat{r}_{0:t}^{smooth}$')
    ax.set_xlabel('Year')
    ax.set_ylabel('Log cumulative return')
    ax.set_title('Cumulative Returns: True vs SSM Estimates')
    ax.legend(fontsize=9)

    # Panel 2: Weekly returns
    ax = axes[0, 1]
    ax.plot(years, fund.r_true_log, 'b-', alpha=0.3, linewidth=0.5,
            label='True $r_t$')
    ax.plot(years, r_filtered, 'r-', alpha=0.7, linewidth=0.8,
            label='Filtered $\\hat{r}_t$')
    ax.set_xlabel('Year')
    ax.set_ylabel('Log return')
    ax.set_title('Weekly Returns: True vs Filtered')
    ax.legend(fontsize=9)

    # Panel 3: Implied fund value
    ax = axes[1, 0]
    V_filtered = np.exp(r_cum_filtered - fund.m_t)
    ax.plot(years, fund.V_true, 'b-', linewidth=1.5, alpha=0.7,
            label='True $V_t$')
    ax.plot(years, V_filtered, 'r-', linewidth=1.5,
            label='SSM Nowcast')
    if nav_naive is not None:
        ax.plot(years, nav_naive, 'g--', linewidth=1, alpha=0.7,
                label='Naive Nowcast')
    nav_weeks = np.where(~np.isnan(fund.NAV_reported))[0]
    ax.scatter(nav_weeks / 52, fund.NAV_reported[nav_weeks],
               c='orange', s=25, zorder=5, label='Reported NAV')
    ax.set_xlabel('Year')
    ax.set_ylabel('Value ($M)')
    ax.set_title('Fund Value: SSM vs Naive vs True')
    ax.legend(fontsize=9)

    # Panel 4: Kalman gain / state uncertainty
    ax = axes[1, 1]
    P_diag = np.array([kalman_result.P_filtered[t, 0, 0]
                        for t in range(fund.T)])
    ax.plot(years, np.sqrt(P_diag), 'purple', linewidth=1.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Std dev of $r_t$ estimate')
    ax.set_title('Filtered Return Uncertainty')

    plt.tight_layout()
    return fig, axes


def plot_return_comparison(
    true_returns: np.ndarray,
    filtered_returns: np.ndarray,
    naive_returns: np.ndarray = None,
    figsize=(12, 5),
):
    """
    Compare return series: true, filtered, and naive.

    Includes autocorrelation bars for each series.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: Return scatter
    ax = axes[0]
    valid = ~np.isnan(filtered_returns) & ~np.isnan(true_returns)
    ax.scatter(true_returns[valid], filtered_returns[valid],
               alpha=0.3, s=10, c='red', label='Filtered')
    if naive_returns is not None:
        valid_n = ~np.isnan(naive_returns) & ~np.isnan(true_returns)
        ax.scatter(true_returns[valid_n], naive_returns[valid_n],
                   alpha=0.2, s=10, c='green', label='Naive')
    lim = max(abs(true_returns[valid].min()), abs(true_returns[valid].max())) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=0.5)
    ax.set_xlabel('True weekly log return')
    ax.set_ylabel('Estimated weekly log return')
    ax.set_title('Return Estimates vs Truth')
    ax.legend()

    # Panel 2: Autocorrelation comparison
    ax = axes[1]
    from .metrics import return_autocorrelation
    max_lag = 8
    lags = range(1, max_lag + 1)

    ac_true = [return_autocorrelation(true_returns, lag) for lag in lags]
    ac_filtered = [return_autocorrelation(filtered_returns, lag) for lag in lags]

    x = np.arange(max_lag)
    width = 0.25
    ax.bar(x - width, ac_true, width, label='True', alpha=0.7, color='blue')
    ax.bar(x, ac_filtered, width, label='Filtered', alpha=0.7, color='red')

    if naive_returns is not None:
        ac_naive = [return_autocorrelation(naive_returns, lag) for lag in lags]
        ax.bar(x + width, ac_naive, width, label='Naive', alpha=0.7, color='green')

    ax.set_xlabel('Lag (weeks)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Return Autocorrelation by Lag')
    ax.set_xticks(x)
    ax.set_xticklabels(lags)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.legend()

    plt.tight_layout()
    return fig, axes


def plot_profile_likelihood(
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    ll_grid: np.ndarray,
    true_alpha: float = None,
    true_beta: float = None,
    figsize=(8, 6),
):
    """
    Heatmap of penalized log-likelihood over (α, β) grid.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    # Replace -inf with NaN for plotting
    ll_plot = ll_grid.copy()
    ll_plot[np.isinf(ll_plot)] = np.nan

    im = ax.imshow(
        ll_plot.T, origin='lower',
        extent=[alpha_grid[0], alpha_grid[-1], beta_grid[0], beta_grid[-1]],
        aspect='auto', cmap='viridis',
    )
    plt.colorbar(im, ax=ax, label='Penalized Log-Likelihood')

    # Mark the optimum
    best_idx = np.unravel_index(np.nanargmax(ll_plot), ll_plot.shape)
    ax.plot(alpha_grid[best_idx[0]], beta_grid[best_idx[1]],
            'r*', markersize=15, label=f'MLE ({alpha_grid[best_idx[0]]:.4f}, {beta_grid[best_idx[1]]:.2f})')

    if true_alpha is not None and true_beta is not None:
        ax.plot(true_alpha, true_beta, 'wx', markersize=12, markeredgewidth=2,
                label=f'True ({true_alpha:.4f}, {true_beta:.2f})')

    ax.set_xlabel('α (weekly excess return)')
    ax.set_ylabel('β (market exposure)')
    ax.set_title('Profile Likelihood Surface')
    ax.legend()

    plt.tight_layout()
    return fig, ax


def plot_pme_series(
    pme_ssm: np.ndarray,
    pme_naive: np.ndarray = None,
    tau_cutoff: int = None,
    figsize=(10, 5),
):
    """
    PME series for SSM vs naive nowcast.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    T = len(pme_ssm)
    years = np.arange(T) / 52

    ax.plot(years, pme_ssm, 'r-', linewidth=1.5, label='SSM PME')
    if pme_naive is not None:
        ax.plot(years, pme_naive, 'g--', linewidth=1.5, label='Naive PME')

    ax.axhline(1.0, color='black', linewidth=1, linestyle='-', alpha=0.5,
               label='Perfect (PME=1)')

    if tau_cutoff is not None:
        ax.axvline(tau_cutoff / 52, color='gray', linestyle=':', linewidth=1,
                   label='OOS cutoff')

    ax.set_xlabel('Year')
    ax.set_ylabel('PME')
    ax.set_title('Public Market Equivalent Over Time')
    ax.legend()

    plt.tight_layout()
    return fig, ax


def plot_monte_carlo_results(
    param_estimates: dict,
    true_params: dict,
    ssm_rmse_list: np.ndarray,
    naive_rmse_list: np.ndarray,
    figsize=(14, 5),
):
    """
    Monte Carlo results: parameter recovery and nowcast RMSE distributions.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: β recovery
    ax = axes[0]
    betas = param_estimates.get('beta', [])
    if len(betas) > 0:
        ax.hist(betas, bins=20, alpha=0.7, color='steelblue', edgecolor='white')
        ax.axvline(true_params.get('beta', 1.19), color='red', linewidth=2,
                   linestyle='--', label=f"True β = {true_params.get('beta', 1.19):.2f}")
        ax.axvline(np.mean(betas), color='blue', linewidth=2,
                   linestyle='-', label=f"Mean β̂ = {np.mean(betas):.2f}")
    ax.set_xlabel('β')
    ax.set_title('β Recovery')
    ax.legend(fontsize=9)

    # Panel 2: λ recovery
    ax = axes[1]
    lams = param_estimates.get('lam', [])
    if len(lams) > 0:
        ax.hist(lams, bins=20, alpha=0.7, color='steelblue', edgecolor='white')
        ax.axvline(true_params.get('lam', 0.9), color='red', linewidth=2,
                   linestyle='--', label=f"True λ = {true_params.get('lam', 0.9):.2f}")
        ax.axvline(np.mean(lams), color='blue', linewidth=2,
                   linestyle='-', label=f"Mean λ̂ = {np.mean(lams):.2f}")
    ax.set_xlabel('λ')
    ax.set_title('λ Recovery')
    ax.legend(fontsize=9)

    # Panel 3: RMSE comparison
    ax = axes[2]
    valid = ~np.isnan(ssm_rmse_list) & ~np.isnan(naive_rmse_list)
    if valid.sum() > 0:
        ax.hist(ssm_rmse_list[valid], bins=20, alpha=0.6, color='red',
                edgecolor='white', label='SSM')
        ax.hist(naive_rmse_list[valid], bins=20, alpha=0.6, color='green',
                edgecolor='white', label='Naive')
        improvement = (ssm_rmse_list[valid] < naive_rmse_list[valid]).mean()
        ax.set_title(f'RMSE Distribution (SSM wins {improvement:.0%})')
    else:
        ax.set_title('RMSE Distribution')
    ax.set_xlabel('RMSE')
    ax.legend(fontsize=9)

    plt.tight_layout()
    return fig, axes


def plot_sensitivity(
    param_name: str,
    param_values: np.ndarray,
    rmse_values: np.ndarray,
    baseline_value: float = None,
    figsize=(8, 5),
):
    """
    Sensitivity analysis: RMSE as a function of a single parameter.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(param_values, rmse_values, 'bo-', linewidth=1.5, markersize=6)

    if baseline_value is not None:
        ax.axvline(baseline_value, color='red', linestyle='--', linewidth=1,
                   label=f'True {param_name} = {baseline_value}')
        ax.legend()

    ax.set_xlabel(param_name)
    ax.set_ylabel('RMSE')
    ax.set_title(f'Sensitivity to {param_name}')

    plt.tight_layout()
    return fig, ax
