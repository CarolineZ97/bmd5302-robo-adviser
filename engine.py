"""
engine.py — Pure-function Markowitz engine for the Robo-Adviser.

This module re-implements (in Python) the two computational cores of the BMD5302
project:

* Part 1  - Efficient Frontier (analytical matrix algebra and SLSQP-constrained).
* Part 2  - Utility maximization  U = r - A * sigma^2 / 2  with no-short-sales.

All functions are pure: given identical inputs they produce identical outputs and
have no UI / filesystem side effects.  This is important for:

* Reproducibility (test_engine.py cross-checks against Excel Solver).
* Migration (the same functions can power a Flask/FastAPI backend later).
* Caching (Streamlit's @st.cache_data can hash arguments cleanly).

Conventions
-----------
* mu        - 1-D ndarray of shape (N,), **annualized** expected returns.
* sigma     - 2-D ndarray of shape (N, N), **annualized** variance-covariance
              matrix.  Annualized = monthly * 12 (same convention used by the
              Excel workbook's Optimal_Portfolio sheet).
* weights   - 1-D ndarray of shape (N,), sum to 1, >= 0 unless allow_short=True.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# Trading-days-per-year convention.  The workbook annualizes by *12 on monthly
# moments, so we follow the same convention here to stay consistent with Part 2.
ANNUALIZATION_FACTOR = 12

# Numeric tolerances used by SLSQP and by the test suite.
_DEFAULT_FTOL = 1e-10
_DEFAULT_MAXITER = 500


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PortfolioResult:
    """Result of a single optimization call."""

    weights: np.ndarray
    expected_return: float
    variance: float
    std: float
    utility: float
    sharpe: float
    success: bool = True
    message: str = ""

    def as_dict(self) -> dict:
        return {
            "weights": self.weights.tolist(),
            "expected_return": float(self.expected_return),
            "variance": float(self.variance),
            "std": float(self.std),
            "utility": float(self.utility),
            "sharpe": float(self.sharpe),
            "success": bool(self.success),
            "message": self.message,
        }


@dataclass
class FrontierResult:
    """Grid of points along the efficient frontier."""

    target_returns: np.ndarray
    variances: np.ndarray
    stds: np.ndarray
    weights: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    gmvp_return: float = 0.0
    gmvp_std: float = 0.0
    gmvp_weights: np.ndarray = field(default_factory=lambda: np.empty(0))


# ---------------------------------------------------------------------------
# Moment estimation
# ---------------------------------------------------------------------------

def compute_mu_sigma(
    prices: pd.DataFrame,
    annualize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean return vector and variance-covariance matrix from prices.

    Parameters
    ----------
    prices : pd.DataFrame
        Row-per-period, column-per-asset adjusted NAV / price matrix.  The index
        does not need to be a DatetimeIndex - the workbook's integer "Month"
        column works equally well.
    annualize : bool
        If True (default), mu and sigma are scaled by 12 to match the workbook
        convention.  Set False if you want to stay in the native monthly units.

    Returns
    -------
    (mu, sigma) : (ndarray, ndarray)
        Shapes (N,) and (N, N) respectively.
    """
    if prices.shape[0] < 2:
        raise ValueError("Need at least 2 price rows to compute returns.")

    returns = prices.pct_change().dropna(how="any")
    mu_monthly = returns.mean().to_numpy(dtype=float)
    sigma_monthly = returns.cov().to_numpy(dtype=float)

    if annualize:
        return mu_monthly * ANNUALIZATION_FACTOR, sigma_monthly * ANNUALIZATION_FACTOR
    return mu_monthly, sigma_monthly


# ---------------------------------------------------------------------------
# Utility optimization  (Part 2 core)
# ---------------------------------------------------------------------------

def optimize_portfolio(
    mu: np.ndarray,
    sigma: np.ndarray,
    A: float,
    allow_short: bool = False,
    risk_free: float = 0.0,
    w0: np.ndarray | None = None,
) -> PortfolioResult:
    """Maximize U = r - A * sigma^2 / 2  subject to sum(w) = 1, w >= 0.

    Matches the Excel Solver setup on the Optimal_Portfolio sheet exactly:
    * Objective: U = mu.T w - A/2 * w.T sigma w  (maximized => we minimize -U).
    * Constraint (a): weights sum to 1.
    * Constraint (b): each weight >= 0 when allow_short is False.
    * Method: SLSQP (SciPy's sequential quadratic programming, the closest
      free equivalent to Excel's GRG Nonlinear).
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float)
    n = mu.size
    if sigma.shape != (n, n):
        raise ValueError(f"sigma shape {sigma.shape} incompatible with mu length {n}")
    if A <= 0:
        raise ValueError("Risk aversion A must be positive.")

    # Initial guess: equal-weight portfolio.
    w0 = np.full(n, 1.0 / n) if w0 is None else np.asarray(w0, dtype=float).ravel()

    bounds = [(None, None) if allow_short else (0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def neg_utility(w: np.ndarray) -> float:
        r = float(w @ mu)
        var = float(w @ sigma @ w)
        return -(r - A * var / 2.0)

    def neg_utility_grad(w: np.ndarray) -> np.ndarray:
        # dU/dw = mu - A * sigma w     =>   d(-U)/dw = -mu + A * sigma w
        return -mu + A * (sigma @ w)

    res = minimize(
        neg_utility,
        w0,
        jac=neg_utility_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": _DEFAULT_FTOL, "maxiter": _DEFAULT_MAXITER, "disp": False},
    )

    w = res.x
    # Numerical cleanup: clip tiny negatives from SLSQP round-off.
    if not allow_short:
        w = np.where(w < 1e-9, 0.0, w)
    s = w.sum()
    if s > 0:
        w = w / s

    r = float(w @ mu)
    var = float(w @ sigma @ w)
    std = float(np.sqrt(max(var, 0.0)))
    util = r - A * var / 2.0
    sharpe = (r - risk_free) / std if std > 1e-12 else 0.0

    return PortfolioResult(
        weights=w,
        expected_return=r,
        variance=var,
        std=std,
        utility=util,
        sharpe=sharpe,
        success=bool(res.success),
        message=str(res.message),
    )


# ---------------------------------------------------------------------------
# Global Minimum Variance Portfolio  (Part 1)
# ---------------------------------------------------------------------------

def gmvp(
    mu: np.ndarray,
    sigma: np.ndarray,
    allow_short: bool = True,
) -> PortfolioResult:
    """Return the Global Minimum Variance Portfolio.

    * allow_short=True uses the analytical formula  w = Sigma^-1 * 1 / (1' Sigma^-1 1)
      which exactly matches Part 1's Frontier_NoShort_Free sheet.
    * allow_short=False uses SLSQP with sum(w)=1, w>=0 only.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float)
    n = mu.size
    ones = np.ones(n)

    if allow_short:
        inv_sigma_ones = np.linalg.solve(sigma, ones)
        w = inv_sigma_ones / (ones @ inv_sigma_ones)
    else:
        w0 = np.full(n, 1.0 / n)
        bounds = [(0.0, 1.0)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        res = minimize(
            lambda w: float(w @ sigma @ w),
            w0,
            jac=lambda w: 2.0 * (sigma @ w),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": _DEFAULT_FTOL, "maxiter": _DEFAULT_MAXITER, "disp": False},
        )
        w = res.x
        w = np.where(w < 1e-9, 0.0, w)
        w = w / w.sum()

    r = float(w @ mu)
    var = float(w @ sigma @ w)
    std = float(np.sqrt(max(var, 0.0)))
    return PortfolioResult(
        weights=w,
        expected_return=r,
        variance=var,
        std=std,
        utility=r,  # A=0 degenerates to expected return
        sharpe=r / std if std > 1e-12 else 0.0,
        success=True,
        message="GMVP analytical" if allow_short else "GMVP SLSQP",
    )


# ---------------------------------------------------------------------------
# Efficient frontier  (Part 1)
# ---------------------------------------------------------------------------

def efficient_frontier(
    mu: np.ndarray,
    sigma: np.ndarray,
    n_points: int = 40,
    allow_short: bool = True,
    target_range: tuple[float, float] | None = None,
) -> FrontierResult:
    """Compute a grid of frontier points (target_return -> min variance).

    * allow_short=True uses the closed-form matrix-algebra solution from Part 1.
    * allow_short=False uses per-target SLSQP (matches Frontier_NoShort_Constrained).
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float)
    n = mu.size
    ones = np.ones(n)

    if target_range is None:
        lo = float(min(mu.min(), mu.mean() - 2 * mu.std()))
        hi = float(max(mu.max(), mu.mean() + 2 * mu.std()))
    else:
        lo, hi = target_range
    targets = np.linspace(lo, hi, n_points)

    variances = np.zeros(n_points)
    stds = np.zeros(n_points)
    weights_grid = np.zeros((n_points, n))

    if allow_short:
        inv = np.linalg.inv(sigma)
        A_ = float(ones @ inv @ mu)
        B_ = float(mu @ inv @ mu)
        C_ = float(ones @ inv @ ones)
        D_ = B_ * C_ - A_ * A_
        # w*(mu_p) = [(C mu - A 1) mu_p + (B 1 - A mu)] / D * Sigma^-1
        for i, mu_p in enumerate(targets):
            lam = (C_ * mu_p - A_) / D_
            gam = (B_ - A_ * mu_p) / D_
            w = inv @ (lam * mu + gam * ones)
            weights_grid[i] = w
            var = float(w @ sigma @ w)
            variances[i] = var
            stds[i] = float(np.sqrt(max(var, 0.0)))
        gmvp_ret = A_ / C_
        gmvp_std = float(np.sqrt(1.0 / C_))
        gmvp_w = (inv @ ones) / C_
    else:
        bounds = [(0.0, 1.0)] * n
        for i, mu_p in enumerate(targets):
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w, m=mu_p: float(w @ mu) - m},
            ]
            res = minimize(
                lambda w: float(w @ sigma @ w),
                np.full(n, 1.0 / n),
                jac=lambda w: 2.0 * (sigma @ w),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": _DEFAULT_FTOL, "maxiter": _DEFAULT_MAXITER, "disp": False},
            )
            w = np.where(res.x < 1e-9, 0.0, res.x)
            if w.sum() > 0:
                w = w / w.sum()
            weights_grid[i] = w
            var = float(w @ sigma @ w)
            variances[i] = var
            stds[i] = float(np.sqrt(max(var, 0.0)))
        g = gmvp(mu, sigma, allow_short=False)
        gmvp_ret, gmvp_std, gmvp_w = g.expected_return, g.std, g.weights

    return FrontierResult(
        target_returns=targets,
        variances=variances,
        stds=stds,
        weights=weights_grid,
        gmvp_return=gmvp_ret,
        gmvp_std=gmvp_std,
        gmvp_weights=gmvp_w,
    )


# ---------------------------------------------------------------------------
# Backtest  (Part 3 value-add)
# ---------------------------------------------------------------------------

def backtest(
    weights: Sequence[float],
    prices: pd.DataFrame,
    initial_capital: float = 10_000.0,
    rebalance: bool = False,
) -> pd.DataFrame:
    """Simulate a portfolio's equity curve over the price history.

    Returns a DataFrame with columns ``['value', 'pct_return']`` indexed like
    ``prices``.

    If rebalance=False, weights are fixed at t=0 and allowed to drift - this is
    what most retail Robo-Advisers do.  If True, we rebalance each period back
    to target weights (useful for stress-testing the allocation in isolation).
    """
    w = np.asarray(weights, dtype=float).ravel()
    if not np.isclose(w.sum(), 1.0, atol=1e-4):
        raise ValueError(f"weights must sum to 1 (got {w.sum():.4f})")

    prices = prices.dropna(how="any")
    returns = prices.pct_change().fillna(0.0).to_numpy()

    values = np.zeros(len(returns))
    if rebalance:
        values[0] = initial_capital
        for t in range(1, len(returns)):
            port_ret = float(w @ returns[t])
            values[t] = values[t - 1] * (1.0 + port_ret)
    else:
        # Simulate buy-and-hold: allocate initial_capital across assets once.
        holdings = initial_capital * w / prices.iloc[0].to_numpy()
        values = (prices.to_numpy() * holdings).sum(axis=1)

    pct = values / initial_capital - 1.0
    return pd.DataFrame(
        {"value": values, "pct_return": pct},
        index=prices.index,
    )


def sharpe_ratio(mu_p: float, sigma_p: float, risk_free: float = 0.0) -> float:
    return (mu_p - risk_free) / sigma_p if sigma_p > 1e-12 else 0.0


__all__ = [
    "ANNUALIZATION_FACTOR",
    "PortfolioResult",
    "FrontierResult",
    "compute_mu_sigma",
    "optimize_portfolio",
    "gmvp",
    "efficient_frontier",
    "backtest",
    "sharpe_ratio",
]
