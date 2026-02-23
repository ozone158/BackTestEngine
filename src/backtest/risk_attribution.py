"""
Step 5.2 — Risk attribution (regression-based and optional PCA).

Regress strategy returns on factor/benchmark returns: strategy = alpha + beta * market + error.
Alpha = excess return not explained by market; beta = market exposure.
Beta near 0 implies market-neutral. Pairs strategy should have beta near 0.

Methods: OLS (default) or PCA (5.2.3): regress on first n_components of PCA of factor returns.
Handles insufficient data and zero-variance series by returning NaN where appropriate.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np


def _align_returns(
    strategy_returns: np.ndarray,
    factor_returns: Union[np.ndarray, None],
    factor_matrix: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    5.2.1: Align strategy and factor return series (same length).
    Drops indices where any series has NaN. Returns (y, X) for OLS/PCA.
    """
    y = np.asarray(strategy_returns, dtype=float).ravel()
    if factor_matrix is not None:
        X_factors = np.asarray(factor_matrix, dtype=float)
        if X_factors.ndim == 1:
            X_factors = X_factors.reshape(-1, 1)
        n_obs = min(len(y), len(X_factors))
        y, X_factors = y[:n_obs], X_factors[:n_obs]
        mask = np.isfinite(y) & np.all(np.isfinite(X_factors), axis=1)
    elif factor_returns is not None:
        x = np.asarray(factor_returns, dtype=float).ravel()
        n_obs = min(len(y), len(x))
        y, x = y[:n_obs], x[:n_obs]
        mask = np.isfinite(y) & np.isfinite(x)
        X_factors = x[mask].reshape(-1, 1)
    else:
        raise ValueError("Provide either factor_returns or factor_matrix")
    y = y[mask]
    if factor_matrix is not None:
        X_factors = X_factors[mask]
    return y, X_factors


def _pca_project(X: np.ndarray, n_components: int) -> np.ndarray:
    """Center X and project onto first n_components principal components. Returns (n_obs, n_components)."""
    X_centered = X - np.nanmean(X, axis=0)
    if X_centered.shape[1] == 0 or n_components <= 0:
        return np.empty((X.shape[0], 0))
    n_components = min(n_components, X_centered.shape[1], X_centered.shape[0] - 1)
    if n_components <= 0:
        return np.empty((X.shape[0], 0))
    try:
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        components = (U[:, :n_components] * s[:n_components]).astype(float)
        return components
    except np.linalg.LinAlgError:
        return np.empty((X.shape[0], 0))


def _ols_alpha_beta(y: np.ndarray, X_factors: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    5.2.2: OLS regression y = alpha + X_factors @ beta_vec.
    X: n x k (factors). Prepends column of ones for intercept.
    Returns (alpha, beta_vec, residuals). Returns NaN alpha/beta when variance is zero or underdetermined.
    """
    n, k = X_factors.shape
    y_var = np.nanvar(y)
    if n < 2 or (k > 0 and not np.isfinite(y_var)) or (k > 0 and y_var == 0 and np.all(np.isfinite(y))):
        alpha = float(np.nanmean(y)) if n > 0 else np.nan
        beta_vec = np.full(k, np.nan)
        residuals = y - alpha if n > 0 else np.array([])
        return alpha, beta_vec, residuals
    X = np.column_stack([np.ones(n), X_factors])
    if n <= k + 1:
        alpha = float(np.mean(y))
        beta_vec = np.zeros(k)
        residuals = y - alpha
        return alpha, beta_vec, residuals
    coeffs, _res, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha = float(coeffs[0])
    beta_vec = np.asarray(coeffs[1:], dtype=float)
    fitted = X @ coeffs
    residuals = y - fitted
    return alpha, beta_vec, residuals


def decompose(
    strategy_returns: Union[list, np.ndarray],
    factor_returns: Optional[Union[list, np.ndarray]] = None,
    factor_matrix: Optional[Union[list, np.ndarray]] = None,
    periods_per_year: float = 252.0,
    method: str = "ols",
    n_components: Optional[int] = None,
) -> Dict[str, Any]:
    """
    5.2.5: RiskAttribution.decompose(strategy_returns, factor_returns or factor_matrix) -> dict.

    Args:
        strategy_returns: Strategy return series (same frequency as backtest).
        factor_returns: Single factor/benchmark return series (e.g. market index), aligned length.
        factor_matrix: Optional (n_obs x n_factors) instead of factor_returns for multi-factor.
        periods_per_year: For annualizing alpha (e.g. 252 for daily). Alpha in output is annualized.
        method: "ols" (default) or "pca". PCA regresses strategy returns on first n_components PCs of factors.
        n_components: For method="pca", number of components (default 2). Ignored for OLS.

    Returns:
        dict with alpha (annualized), beta (scalar or array), residuals (series), R2.
        When data are insufficient or zero-variance, alpha/beta may be NaN.
        Beta near 0 implies market-neutral. Alpha is strategy return not explained by market (5.2.6).
    """
    y, X_factors = _align_returns(
        np.asarray(strategy_returns),
        factor_returns,
        factor_matrix,
    )
    k = X_factors.shape[1] if len(X_factors) > 0 else 0
    nan_beta: Union[float, np.ndarray] = float("nan") if k <= 1 else np.full(k, np.nan)

    if len(y) < 2:
        if len(y) == 0:
            return {"alpha": np.nan, "beta": nan_beta, "residuals": np.array([]), "R2": np.nan}
        return {
            "alpha": 0.0,
            "beta": 0.0 if k == 1 else (np.zeros(k) if k > 1 else nan_beta),
            "residuals": np.array([]),
            "R2": 0.0,
        }
    if not np.any(np.isfinite(y)):
        return {"alpha": np.nan, "beta": nan_beta, "residuals": np.asarray(y), "R2": np.nan}

    if method == "pca" and X_factors.shape[1] > 0:
        nc = n_components if n_components is not None else min(2, X_factors.shape[1])
        X_proj = _pca_project(X_factors, nc)
        if X_proj.shape[1] == 0:
            alpha_per_period, beta_vec, residuals = _ols_alpha_beta(y, X_factors)
        else:
            alpha_per_period, beta_vec, residuals = _ols_alpha_beta(y, X_proj)
    else:
        alpha_per_period, beta_vec, residuals = _ols_alpha_beta(y, X_factors)

    alpha_annual = alpha_per_period * periods_per_year if np.isfinite(alpha_per_period) else np.nan
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum(residuals ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else (0.0 if np.isfinite(ss_res) else np.nan)
    beta_out: Union[float, np.ndarray] = float(beta_vec[0]) if len(beta_vec) == 1 else beta_vec

    return {
        "alpha": alpha_annual,
        "beta": beta_out,
        "residuals": residuals,
        "R2": r2,
    }


class RiskAttribution:
    """
    5.2.5: API for risk attribution.

    Regression-based (OLS, default) or PCA: strategy_returns = alpha + beta * factors + error.
    Pairs strategy should have beta near 0 (market-neutral). Alpha is excess return not explained by market (5.2.6).
    """

    @staticmethod
    def decompose(
        strategy_returns: Union[list, np.ndarray],
        factor_returns: Optional[Union[list, np.ndarray]] = None,
        factor_matrix: Optional[Union[list, np.ndarray]] = None,
        periods_per_year: float = 252.0,
        method: str = "ols",
        n_components: Optional[int] = None,
    ) -> Dict[str, Any]:
        """See module-level decompose()."""
        return decompose(
            strategy_returns,
            factor_returns=factor_returns,
            factor_matrix=factor_matrix,
            periods_per_year=periods_per_year,
            method=method,
            n_components=n_components,
        )
