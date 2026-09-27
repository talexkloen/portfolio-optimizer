"""Chronological holdout evaluation; weights are fixed and rebalanced daily."""

import numpy as np
import pandas as pd

from .data import _validate_frame


def holdout_metrics(
    returns: pd.DataFrame,
    weights: pd.Series,
    risk_free_rate: float = 0.02,
    periods: int = 252,
) -> dict[str, float]:
    """Gross-of-cost realized statistics for a constant-weight portfolio.

    Daily rebalancing is assumed. The initial wealth observation is included in
    the drawdown peak, so a loss on the first test day is correctly counted.
    """
    _validate_frame(returns, 3)
    if not isinstance(periods, int) or isinstance(periods, bool) or periods <= 0:
        raise ValueError("periods must be a positive integer.")
    if not np.isfinite(risk_free_rate):
        raise ValueError("Risk-free rate must be finite.")
    if not isinstance(weights, pd.Series) or not weights.index.equals(returns.columns):
        raise ValueError("Weight labels and order must match the return columns.")
    w = weights.to_numpy(dtype=float)
    if not np.isfinite(w).all() or (w < 0).any() or abs(w.sum() - 1) > 1e-7:
        raise ValueError("Weights must be finite, nonnegative, and sum to one.")
    if (returns.to_numpy(dtype=float) <= -1).any():
        raise ValueError("Simple returns must exceed -100%.")
    daily = returns.to_numpy(dtype=float) @ w
    wealth = np.r_[1.0, np.cumprod(1 + daily)]
    vol = float(daily.std(ddof=1) * np.sqrt(periods))
    mean = float(daily.mean() * periods)
    return {
        "annualized_return": mean,
        "annualized_volatility": vol,
        "sharpe": (mean - risk_free_rate) / vol if vol > 1e-14 else float("nan"),
        "cagr": float(wealth[-1] ** (periods / len(daily)) - 1),
        "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
        "total_return": float(wealth[-1] - 1),
    }
