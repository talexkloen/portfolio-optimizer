"""Strict price validation, transparent estimators, and reproducible synthetic data."""

from pathlib import Path

import numpy as np
import pandas as pd


def _validate_frame(frame: pd.DataFrame, min_rows: int) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.shape[0] < min_rows or frame.shape[1] < 2:
        raise ValueError(f"Provide at least {min_rows} observations and two assets.")
    if not frame.columns.is_unique or any(not isinstance(c, str) or not c for c in frame):
        raise ValueError("Asset names must be unique nonempty strings.")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans:
        raise ValueError("Use a valid DatetimeIndex.")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and increasing.")
    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("All observations must be numeric.") from exc
    if not np.isfinite(values).all():
        raise ValueError(
            "Missing or infinite observations are not permitted; align data explicitly."
        )


def load_prices(path: str | Path) -> pd.DataFrame:
    """Read date + adjusted total-return price columns; never forward-fill gaps."""
    import csv

    with Path(path).open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), [])
    if not header or header[0] != "date" or len(header) != len(set(header)):
        raise ValueError("CSV must start with 'date' and have unique column names.")
    frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
    _validate_frame(frame, 4)
    if (frame.to_numpy(dtype=float) <= 0).any():
        raise ValueError("Prices must be strictly positive.")
    return frame.astype(float)


def price_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple returns P[t]/P[t-1] - 1 with no implicit missing-data filling."""
    _validate_frame(prices, 4)
    if (prices.to_numpy(dtype=float) <= 0).any():
        raise ValueError("Prices must be strictly positive.")
    return prices.astype(float).pct_change(fill_method=None).iloc[1:]


def estimate_moments(
    returns: pd.DataFrame,
    periods: int = 252,
    shrinkage: float = 0.10,
) -> tuple[pd.Series, pd.DataFrame]:
    """Annualize arithmetic mean and sample covariance; shrink toward its diagonal.

    Shrinkage is a user-selected intensity, not a fitted Ledoit-Wolf estimator.
    """
    _validate_frame(returns, 3)
    if not isinstance(periods, int) or isinstance(periods, bool) or periods <= 0:
        raise ValueError("periods must be a positive integer.")
    if not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be between 0 and 1.")
    if (returns.to_numpy(dtype=float) <= -1).any():
        raise ValueError("Simple asset returns must exceed -100%.")
    sample = returns.astype(float).cov() * periods
    covariance = (1 - shrinkage) * sample + shrinkage * np.diag(np.diag(sample))
    return returns.astype(float).mean() * periods, covariance


def synthetic_prices(seed: int = 42, observations: int = 1009) -> pd.DataFrame:
    """Generate fictional correlated asset indices; no market data or ticker claims."""
    if not isinstance(observations, int) or observations < 4:
        raise ValueError("observations must be an integer of at least four.")
    rng = np.random.default_rng(seed)
    names = ["US_Equity", "Europe_Equity", "EM_Equity", "Gov_Bonds", "Corp_Bonds", "Gold"]
    annual_drift = np.array([0.09, 0.075, 0.095, 0.025, 0.045, 0.055])
    loadings = np.array(
        [
            [0.15, 0.02, 0],
            [0.13, 0.04, 0.02],
            [0.16, 0.01, 0.04],
            [-0.02, 0.045, 0],
            [0.035, 0.045, 0.005],
            [0.015, -0.01, 0.12],
        ]
    )
    covariance = loadings @ loadings.T + np.diag(
        np.array([0.08, 0.08, 0.11, 0.025, 0.035, 0.07]) ** 2
    )
    increments = rng.multivariate_normal(
        (annual_drift - 0.5 * np.diag(covariance)) / 252,
        covariance / 252,
        size=observations - 1,
    )
    prices = np.vstack([np.ones(6), np.exp(np.cumsum(increments, axis=0))]) * 100
    return pd.DataFrame(
        prices, index=pd.bdate_range("2021-01-04", periods=observations, name="date"), columns=names
    )
