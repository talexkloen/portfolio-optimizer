"""Convex minimum-variance and positive-excess maximum-Sharpe optimization."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize


class OptimizationError(RuntimeError):
    """The solver failed or a computed portfolio failed independent feasibility checks."""


@dataclass(frozen=True)
class Portfolio:
    """Annualized ex-ante statistics and asset-labelled portfolio weights."""

    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float


class PortfolioOptimizer:
    """Fully invested, long-only allocation with scalar or per-asset box bounds.

    Inputs are annual arithmetic expected returns, annual covariance, and an annual
    arithmetic risk-free rate. Covariance must be symmetric positive definite.
    """

    def __init__(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        risk_free_rate: float = 0.02,
        lower: float | pd.Series = 0.0,
        upper: float | pd.Series = 1.0,
    ) -> None:
        if not isinstance(expected_returns, pd.Series) or len(expected_returns) < 2:
            raise ValueError("Expected returns must be a Series with at least two assets.")
        names = expected_returns.index
        if not names.is_unique or any(not isinstance(n, str) or not n for n in names):
            raise ValueError("Asset names must be unique nonempty strings.")
        if not isinstance(covariance, pd.DataFrame):
            raise ValueError("Covariance must be an asset-labelled DataFrame.")
        if not covariance.index.equals(names) or not covariance.columns.equals(names):
            raise ValueError("Covariance labels and order must match expected returns exactly.")
        self.names = names.copy()
        self.mu = expected_returns.to_numpy(dtype=float, copy=True)
        self.cov = covariance.to_numpy(dtype=float, copy=True)
        self.rf = float(risk_free_rate)
        if not np.isfinite(self.mu).all() or not np.isfinite(self.cov).all():
            raise ValueError("Expected returns and covariance must be finite.")
        if not np.isfinite(self.rf):
            raise ValueError("Risk-free rate must be finite.")
        if not np.allclose(self.cov, self.cov.T, atol=1e-12, rtol=1e-10):
            raise ValueError("Covariance must be symmetric.")
        self.cov = (self.cov + self.cov.T) / 2
        if np.linalg.eigvalsh(self.cov).min() <= 1e-12:
            raise ValueError(
                "Covariance must be positive definite; use shrinkage or remove assets."
            )
        self.lower, self.upper = self._bounds(lower), self._bounds(upper)
        if (self.lower < 0).any() or (self.upper > 1).any():
            raise ValueError("Bounds must lie in [0, 1]; shorting and leverage are unsupported.")
        if (self.lower > self.upper).any():
            raise ValueError("Each lower bound must not exceed its upper bound.")
        if self.lower.sum() > 1 + 1e-10 or self.upper.sum() < 1 - 1e-10:
            raise ValueError("Infeasible bounds: weights must be able to sum to one.")
        self._box = list(zip(self.lower, self.upper, strict=True))
        self._scale = float(np.diag(self.cov).max())
        self._start = self._linear(np.zeros(len(self.mu)))

    def _bounds(self, value: float | pd.Series) -> np.ndarray:
        if isinstance(value, pd.Series):
            if not value.index.equals(self.names):
                raise ValueError("Bound labels and order must match assets exactly.")
            result = value.to_numpy(dtype=float, copy=True)
        elif np.isscalar(value):
            result = np.full(len(self.mu), float(value))
        else:
            raise ValueError("Bounds must be scalars or asset-labelled Series.")
        if not np.isfinite(result).all():
            raise ValueError("Bounds must be finite.")
        return result

    def _linear(self, objective: np.ndarray) -> np.ndarray:
        result = linprog(
            objective, A_eq=np.ones((1, len(self.mu))), b_eq=[1], bounds=self._box, method="highs"
        )
        if not result.success:
            raise OptimizationError(f"Feasibility problem failed: {result.message}")
        return result.x

    def evaluate(self, weights: np.ndarray | pd.Series) -> Portfolio:
        """Validate an allocation and evaluate it using this model's annual moments."""
        if isinstance(weights, pd.Series) and not weights.index.equals(self.names):
            raise ValueError("Weight labels and order must match assets exactly.")
        w = np.asarray(weights, dtype=float)
        if w.shape != self.mu.shape or not np.isfinite(w).all():
            raise ValueError("Weights must be a finite vector with one entry per asset.")
        if (
            abs(w.sum() - 1) > 1e-7
            or (w < self.lower - 1e-7).any()
            or (w > self.upper + 1e-7).any()
        ):
            raise ValueError("Weights violate full-investment or asset-bound constraints.")
        ret = float(w @ self.mu)
        vol = float(np.sqrt(w @ self.cov @ w))
        return Portfolio(
            pd.Series(w.copy(), index=self.names, name="weight"), ret, vol, (ret - self.rf) / vol
        )

    def minimum_variance(self, target_return: float | None = None) -> Portfolio:
        """Minimize variance, optionally requiring expected return >= target_return."""
        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(len(w))}
        ]
        start = self._start
        if target_return is not None:
            if not np.isfinite(target_return):
                raise ValueError("Target return must be finite.")
            highest = self._linear(-self.mu)
            if target_return > self.mu @ highest + 1e-9:
                raise ValueError("Target return exceeds the feasible maximum.")
            start = highest
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: w @ self.mu - target_return,
                    "jac": lambda w: self.mu,
                }
            )
        matrix = self.cov / self._scale
        result = minimize(
            lambda w: w @ matrix @ w,
            start,
            jac=lambda w: 2 * matrix @ w,
            bounds=self._box,
            constraints=constraints,
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 2000},
        )
        portfolio = self._checked(result)
        if target_return is not None and portfolio.expected_return < target_return - 1e-7:
            raise OptimizationError("Solver returned a portfolio below the requested target.")
        return portfolio

    def maximum_sharpe(self) -> Portfolio:
        """Solve the positive-excess Sharpe problem as a convex quadratic program.

        Set y = w / (a @ w), where a is scaled excess return. Then a @ y = 1,
        and lower * sum(y) <= y <= upper * sum(y). Minimize y' Sigma y and
        recover w = y / sum(y). A positive feasible excess return is required.
        """
        excess = self.mu - self.rf
        highest = self._linear(-excess)
        if highest @ excess <= 1e-10:
            raise ValueError(
                "No feasible positive excess return; maximum-Sharpe is undefined here."
            )
        a = excess / np.max(np.abs(excess))
        start = highest / (a @ highest)
        n = len(a)
        lower_matrix = np.eye(n) - self.lower[:, None] * np.ones((1, n))
        upper_matrix = self.upper[:, None] * np.ones((1, n)) - np.eye(n)
        matrix = self.cov / self._scale
        result = minimize(
            lambda y: y @ matrix @ y,
            start,
            jac=lambda y: 2 * matrix @ y,
            method="SLSQP",
            bounds=[(0, None)] * n,
            constraints=[
                {"type": "eq", "fun": lambda y: a @ y - 1, "jac": lambda y: a},
                {"type": "ineq", "fun": lambda y: lower_matrix @ y, "jac": lambda y: lower_matrix},
                {"type": "ineq", "fun": lambda y: upper_matrix @ y, "jac": lambda y: upper_matrix},
            ],
            options={"ftol": 1e-12, "maxiter": 2000},
        )
        if not result.success or abs(a @ result.x - 1) > 1e-7 or result.x.sum() <= 0:
            raise OptimizationError(f"Maximum-Sharpe solver failed: {result.message}")
        result.x = result.x / result.x.sum()
        return self._checked(result)

    def _checked(self, result) -> Portfolio:
        if not result.success:
            raise OptimizationError(f"Optimization failed: {result.message}")
        try:
            return self.evaluate(result.x)
        except ValueError as exc:
            raise OptimizationError(f"Solver feasibility check failed: {exc}") from exc

    def efficient_frontier(self, points: int = 60) -> pd.DataFrame:
        """Return the upper efficient branch from GMV to maximum feasible return."""
        if not isinstance(points, int) or isinstance(points, bool) or points < 2:
            raise ValueError("points must be an integer >= 2.")
        gmv = self.minimum_variance()
        highest = float(self._linear(-self.mu) @ self.mu)
        rows = []
        for target in np.linspace(gmv.expected_return, highest, points):
            p = self.minimum_variance(float(target))
            rows.append(
                {
                    "target_return": target,
                    "expected_return": p.expected_return,
                    "volatility": p.volatility,
                    "sharpe": p.sharpe,
                    **{f"weight_{k}": v for k, v in p.weights.items()},
                }
            )
        return pd.DataFrame(rows)
