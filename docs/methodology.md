# Mathematical and implementation notes

## Why transform maximum Sharpe?

Let `e = μ − rf·1`. On the fully invested feasible set, excess portfolio return is `eᵀw`. If a positive-excess portfolio exists, the maximizer of Sharpe must also have positive excess (all volatilities are strictly positive under the positive-definite covariance assumption).

Rescale `a = e / max(|e|)` for numerical conditioning; this does not change the optimizer. Define:

\[
y=\frac{w}{a^\top w},\qquad k=\mathbf1^\top y,\qquad w=\frac{y}{k}.
\]

Then `aᵀy=1`, `y≥0`, and allocation bounds become linear:

\[
y_i-\ell_i\mathbf1^\top y\ge0,\qquad
u_i\mathbf1^\top y-y_i\ge0.
\]

The squared Sharpe ratio is a positive constant divided by `yᵀΣy`. Maximizing Sharpe is therefore equivalent to minimizing this convex quadratic form under the transformed linear constraints. With positive-definite covariance the objective is strictly convex. SLSQP still operates in floating point; successful termination is followed by independent constraint checks, and failures are exposed as `OptimizationError`.

The covariance objective is divided by its largest diagonal entry for numerical scaling. This positive scalar does not change the optimum. The linear feasibility problem uses HiGHS; its maximum-excess solution gives a feasible initial point for the transformed problem. If maximum feasible excess is at most `1e-10`, the method rejects the request rather than reporting a misleading positive-excess solution. Covariance eigenvalues at most `1e-12` are rejected. Feasibility tolerance is `1e-7`; these absolute thresholds assume annual returns expressed as decimals.

## Independent validation

For an interior minimum-variance solution:

\[
w_{GMV}=\frac{\Sigma^{-1}\mathbf1}{\mathbf1^\top\Sigma^{-1}\mathbf1}.
\]

For an interior tangency solution:

\[
w_{SR}=\frac{\Sigma^{-1}(\mu-r_f\mathbf1)}{\mathbf1^\top\Sigma^{-1}(\mu-r_f\mathbf1)}.
\]

Tests use inputs where these solutions satisfy the bounds. A separate two-asset grid test checks the constrained Sharpe ratio against 20,001 feasible allocations. These are independent reference calculations, rather than tests that repeat the solver implementation.

## Evaluation conventions

The report slices returns chronologically **before** estimation. With 1,009 prices there are 1,008 simple returns: 756 for training and 252 for testing. The first held-out return uses the last training price as its starting price, which is known at the decision time.

Daily portfolio returns are `r[t]ᵀw` with unchanged target weights. That describes frictionless daily rebalancing, not buy-and-hold shares. The holdout Sharpe is `(A·mean(rp)−rf)/(sqrt(A)·std(rp, ddof=1))`. CAGR is `(product(1+rp))^(A/T)−1`. Drawdown includes initial wealth of one, and the reported maximum drawdown is nonpositive. If realized volatility is effectively zero, realized Sharpe is undefined and written as a missing CSV field.

There is no risk-free asset in the allocation universe. The risk-free rate is an input to Sharpe, not an available borrowing/lending instrument. The frontier is the risky-asset frontier, not a capital allocation line.

## Deliberate boundaries

No implicit covariance repair, imputation, resampling, automatic asset reordering, or adjustment of infeasible limits occurs. The program fails explicitly so the researcher can decide how to change the data or model. Highly ill-conditioned inputs can still fail numerically even if they pass the positive-definiteness threshold.

The report includes equal weights, so its uniform maximum position weight must be at least `1/n`. The library supports nonuniform bounds; a general equal-weight benchmark may be infeasible under those custom bounds and is not forced into them.
