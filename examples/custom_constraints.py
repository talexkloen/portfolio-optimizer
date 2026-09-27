"""Run from the repository root after pip install -e ."""

import pandas as pd

from portfolio_optimizer import PortfolioOptimizer, estimate_moments, load_prices, price_returns

prices = load_prices("data/sample_prices.csv")
returns = price_returns(prices)
mu, cov = estimate_moments(returns.iloc[:756], shrinkage=0.10)
lower = pd.Series(0.0, index=mu.index)
upper = pd.Series(0.40, index=mu.index)
lower["Gov_Bonds"] = 0.10
upper["EM_Equity"] = 0.20
model = PortfolioOptimizer(mu, cov, risk_free_rate=0.02, lower=lower, upper=upper)
print("Maximum Sharpe with a bond floor and emerging-market cap:")
print(model.maximum_sharpe().weights.round(4))
print("Minimum variance with an annual expected-return floor of 6%:")
print(model.minimum_variance(target_return=0.06).weights.round(4))
