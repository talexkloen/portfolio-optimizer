# Explaining the project in an interview

## A 45-second introduction

“I built a reproducible Python allocation tool to explore how constraints change the efficient frontier. It estimates annual return and covariance from a training window, applies covariance shrinkage and position limits, and solves minimum-variance and maximum-Sharpe portfolios. I validated the optimizer against analytical solutions and evaluated the frozen allocations on a later window. The demo uses synthetic data, so the focus is numerical correctness and research design rather than a claimed trading edge.”

## Questions to be ready for

1. **Why does diversification work?** Portfolio variance depends on covariances, not just individual variances. Less-than-perfectly correlated returns can reduce aggregate variance.
2. **Why is maximum Sharpe unstable?** It depends on expected returns, which are particularly noisy. An optimizer can turn estimation errors into concentrated allocations.
3. **Why shrink covariance?** Reducing estimated correlations can reduce sampling noise and improve conditioning, at the price of bias. The current intensity is an explicit assumption, not a fitted optimal value.
4. **What makes the Sharpe solve convex?** Normalize positive expected excess return to one, transform weight bounds into linear inequalities, and minimize variance in the transformed variables. Explain the mapping back to weights.
5. **What does the holdout prove?** It tests the evaluation plumbing on unseen observations. One synthetic holdout cannot establish generalization to markets or statistically significant outperformance.
6. **What would you add next?** Rolling estimation, transaction costs and turnover, alternative expected-return estimators, robust optimization, and regime-aware evaluation on licensed adjusted market data.
7. **Is this a buy-and-hold backtest?** No. Constant target weights imply daily rebalancing. Holding shares would allow weights to drift and needs a different simulation.
8. **How did you use AI?** Explain honestly which parts were assisted, what you reviewed, and how you verified the mathematics and tests. Be able to derive and modify the core methods yourself.

## Useful experiments

- Compare shrinkage 0, 0.1, and 0.5 with the same train/test split.
- Relax the maximum weight from 40% to 100% and inspect concentration.
- Perturb one expected return by one percentage point and observe maximum-Sharpe sensitivity.
- Change the risk-free assumption and explain why the minimum-variance solution stays the same.
- Repeat across seeds. Discuss why selecting the best seed is selection bias.
