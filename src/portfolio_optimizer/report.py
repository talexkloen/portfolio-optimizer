"""Reproducible figures and machine-readable research outputs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from .data import estimate_moments, price_returns
from .evaluation import holdout_metrics
from .optimizer import PortfolioOptimizer


def build_report(
    prices: pd.DataFrame,
    output: str | Path,
    *,
    risk_free_rate: float = 0.02,
    max_weight: float = 0.40,
    shrinkage: float = 0.10,
    train_fraction: float = 0.75,
    periods: int = 252,
    source: str = "user-supplied prices",
) -> dict:
    """Fit only on the first chronological segment, then evaluate frozen weights."""
    if not 0.1 <= train_fraction <= 0.9:
        raise ValueError("train_fraction must lie between 0.1 and 0.9.")
    returns = price_returns(prices)
    split = int(len(returns) * train_fraction)
    train, test = returns.iloc[:split], returns.iloc[split:]
    if min(len(train), len(test)) < 3:
        raise ValueError("Training and test sets each need at least three observations.")
    mu, cov = estimate_moments(train, periods=periods, shrinkage=shrinkage)
    model = PortfolioOptimizer(mu, cov, risk_free_rate, upper=max_weight)
    portfolios = {
        "Minimum variance": model.minimum_variance(),
        "Maximum Sharpe": model.maximum_sharpe(),
        "Equal weight": model.evaluate(np.full(len(mu), 1 / len(mu))),
    }
    frontier = model.efficient_frontier()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    weights = pd.DataFrame({name: p.weights for name, p in portfolios.items()})
    training = pd.DataFrame(
        {
            name: {
                "expected_return": p.expected_return,
                "volatility": p.volatility,
                "sharpe": p.sharpe,
            }
            for name, p in portfolios.items()
        }
    ).T
    heldout = pd.DataFrame(
        {
            name: holdout_metrics(test, p.weights, risk_free_rate, periods)
            for name, p in portfolios.items()
        }
    ).T
    weights.to_csv(output / "weights.csv", index_label="asset")
    training.to_csv(output / "training_metrics.csv", index_label="portfolio")
    heldout.to_csv(output / "holdout_metrics.csv", index_label="portfolio")
    frontier.to_csv(output / "frontier.csv", index=False)
    colors = ["#1d5b79", "#db815b", "#79958c"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "portfolio-optimizer",
        }
    )
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor("#f7f8fa")
    for pane in (ax, bx):
        pane.set_facecolor("#f7f8fa")
        pane.grid(axis="y", alpha=0.18)
        pane.set_axisbelow(True)
    ax.plot(
        frontier.volatility,
        frontier.expected_return,
        color="#1d5b79",
        linewidth=2.5,
        label="Constrained efficient frontier",
    )
    for (name, p), color, marker in zip(portfolios.items(), colors, ["o", "*", "D"], strict=True):
        ax.scatter(
            p.volatility,
            p.expected_return,
            color=color,
            marker=marker,
            s=180 if marker == "*" else 65,
            edgecolors="white",
            zorder=5,
            label=name,
        )
    ax.set(
        xlabel="Annualized volatility",
        ylabel="Annualized expected return",
        title="01  /  Risk–return opportunity set",
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False, loc="best", fontsize=9)
    weights.plot.bar(ax=bx, color=colors, width=0.78, rot=30)
    bx.set(title="02  /  Portfolio allocations", xlabel="", ylabel="Weight")
    bx.yaxis.set_major_formatter(PercentFormatter(1))
    bx.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "PORTFOLIO OPTIMIZER",
        x=0.07,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color="#143347",
    )
    fig.text(
        0.07,
        0.895,
        f"{source}  •  training estimates  •  {max_weight:.0%} position cap"
        f"  •  risk-free rate {risk_free_rate:.1%}",
        color="#576574",
        fontsize=10,
    )
    fig.text(
        0.07,
        0.02,
        "Personal practice project. Estimated returns are not forecasts or realized performance.",
        fontsize=8,
        color="#576574",
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.99, 0.87))
    for extension in ("svg", "png"):
        fig.savefig(
            output / f"efficient_frontier.{extension}",
            dpi=180,
            metadata={"Date": None} if extension == "svg" else None,
        )
    plt.close(fig)
    metadata = {
        "source": source,
        "train_start": str(train.index[0].date()),
        "train_end": str(train.index[-1].date()),
        "test_start": str(test.index[0].date()),
        "test_end": str(test.index[-1].date()),
        "train_observations": len(train),
        "test_observations": len(test),
        "risk_free_rate": risk_free_rate,
        "max_weight": max_weight,
        "shrinkage": shrinkage,
        "periods_per_year": periods,
        "holdout_assumption": "Daily rebalancing to fixed training weights; no costs",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
