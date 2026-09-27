"""Data hygiene, time separation, drawdowns and end-to-end command-line checks."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from mafinrisk import estimate_moments, load_prices, price_returns, synthetic_prices
from mafinrisk.evaluation import holdout_metrics
from mafinrisk.report import build_report


def test_sample_reproducibility_and_roundtrip(tmp_path):
    a = synthetic_prices()
    pd.testing.assert_frame_equal(a, synthetic_prices())
    assert not a.equals(synthetic_prices(43))
    path = tmp_path / "prices.csv"
    a.to_csv(path)
    np.testing.assert_allclose(load_prices(path), a)
    assert (a > 0).all().all()


def test_estimation_matches_hand_calculation():
    r = pd.DataFrame(
        [[0.01, 0.02], [-0.02, 0.03], [0.04, -0.01]],
        index=pd.date_range("2024-01-01", periods=3),
        columns=["A", "B"],
    )
    mu, cov = estimate_moments(r, periods=12, shrinkage=0)
    np.testing.assert_allclose(mu, np.mean(r, axis=0) * 12)
    np.testing.assert_allclose(cov, np.cov(r.T, ddof=1) * 12)
    _, shrunk = estimate_moments(r, periods=12, shrinkage=1)
    np.testing.assert_allclose(shrunk, np.diag(np.diag(cov)))


@pytest.mark.parametrize("kind", ["missing", "negative", "duplicate_date", "unsorted", "infinite"])
def test_bad_prices_rejected(kind):
    p = synthetic_prices(observations=12)
    if kind == "missing":
        p.iloc[3, 0] = np.nan
    elif kind == "negative":
        p.iloc[3, 0] = -1
    elif kind == "infinite":
        p.iloc[3, 0] = np.inf
    elif kind == "unsorted":
        p = p.iloc[::-1]
    else:
        p.index = [p.index[0]] * len(p)
        p.index = pd.DatetimeIndex(p.index)
    with pytest.raises(ValueError):
        price_returns(p)


def test_csv_duplicate_headers_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("date,A,A\n2024-01-01,1,2\n")
    with pytest.raises(ValueError, match="unique"):
        load_prices(p)


def test_drawdown_includes_first_loss():
    r = pd.DataFrame(
        [[-0.1, -0.1], [0, 0], [0.05, 0.05]],
        index=pd.date_range("2024-01-01", periods=3),
        columns=["A", "B"],
    )
    metrics = holdout_metrics(r, pd.Series([0.5, 0.5], index=r.columns))
    assert metrics["max_drawdown"] == pytest.approx(-0.1)
    assert metrics["total_return"] == pytest.approx(0.9 * 1.05 - 1)


def test_report_no_lookahead(tmp_path):
    prices = synthetic_prices()
    first = tmp_path / "first"
    second = tmp_path / "second"
    metadata = build_report(prices, first, source="Synthetic demonstration")
    changed = prices.copy()
    # Change only prices after the last training return; estimated weights must not move.
    split = int((len(prices) - 1) * 0.75)
    changed.iloc[split + 1 :, 0] *= np.linspace(1, 2, len(prices) - split - 1)
    build_report(changed, second)
    pd.testing.assert_frame_equal(
        pd.read_csv(first / "weights.csv"), pd.read_csv(second / "weights.csv")
    )
    assert metadata["train_end"] < metadata["test_start"]
    assert (first / "efficient_frontier.svg").stat().st_size > 1000
    assert (first / "efficient_frontier.png").stat().st_size > 1000
    assert json.loads((first / "metadata.json").read_text())["test_observations"] == 252
    assert not pd.read_csv(first / "holdout_metrics.csv").equals(
        pd.read_csv(second / "holdout_metrics.csv")
    )


def test_cli_errors_and_sample(tmp_path):
    path = tmp_path / "prices.csv"
    good = subprocess.run(
        [sys.executable, "-m", "mafinrisk.cli", "sample", "--output", str(path)],
        capture_output=True,
        text=True,
    )
    assert good.returncode == 0, good.stderr
    assert len(load_prices(path)) == 1009
    bad = subprocess.run(
        [
            sys.executable,
            "-m",
            "mafinrisk.cli",
            "analyze",
            "--prices",
            str(path),
            "--max-weight",
            ".01",
            "--output",
            str(tmp_path / "bad"),
        ],
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 2
    assert "Infeasible" in bad.stderr
