"""Command-line entry points for sample generation and portfolio analysis."""

import argparse
from pathlib import Path

from .data import load_prices, synthetic_prices
from .optimizer import OptimizationError


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio optimizer for mean-variance analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample", help="Generate fictional correlated asset price indices")
    sample.add_argument("--output", type=Path, default=Path("data/sample_prices.csv"))
    sample.add_argument("--seed", type=int, default=42)
    analyze = sub.add_parser("analyze", help="Optimize using a chronological train/test split")
    analyze.add_argument("--prices", type=Path, required=True)
    analyze.add_argument("--output", type=Path, default=Path("reports"))
    analyze.add_argument("--risk-free-rate", type=float, default=0.02)
    analyze.add_argument("--max-weight", type=float, default=0.40)
    analyze.add_argument("--shrinkage", type=float, default=0.10)
    analyze.add_argument("--train-fraction", type=float, default=0.75)
    analyze.add_argument("--periods", type=int, default=252)
    analyze.add_argument("--source-label", default="User-supplied prices (provenance unverified)")
    args = parser.parse_args()
    try:
        if args.command == "sample":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            synthetic_prices(args.seed).to_csv(args.output, float_format="%.10f")
            print(f"Synthetic sample written to {args.output}")
        else:
            from .report import build_report

            metadata = build_report(
                load_prices(args.prices),
                args.output,
                risk_free_rate=args.risk_free_rate,
                max_weight=args.max_weight,
                shrinkage=args.shrinkage,
                train_fraction=args.train_fraction,
                periods=args.periods,
                source=args.source_label,
            )
            print(f"Report written to {args.output}")
            print(
                f"Training: {metadata['train_observations']} returns; "
                f"holdout: {metadata['test_observations']} returns"
            )
    except (ValueError, OSError, OptimizationError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
