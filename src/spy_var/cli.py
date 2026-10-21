import argparse
from pathlib import Path

from .config import load_config
from .pipeline import evaluate_cached, run_experiment, run_seeds


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Reproduce SPY VaR experiments or evaluate archived predictions")
    parser.add_argument("--root", type=Path, default=root, help="Repository root (required if installed outside the checkout)")
    subs = parser.add_subparsers(dest="command", required=True)
    cached = subs.add_parser("evaluate", help="Evaluate saved predictions; does not train models")
    cached.add_argument("--include-c2", action="store_true", help="Add archived exploratory C2; restrict comparison to its dates")
    cached.add_argument("--output", type=Path, required=True)
    run = subs.add_parser("train", help="Generate forecasts from the data snapshot and evaluate them")
    run.add_argument("--config", type=Path)
    run.add_argument("--data", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--seeds", type=int, nargs="+", help="Run one independent experiment per seed and summarize variability")
    run.add_argument("--max-observations", type=int, help="Use a chronological prefix for smoke tests only")
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        evaluate_cached(args.root, args.output, args.include_c2)
    else:
        config = load_config(args.config or args.root / "configs/full_exploratory.json")
        data_path = args.data or args.root / "code/data/spy_data.csv"
        if args.seeds:
            run_seeds(args.root, data_path, config, args.output, args.seeds, args.max_observations)
        else:
            run_experiment(args.root, data_path, config, args.output, args.max_observations)
