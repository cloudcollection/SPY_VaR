"""Compatibility entry point for the configuration-driven training pipeline."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from spy_var.cli import main

if __name__ == "__main__":
    main(["train", "--output", str(ROOT / "results/full-exploratory"), *sys.argv[1:]])
