"""Compatibility entry point: evaluate cached predictions without training."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spy_var.cli import main

if __name__ == "__main__":
    main(["evaluate", "--output", str(ROOT.parent / "spy-var-runs/cached-check"), *sys.argv[1:]])
