"""Run from any working directory; default inputs are resolved from this checkout."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spy_var.cli import main

if __name__ == "__main__":
    main()
