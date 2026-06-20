from __future__ import annotations

import runpy
from pathlib import Path

from utils import ensure_directories


SCRIPT_DIR = Path(__file__).resolve().parent


PIPELINE = [
    "01_data_exploration.py",
    "02_historical_simulation.py",
    "04_kde_weighted_hs.py",
    "03_garch_var.py",
    "04_neural_quantile_model.py",
    "05_backtesting.py",
    "06_make_tables_figures.py",
]


def run_script(script_name: str) -> None:
    script_path = SCRIPT_DIR / script_name
    print(f"\n=== Running {script_name} ===")
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    except Exception as exc:
        raise RuntimeError(f"Pipeline step failed: {script_name}. Error: {exc}") from exc


def main() -> None:
    ensure_directories()
    for script_name in PIPELINE:
        run_script(script_name)
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
