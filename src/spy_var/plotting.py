from pathlib import Path

import matplotlib.pyplot as plt

from .config import LEVELS


def plot_failure_rates(table, path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, alpha in zip(axes, LEVELS):
        subset = table[table.alpha == alpha]
        ax.barh(subset.model, subset.failure_rate, color="#4C78A8")
        ax.axvline(alpha, color="#D62728", linestyle="--", linewidth=1.2)
        ax.set_title(f"{alpha:.0%} lower-tail quantile: observed violation rate")
        ax.grid(axis="x", alpha=0.25)
    axes[-1].set_xlabel("Violation rate (target shown by dashed line)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
