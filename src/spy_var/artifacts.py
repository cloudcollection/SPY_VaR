"""One registry for the preserved historical prediction artifacts."""
from pathlib import Path

import pandas as pd

from .config import MODEL_NAMES

C2_RELATIVE_PATH = Path("code/sections/section_05_mlp_quantile/outputs/external_section5_results/section5_model_c_garch_anchored_predictions.csv")


def archived_c2(root):
    frame = pd.read_csv(root / C2_RELATIVE_PATH)
    selected = frame.loc[frame.model_name.eq("Model C2: Conservative GARCH-anchored MLP-QR")].copy()
    if selected.empty:
        raise ValueError("The archived prediction file contains no C2 records")
    return selected


def load_archived_forecasts(root, include_c2=False):
    folder = root / "code/outputs/forecasts"
    files = {key: folder / name for key, name in {
        "weighted_hs": "var_historical_simulation.csv", "kde_hs": "var_kde_weighted_hs.csv",
        "garch_t": "var_garch_t.csv", "gjr_garch_t": "var_gjr_garch_t.csv", "mlp": "var_mlp_quantile.csv",
    }.items()}
    frames = {MODEL_NAMES[key]: pd.read_csv(path) for key, path in files.items()}
    if include_c2:
        frames[MODEL_NAMES["c2"]] = archived_c2(root)
        files["c2"] = root / C2_RELATIVE_PATH
    return frames, files
