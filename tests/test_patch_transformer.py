from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from spy_var.config import ExperimentConfig, PatchConfig
from spy_var.data import load_data
from spy_var.models.patch_transformer import PatchQuantileTransformer, supervised_window

ROOT = Path(__file__).resolve().parents[1]


def test_patch_count_quantile_order_and_positive_inverse_scale():
    config = PatchConfig()
    model = PatchQuantileTransformer(config).eval()
    data = load_data(ROOT / "code/data/spy_data.csv")
    history = torch.tensor(data.log_ret.iloc[:64].to_numpy(), dtype=torch.float32).view(1, -1)
    with torch.no_grad():
        predicted = model(history)
    assert model.n_patches == 7
    assert predicted.shape == (1, 3)
    assert torch.isfinite(predicted).all()
    assert torch.all(predicted[:, 1:] >= predicted[:, :-1])
    # Zero standardized head outputs must invert to the observed-window mean.
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        output = model(history)
    assert torch.allclose(output, history.mean().expand(1, 3))


def test_supervised_targets_are_strictly_after_their_input_and_before_refit():
    values = load_data(ROOT / "code/data/spy_data.csv").log_ret.to_numpy()
    x, y, targets = supervised_window(values, 0, 80, 16)
    assert targets.min() == 16 and targets.max() == 79
    assert np.allclose(x[-1], values[63:79])
    assert np.allclose(y, values[targets])
    changed = values.copy()
    changed[80:] += 10
    other_x, other_y, _ = supervised_window(changed, 0, 80, 16)
    assert np.array_equal(x, other_x) and np.array_equal(y, other_y)


@pytest.mark.parametrize("patch", [PatchConfig(heads=3), PatchConfig(lookback=1000), PatchConfig(stride=0)])
def test_invalid_patch_configuration(patch):
    with pytest.raises(ValueError):
        ExperimentConfig(models=["patchtst_qr"], patch=patch).validate()


def test_compatibility_alias_preserves_mutable_configuration():
    from spy_var import anchored, neural
    from spy_var.models import garch_anchored, mlp
    assert anchored is garch_anchored
    assert neural is mlp
