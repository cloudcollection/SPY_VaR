import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spy_var.config import ExperimentConfig, load_config
from spy_var.data import load_data
from spy_var.pipeline import prepare_output

ROOT = Path(__file__).resolve().parents[1]


def test_repository_snapshot_is_chronological():
    data = load_data(ROOT / "code/data/spy_data.csv")
    assert len(data) == 4640
    assert str(data.index[0].date()) == "2000-01-04"
    assert str(data.index[-1].date()) == "2018-06-27"


def test_data_rejects_out_of_order_dates(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("date,log_ret,rv5,bv\n2020-01-02,0,1,1\n2020-01-01,0,1,1\n")
    with pytest.raises(ValueError, match="increasing"):
        load_data(path)


def test_unknown_config_is_not_silently_ignored(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"window_szie": 500}))
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)


@pytest.mark.parametrize("change", [{"window_size": 1}, {"decay": 0}, {"models": ["typo"]}, {"models": ["hs", "hs"]}])
def test_invalid_config(change):
    with pytest.raises(ValueError):
        ExperimentConfig(**change).validate()


def test_existing_results_cannot_be_overwritten(tmp_path):
    (tmp_path / "user-result.txt").write_text("preserve me")
    with pytest.raises(FileExistsError):
        prepare_output(tmp_path)
    assert (tmp_path / "user-result.txt").read_text() == "preserve me"


def test_features_do_not_use_future_information():
    from spy_var.anchored import BASE_FEATURES, RV_FEATURES, make_features
    original = load_data(ROOT / "code/data/spy_data.csv").iloc[:100].copy()
    perturbed = original.copy()
    perturbed.iloc[70:] *= 100
    a = make_features(original)
    b = make_features(perturbed)
    columns = BASE_FEATURES + RV_FEATURES
    pd.testing.assert_frame_equal(a.loc[:original.index[69], columns], b.loc[:original.index[69], columns])
    assert a.loc[original.index[40], "target"] == original.log_ret.iloc[41]
    assert a.loc[original.index[40], "target_date"] == original.index[41]


def test_historical_forecast_never_sees_the_target_day():
    from spy_var.classical import historical_forecast
    config = ExperimentConfig(window_size=60, models=["hs"]).validate()
    data = load_data(ROOT / "code/data/spy_data.csv").iloc[:80].copy()
    changed = data.copy()
    changed.iloc[60:, 0] = 999
    before = historical_forecast(data, config, "hs")
    after = historical_forecast(changed, config, "hs")
    assert np.allclose(before.loc[0, ["VaR_1", "VaR_5", "VaR_10"]].to_numpy(dtype=float),
                       after.loc[0, ["VaR_1", "VaR_5", "VaR_10"]].to_numpy(dtype=float))
