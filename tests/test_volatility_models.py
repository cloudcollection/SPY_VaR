from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from spy_var.classical import garch_forecast
from spy_var.config import ExperimentConfig
from spy_var.data import load_data

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("convergence_flag", [0, 1])
def test_garch_timing_scaling_and_failed_optimizer(monkeypatch, convergence_flag):
    import arch

    data = load_data(ROOT / "code/data/spy_data.csv").iloc[:61].copy()
    # The last row is the target return and must never enter the fit.
    data.iloc[60, 0] = 999
    received = []

    def fake_arch(window, **kwargs):
        received.append(window.copy())
        result = SimpleNamespace(
            convergence_flag=convergence_flag,
            params={"nu": 8.0},
            forecast=lambda **kwargs: SimpleNamespace(mean=pd.DataFrame([[0.0]]), variance=pd.DataFrame([[4.0]])),
        )
        return SimpleNamespace(fit=lambda **kwargs: result)

    monkeypatch.setattr(arch, "arch_model", fake_arch)
    result = garch_forecast(data, ExperimentConfig(window_size=60, models=["garch_t"]))
    assert len(received[0]) == 60
    pd.testing.assert_series_equal(received[0], data.log_ret.iloc[:60] * 100)
    if convergence_flag:
        assert np.isnan(result.VaR_1.iloc[0])
        assert "convergence flag 1" in result.fit_error.iloc[0]
    else:
        assert result.sigma_next.iloc[0] == pytest.approx(0.02)
        assert result.VaR_1.iloc[0] == pytest.approx(0.02 * t.ppf(0.01, 8) * np.sqrt(6 / 8))
