import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from spy_var.backtesting import christoffersen_independence, evaluate, kupiec_test, pinball_loss


def forecast(n=8):
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n),
                         "realized_log_ret": np.linspace(-0.04, 0.02, n),
                         "VaR_1": -0.03, "VaR_5": -0.02, "VaR_10": -0.01})


def test_pinball_hand_calculation():
    assert pinball_loss([-2, 2], [0, 0], 0.1) == pytest.approx(1.0)


@pytest.mark.parametrize("value,alpha", [(0, 0.05), (1, 0.05)])
def test_kupiec_extreme_sequences(value, alpha):
    lr, p = kupiec_test(np.repeat(value, 10), alpha)
    expected = -20 * np.log(alpha if value else 1 - alpha)
    assert lr == pytest.approx(expected)
    assert p == pytest.approx(chi2.sf(expected, 1))


def test_empty_test_and_unidentified_independence():
    assert np.isnan(kupiec_test([], 0.01)[1])
    assert np.isnan(christoffersen_independence([0] * 10)[1])
    assert np.isnan(christoffersen_independence([1] * 10)[1])


def test_independence_does_not_bridge_gaps():
    sequence = [0, 1, 1, 0, 0, 1]
    assert np.isnan(christoffersen_independence(sequence, [0, 2, 4, 6, 8, 10])[1])
    assert np.isfinite(christoffersen_independence(sequence)[1])


def test_same_finite_dates_for_every_model():
    a, b = forecast(), forecast()
    b.loc[3, "VaR_1"] = np.nan
    table, dates = evaluate({"a": a, "b": b}, calendar=pd.DatetimeIndex(a.date))
    tail = table[table.alpha == 0.01]
    assert tail.n_obs.tolist() == [7, 7]
    assert tail.n_independence_transitions.tolist() == [5, 5]
    assert not dates.loc[dates.alpha == 0.01, "date"].eq(a.date.iloc[3]).any()
    assert table.loc[table.alpha == 0.05, "n_obs"].tolist() == [8, 8]


def test_duplicate_dates_fail():
    a = forecast()
    with pytest.raises(ValueError, match="unique"):
        evaluate({"a": pd.concat([a, a.iloc[:1]])})


def test_conflicting_realized_returns_fail():
    a, b = forecast(), forecast()
    b.loc[2, "realized_log_ret"] += 0.01
    with pytest.raises(ValueError, match="disagree"):
        evaluate({"a": a, "b": b})


def test_no_common_dates_or_finite_values_fail():
    a, b = forecast(), forecast()
    b.date += pd.Timedelta(days=100)
    with pytest.raises(ValueError, match="no common"):
        evaluate({"a": a, "b": b})
    a.VaR_1 = np.inf
    with pytest.raises(ValueError, match="finite"):
        evaluate({"a": a})


def test_conditional_coverage_is_reported():
    table, _ = evaluate({"a": forecast()})
    assert "christoffersen_cc_pvalue" in table


def test_tail_probability_validation():
    with pytest.raises(ValueError):
        kupiec_test([0, 1], 0)
