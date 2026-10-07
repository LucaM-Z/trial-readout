import json

import numpy as np
import pytest

from readout.calibration import brier_score, score_ledger
from readout.value import Asset, implied_probability, npv_if_success, rnpv


def test_implied_probability_round_trip():
    v_s, v_f, p = 900.0, 150.0, 0.4
    market = p * v_s + (1 - p) * v_f
    assert np.isclose(implied_probability(market, v_s, v_f), p)


def test_rnpv_is_linear_in_p():
    a = Asset(peak_sales=500, years_to_launch=3, post_success_costs=[60, 40])
    v = npv_if_success(a)
    assert np.isclose(rnpv(a, 0.5, 20), 0.5 * v + 0.5 * 20)
    assert v > 0


def test_more_discounting_lowers_value():
    lo = npv_if_success(Asset(peak_sales=500, years_to_launch=3, discount_rate=0.15))
    hi = npv_if_success(Asset(peak_sales=500, years_to_launch=3, discount_rate=0.08))
    assert lo < hi


def test_brier():
    assert brier_score([1, 0], [1, 0]) == 0
    assert np.isclose(brier_score([0.5, 0.5], [1, 0]), 0.25)
    with pytest.raises(ValueError):
        brier_score([1.2], [1])


def test_ledger_ignores_unresolved(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"probability": 0.7, "outcome": 1, "kind": "prospective"}))
    (tmp_path / "b.json").write_text(json.dumps({"probability": 0.3, "outcome": None, "kind": "prospective"}))
    out = score_ledger(tmp_path)
    assert out["n_resolved"] == 1 and np.isclose(out["brier"], 0.09)
