import numpy as np
from statsmodels.stats.meta_analysis import combine_effects

from readout.prior import apply_haircut, dersimonian_laird, predictive_draws


def test_dl_matches_statsmodels():
    y = np.array([4.1, 7.5, 2.8, 5.9])
    se = np.array([1.5, 2.2, 1.1, 1.8])
    ours = dersimonian_laird(y, se)
    ref = combine_effects(y, se**2, method_re="dl")
    assert np.isclose(ours["tau2"], ref.tau2)
    assert np.isclose(ours["mu"], ref.mean_effect_re)
    assert np.isclose(ours["se_mu"], ref.sd_eff_w_re)


def test_homogeneous_studies_give_zero_tau():
    out = dersimonian_laird([3.0, 3.0, 3.0], [1.0, 1.0, 1.0])
    assert out["tau2"] == 0 and np.isclose(out["mu"], 3.0)


def test_single_study():
    out = dersimonian_laird([5.0], [2.0])
    assert out["k"] == 1 and out["tau2"] == 0 and np.isclose(out["se_mu"], 2.0)


def test_predictive_spread():
    d = predictive_draws(5.0, 1.0, 2.0, n=200_000, seed=1)
    assert abs(d.std() - np.sqrt(5)) < 0.02


def test_haircut_shrinks_on_average():
    d = np.full(100_000, 10.0)
    h = apply_haircut(d, 0.6, 0.15, seed=2)
    assert abs(h.mean() - 6.0) < 0.05 and h.min() >= 0
