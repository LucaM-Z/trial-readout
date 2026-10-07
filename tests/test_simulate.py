"""The simulator must reproduce exact (noncentral t) power. This is the core validity check."""
import numpy as np
import pytest

from readout.simulate import analytic_power, probability_of_success, simulated_power

N_SIMS = 200_000


@pytest.mark.parametrize(
    "effect, sd, n, dropout",
    [
        (0.0, 10.0, 110, 0.0),   # null: favourable-direction rejections ~2.5%
        (3.0, 10.0, 110, 0.15),
        (4.0, 9.0, 110, 0.10),
        (6.0, 10.0, 60, 0.0),
        (2.0, 8.0, 300, 0.20),
    ],
)
def test_simulator_matches_exact_power(effect, sd, n, dropout):
    exact = analytic_power(effect, sd, n, dropout=dropout)
    sim = simulated_power(effect, sd, n, dropout=dropout, n_sims=N_SIMS, seed=1)
    mc_se = np.sqrt(exact * (1 - exact) / N_SIMS)
    assert abs(sim - exact) < 4 * mc_se + 1e-4


def test_type_one_error_is_half_alpha_one_sided():
    sim = simulated_power(0.0, 10.0, 100, n_sims=N_SIMS, seed=2)
    assert abs(sim - 0.025) < 0.002


def test_power_increases_with_effect_and_sample_size():
    assert analytic_power(2, 10, 100) < analytic_power(4, 10, 100)
    assert analytic_power(3, 10, 50) < analytic_power(3, 10, 200)


def test_pos_with_point_prior_equals_power():
    draws = np.full(N_SIMS, 4.0)
    out = probability_of_success(draws, 10.0, 110, seed=3)
    assert abs(out["pos"] - analytic_power(4.0, 10.0, 110)) < 4 * out["mc_se"] + 1e-4


def test_uncertainty_pulls_high_power_down():
    """A wide prior around a strong effect gives lower PoS than power at that effect."""
    rng = np.random.default_rng(4)
    draws = rng.normal(6.0, 4.0, N_SIMS)
    pos = probability_of_success(draws, 10.0, 110, seed=5)["pos"]
    assert pos < analytic_power(6.0, 10.0, 110)
