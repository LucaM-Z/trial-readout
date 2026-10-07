"""
Monte Carlo simulation of a two-arm, parallel-group trial with a continuous endpoint.

Sign convention used everywhere in this package:
    effect > 0 means the drug is BETTER than control.
    For a depression scale like MADRS (lower = better), effect = control change - drug change.
    Example: placebo improves 10 points, drug improves 16 points -> effect = 6.

Modelling choices (kept deliberately simple, all stated so they can be challenged):
    * Outcomes in each arm are normal with a common SD.
    * Dropout is missing completely at random; the analysis uses completers only.
      (Real trials use MMRM or ANCOVA, which usually gives a bit MORE power than this.)
    * Analysis is a two-sided pooled-variance t-test. "Success" means p < alpha AND the
      effect points in the drug's favour.

Speed trick: for normal data, the sample mean and sample variance of each arm are
independent with known distributions (normal and scaled chi-square). Drawing those
directly is mathematically identical to simulating every patient, but far faster.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _completers(n_per_arm: int, dropout: float) -> int:
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    n = int(round(n_per_arm * (1 - dropout)))
    if n < 2:
        raise ValueError("fewer than 2 completers per arm")
    return n


def simulate_trials(
    effect,
    sd: float,
    n_per_arm: int,
    dropout: float = 0.0,
    alpha: float = 0.05,
    n_sims: int = 20_000,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate n_sims trials and return a boolean array: True where the trial succeeded.

    `effect` can be a single number (fixed true effect) or an array of length n_sims
    (one true effect per simulated trial, e.g. draws from a prior).
    """
    rng = np.random.default_rng(seed)
    n = _completers(n_per_arm, dropout)
    effect = np.broadcast_to(np.asarray(effect, dtype=float), (n_sims,))

    mean_drug = rng.normal(effect, sd / np.sqrt(n))
    mean_ctrl = rng.normal(0.0, sd / np.sqrt(n), size=n_sims)
    var_drug = sd**2 * rng.chisquare(n - 1, size=n_sims) / (n - 1)
    var_ctrl = sd**2 * rng.chisquare(n - 1, size=n_sims) / (n - 1)

    pooled_var = (var_drug + var_ctrl) / 2
    se = np.sqrt(pooled_var * 2 / n)
    t_stat = (mean_drug - mean_ctrl) / se

    df = 2 * n - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    return t_stat > t_crit


def simulated_power(effect: float, sd: float, n_per_arm: int, **kwargs) -> float:
    """Probability of success at ONE fixed true effect (classic power)."""
    return float(simulate_trials(effect, sd, n_per_arm, **kwargs).mean())


def analytic_power(
    effect: float, sd: float, n_per_arm: int, dropout: float = 0.0, alpha: float = 0.05
) -> float:
    """
    Exact power of the same test from the noncentral t distribution.
    Used to validate the simulator: the two must agree within Monte Carlo error.
    """
    n = _completers(n_per_arm, dropout)
    df = 2 * n - 2
    ncp = effect / (sd * np.sqrt(2 / n))
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(t_crit, df, ncp))


def probability_of_success(
    effect_draws: np.ndarray,
    sd: float,
    n_per_arm: int,
    dropout: float = 0.0,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict:
    """
    Probability of success averaged over uncertainty in the true effect
    (sometimes called 'assurance'). One simulated trial per prior draw.

    Returns the estimate plus its Monte Carlo standard error, so you never report
    more precision than the simulation supports.
    """
    effect_draws = np.asarray(effect_draws, dtype=float)
    wins = simulate_trials(
        effect_draws, sd, n_per_arm, dropout=dropout, alpha=alpha,
        n_sims=effect_draws.size, seed=seed,
    )
    p = wins.mean()
    mc_se = np.sqrt(p * (1 - p) / wins.size)
    return {"pos": float(p), "mc_se": float(mc_se), "n_sims": int(wins.size)}


def power_curve(effects, sd: float, n_per_arm: int, **kwargs) -> list[tuple[float, float]]:
    """Power at several fixed effect sizes. Useful for the 'three scenarios' table."""
    return [(float(e), simulated_power(e, sd, n_per_arm, **kwargs)) for e in effects]
