"""
Turning earlier evidence into a distribution for the TRUE effect in the upcoming trial.

Three steps, each one a separate, inspectable judgment:
    1. Pool earlier trial results with a random-effects meta-analysis (DerSimonian-Laird).
    2. Widen to a PREDICTIVE distribution: the effect in a new trial varies around the
       pooled mean by the between-study SD (tau), on top of uncertainty in the mean itself.
    3. Apply a 'haircut' for Phase 2 to Phase 3 shrinkage (winner's curse, smaller and
       more selected early samples, unblinding, site effects). This is a judgment call and
       must be justified with sources in the case write-up.

Sign convention: effect > 0 means the drug is better (see simulate.py).
"""

from __future__ import annotations

import numpy as np


def dersimonian_laird(effects, ses) -> dict:
    """
    Random-effects meta-analysis.

    effects: per-study effect estimates (same sign convention, same scale).
    ses:     their standard errors.

    Returns pooled mean, its SE, tau^2 (between-study variance), I^2, and k.
    With k = 1, tau^2 cannot be estimated and is returned as 0. In that case you
    should supply tau yourself when building the predictive distribution.
    """
    y = np.asarray(effects, dtype=float)
    se = np.asarray(ses, dtype=float)
    if y.shape != se.shape or y.size == 0:
        raise ValueError("effects and ses must be non-empty and the same length")
    if np.any(se <= 0):
        raise ValueError("standard errors must be positive")

    k = y.size
    w = 1 / se**2
    mu_fixed = np.sum(w * y) / np.sum(w)
    q = float(np.sum(w * (y - mu_fixed) ** 2))

    if k > 1:
        c = np.sum(w) - np.sum(w**2) / np.sum(w)
        tau2 = max(0.0, (q - (k - 1)) / c)
        i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    else:
        tau2, i2 = 0.0, 0.0

    w_re = 1 / (se**2 + tau2)
    mu = float(np.sum(w_re * y) / np.sum(w_re))
    se_mu = float(np.sqrt(1 / np.sum(w_re)))
    return {"mu": mu, "se_mu": se_mu, "tau2": float(tau2), "i2": float(i2), "q": q, "k": k}


def predictive_draws(
    mu: float, se_mu: float, tau: float, n: int = 50_000, seed: int | None = None
) -> np.ndarray:
    """Draws of the true effect in a NEW trial: Normal(mu, sqrt(se_mu^2 + tau^2))."""
    rng = np.random.default_rng(seed)
    return rng.normal(mu, np.sqrt(se_mu**2 + tau**2), size=n)


def apply_haircut(
    draws: np.ndarray,
    shrink_mean: float,
    shrink_sd: float,
    seed: int | None = None,
    bounds: tuple[float, float] = (0.0, 1.5),
) -> np.ndarray:
    """
    Multiply each draw by an uncertain retention factor.

    shrink_mean = 0.6 means 'on average, Phase 3 retains 60% of the Phase 2 effect'.
    shrink_sd expresses how unsure you are about that. The factor is clipped to
    `bounds` so it can't flip the sign or explode.
    """
    rng = np.random.default_rng(seed)
    factor = np.clip(rng.normal(shrink_mean, shrink_sd, size=np.size(draws)), *bounds)
    return np.asarray(draws) * factor


def summarize(draws: np.ndarray) -> dict:
    d = np.asarray(draws)
    lo, med, hi = np.percentile(d, [5, 50, 95])
    return {
        "mean": float(d.mean()),
        "median": float(med),
        "p05": float(lo),
        "p95": float(hi),
        "p_effect_positive": float((d > 0).mean()),
    }
