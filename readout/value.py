"""
Valuation: a simple rNPV for one asset, and the probability the market is implying.

This is intentionally a small, transparent model, not a banker's spreadsheet.
Every input is an assumption that should be sourced or argued in the case write-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Asset:
    peak_sales: float            # annual net sales at peak, e.g. in $M
    years_to_launch: int         # years from today until first sales
    ramp_years: int = 5          # linear ramp from 0 to peak
    exclusivity_years: int = 10  # years of sales after launch before generic erosion
    operating_margin: float = 0.35  # after COGS, SG&A, royalties, before tax
    tax_rate: float = 0.21
    discount_rate: float = 0.12
    # Probabilities AFTER the readout being modelled:
    p_approval_given_success: float = 0.85
    # Costs ($M per year) that will be spent before launch, only if the readout succeeds:
    post_success_costs: list[float] = field(default_factory=list)


def _discount(t: np.ndarray, rate: float) -> np.ndarray:
    return 1 / (1 + rate) ** t


def npv_if_success(a: Asset) -> float:
    """NPV assuming the readout succeeds, still risk-adjusted for regulatory approval."""
    years = np.arange(a.years_to_launch, a.years_to_launch + a.exclusivity_years)
    since_launch = years - a.years_to_launch + 1
    revenue = a.peak_sales * np.minimum(since_launch / a.ramp_years, 1.0)
    cash = revenue * a.operating_margin * (1 - a.tax_rate)
    pv_sales = float(np.sum(cash * _discount(years + 0.5, a.discount_rate)))

    cost_years = np.arange(len(a.post_success_costs))
    pv_costs = float(np.sum(np.asarray(a.post_success_costs) * _discount(cost_years + 0.5, a.discount_rate)))
    return a.p_approval_given_success * pv_sales - pv_costs


def rnpv(a: Asset, p_success: float, value_if_failure: float = 0.0) -> float:
    """Risk-adjusted value of the asset given your probability that the readout succeeds."""
    _check_prob(p_success)
    return p_success * npv_if_success(a) + (1 - p_success) * value_if_failure


def implied_probability(market_value: float, value_if_success: float, value_if_failure: float) -> float:
    """
    The success probability that makes the market's valuation fair:
        market = p * V_success + (1 - p) * V_failure
    Typical use: market_value = enterprise value; value_if_failure = cash plus whatever
    the rest of the pipeline is worth; value_if_success = that plus the asset's NPV.

    Returned unclipped on purpose: a value outside [0, 1] means your success/failure
    values are inconsistent with the market, which is itself informative.
    """
    if value_if_success == value_if_failure:
        raise ValueError("success and failure values must differ")
    return (market_value - value_if_failure) / (value_if_success - value_if_failure)


def _check_prob(p: float) -> None:
    if not 0 <= p <= 1:
        raise ValueError("probability must be in [0, 1]")
