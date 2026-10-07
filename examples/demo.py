"""
End-to-end demo with MADE-UP numbers. Not a real drug, not a prediction.
Shows the full pipeline: evidence -> prior -> probability of success -> value -> implied probability.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from readout.prior import apply_haircut, dersimonian_laird, predictive_draws, summarize
from readout.simulate import analytic_power, probability_of_success
from readout.value import Asset, implied_probability, npv_if_success, rnpv

# 1. Hypothetical Phase 2 evidence (benefit in scale points, SE)
ma = dersimonian_laird([7.0, 5.0], [2.5, 2.0])
draws = predictive_draws(ma["mu"], ma["se_mu"], tau=2.0, n=200_000, seed=1)
draws = apply_haircut(draws, shrink_mean=0.6, shrink_sd=0.15, seed=2)
print("Prior on true effect:", {k: round(v, 2) for k, v in summarize(draws).items()})

# 2. Probability of success for a 110-per-arm trial
sd, n = 10.0, 110
pos = probability_of_success(draws, sd, n, dropout=0.15, seed=3)
naive = analytic_power(ma["mu"], sd, n, dropout=0.15)
print(f"Naive power at pooled Phase 2 effect: {naive:.2f}")
print(f"Probability of success with uncertainty and haircut: {pos['pos']:.2f}")

# 3. Value and the market's implied probability
asset = Asset(peak_sales=800, years_to_launch=3, post_success_costs=[80, 60])
v_success_asset = npv_if_success(asset)
cash_and_rest = 200.0
market_ev = 450.0
p_market = implied_probability(market_ev, cash_and_rest + v_success_asset, cash_and_rest)
print(f"Asset NPV if success: ${v_success_asset:,.0f}M")
print(f"Our rNPV: ${rnpv(asset, pos['pos']):,.0f}M")
print(f"Market-implied probability: {p_market:.2f}  vs ours: {pos['pos']:.2f}")
