# readout

Estimating the probability that a pivotal clinical trial succeeds, and what that means for the asset's value, using only public evidence. Every prediction is committed publicly before the result is known and scored afterward.

## Project goal

1. **Prior** (`readout/prior.py`): pools earlier trial results with a random-effects meta-analysis, widens to a predictive distribution for a new trial, and applies an explicit Phase 2 to Phase 3 shrinkage factor.
2. **Simulation** (`readout/simulate.py`): simulates the actual trial design (sample size, endpoint, dropout, alpha) thousands of times across that prior to get a probability of success.
3. **Valuation** (`readout/value.py`): a transparent rNPV, plus the success probability implied by the market's valuation.
4. **Ledger** (`predictions/`, `readout/calibration.py`): timestamped predictions scored with the Brier score once trials read out. Prospective and retrospective results are reported separately.

## Importance of the Prior

From `examples/demo.py` (made-up numbers): power at the pooled Phase 2 effect is 0.98, but after accounting for uncertainty in the true effect and typical Phase 2 to Phase 3 shrinkage, the probability of success is about 0.59. Most of the analysis lives in that gap.

## Validation

The simulator is checked against exact power from the noncentral t distribution, and the meta-analysis against `statsmodels`. Run:

```
pip install -r requirements.txt
pytest
```

## Cases

| Case | Kind | Status | Probability | Outcome |
|---|---|---|---|---|
| Helus Pharma APPROACH (HLP003, adjunctive MDD) | prospective | in progress | | |

## Limitations

- Normal outcomes, two arms, completers analysis. Real trials use MMRM or ANCOVA.
- The shrinkage factor and between-study variance are judgment calls; each case documents and sources them.
- The rNPV uses simplified commercial assumptions and is the least reliable part of the model.
- A handful of predictions cannot establish calibration. The ledger is a long-term record.

## Licence

MIT
