# Prediction ledger

Rules:
1. One JSON file per prediction, copied from TEMPLATE.json. Commit BEFORE the readout.
2. Never edit a committed prediction. If you update your view, add a new file (v1, v2) and keep the old one.
3. Define the event precisely before committing (usually: primary endpoint met at the prespecified alpha).
4. After committing, run `python -c "from readout.calibration import sha256_of; print(sha256_of('predictions/FILE.json'))"`
   and post the hash somewhere with an independent timestamp. Git dates alone can be faked.
5. When the trial reads out, fill in outcome (1 or 0), resolved_on, and resolution_source in a NEW commit.
6. Retrospective backtests use "kind": "retrospective" and are scored separately.
