"""Run the APPROACH case: evidence -> prior -> probability of success."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))

from readout.prior import apply_haircut, dersimonian_laird, predictive_draws, summarize  # noqa: E402
from readout.simulate import power_curve, probability_of_success  # noqa: E402


def main():
    cfg = json.loads((HERE / "inputs.json").read_text())
    missing = [k for k, v in cfg.items() if v is None and not k.startswith("_")]
    with open(HERE / "evidence.csv") as f:
        rows = list(csv.DictReader(f))
    if missing or not rows:
        sys.exit(f"Not ready. Missing inputs: {missing or 'none'}. Evidence rows: {len(rows)}.")

    y = np.array([float(r["effect_madrs_benefit"]) for r in rows])
    se = np.array([float(r["se"]) for r in rows])
    ma = dersimonian_laird(y, se)
    tau = np.sqrt(ma["tau2"]) if ma["k"] >= 3 else cfg["tau_if_few_studies"]

    draws = predictive_draws(ma["mu"], ma["se_mu"], tau, n=200_000, seed=cfg["seed"])
    draws = apply_haircut(draws, cfg["haircut_mean"], cfg["haircut_sd"], seed=cfg["seed"] + 1)

    pos = probability_of_success(
        draws, cfg["sd_madrs_change"], cfg["n_per_arm"],
        dropout=cfg["dropout"], alpha=cfg["alpha"], seed=cfg["seed"] + 2,
    )
    print("Meta-analysis:", {k: round(v, 3) if isinstance(v, float) else v for k, v in ma.items()})
    print("Prior on true effect (MADRS points):", {k: round(v, 2) for k, v in summarize(draws).items()})
    print(f"Probability of success: {pos['pos']:.3f} (MC SE {pos['mc_se']:.3f})")
    print("Power at fixed effects:")
    for e, p in power_curve([2, 3, 4, 5, 6], cfg["sd_madrs_change"], cfg["n_per_arm"],
                            dropout=cfg["dropout"], seed=cfg["seed"]):
        print(f"  effect {e:.0f}: {p:.2f}")


if __name__ == "__main__":
    main()
