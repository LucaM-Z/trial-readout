"""
Scoring predictions once trials read out.

Each prediction lives in predictions/ as a JSON file (see predictions/TEMPLATE.json).
Unresolved predictions have "outcome": null and are ignored by the scores.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def brier_score(probs, outcomes) -> float:
    """Mean squared error of probabilities. 0 is perfect, 0.25 is always saying 50%."""
    p = np.asarray(probs, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    if p.shape != o.shape or p.size == 0:
        raise ValueError("need equal, non-empty arrays")
    if np.any((p < 0) | (p > 1)) or not np.all(np.isin(o, [0, 1])):
        raise ValueError("probs must be in [0,1] and outcomes 0 or 1")
    return float(np.mean((p - o) ** 2))


def load_ledger(directory: str | Path = "predictions") -> list[dict]:
    rows = []
    for f in sorted(Path(directory).glob("*.json")):
        if f.name == "TEMPLATE.json":
            continue
        rec = json.loads(f.read_text())
        rec["_file"] = f.name
        rows.append(rec)
    return rows


def score_ledger(directory: str | Path = "predictions", kind: str | None = None) -> dict:
    """
    kind: None for everything, or "prospective" / "retrospective".
    Keep the two separate when reporting. Only prospective predictions show judgment.
    """
    rows = [r for r in load_ledger(directory) if r.get("outcome") is not None]
    if kind:
        rows = [r for r in rows if r.get("kind") == kind]
    if not rows:
        return {"n_resolved": 0, "brier": None}
    p = [r["probability"] for r in rows]
    o = [int(r["outcome"]) for r in rows]
    return {"n_resolved": len(rows), "brier": brier_score(p, o)}


def sha256_of(path: str | Path) -> str:
    """
    Hash of a prediction file. Git commit dates can be faked, so also post this hash
    somewhere with an independent timestamp (a public post, an email to yourself,
    or opentimestamps.org) before the readout.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
