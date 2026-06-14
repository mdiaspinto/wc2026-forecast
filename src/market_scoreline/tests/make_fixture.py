"""Generate a self-consistent synthetic odds snapshot from KNOWN goal rates.

Builds a fetch.py-shaped long CSV whose prices are derived from a double-Poisson
score matrix at a chosen (lam_home, lam_away), then inflated by a per-leg vig.
This gives downstream phases ground truth:
    * devig (Phase 2) should recover the true market probabilities.
    * inversion (Phase 3) should recover the (lam_home, lam_away) used here.

Independent Poisson (rho = 0) is used here so the fixture needs no DC code; once
matrix.py exists we can regenerate with the low-score correction.

    python -m market_scoreline.tests.make_fixture
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from market_scoreline import matrix as mx

RHO = 0.0  # fixture generated with no DC correction -> inversion recovers lambdas exactly

# (home, away, lam_home, lam_away)
MATCHES = [
    ("Spain", "Morocco", 2.10, 0.65),   # heavy favorite
    ("Germany", "Croatia", 1.55, 1.10),  # moderate edge
    ("Uruguay", "Portugal", 1.05, 1.35),  # slight away edge
]

TOTAL_LINES = [1.5, 2.0, 2.5, 3.0, 3.5]
HANDICAP_LINES = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]  # home handicap
VIG_PER_LEG = 0.012  # ~ multiplicative margin added symmetrically per outcome


def _true_probs(lh: float, la: float) -> dict:
    """Fair (vig-free) market probabilities straight from the model's own forward
    functions, so the inversion is an exact inverse up to the vig + tolerance."""
    m = mx.score_matrix(lh, la, RHO)
    ph, pd_, pa = mx.wdl(m)
    out = {"moneyline": {"home": ph, "draw": pd_, "away": pa}, "total": {}, "spread": {}}
    for k in TOTAL_LINES:
        over = mx.prob_over(m, k)
        out["total"][k] = {"over": over, "under": 1.0 - over}
    for h in HANDICAP_LINES:
        cover = mx.prob_home_cover(m, h)
        out["spread"][h] = {"home": cover, "away": 1.0 - cover}
    return out


def _odds_from_prob(p: float) -> float:
    """True prob -> decimal odds with a symmetric per-leg vig."""
    return round(1.0 / (p * (1.0 + VIG_PER_LEG)), 4)


def build() -> pd.DataFrame:
    rows = []
    start = datetime(2026, 6, 11, 19, 0, tzinfo=timezone.utc).isoformat()
    for i, (home, away, lh, la) in enumerate(MATCHES):
        tp = _true_probs(lh, la)
        mid = 9_000_000 + i

        def add(mtype, line, side, prob):
            dec = _odds_from_prob(prob)
            rows.append({
                "match_id": mid, "league_id": 2686, "start_time_utc": start,
                "home": home, "away": away, "market_type": mtype,
                "period": 0, "line": line, "side": side,
                "american": np.nan, "decimal": dec,
                "implied_raw": round(1.0 / dec, 6),
                "true_prob": round(prob, 6),       # ground truth (extra column)
            })

        for side in ("home", "draw", "away"):
            add("moneyline", float("nan"), side, tp["moneyline"][side])
        for k, d in tp["total"].items():
            add("total", k, "over", d["over"])
            add("total", k, "under", d["under"])
        for h, d in tp["spread"].items():
            add("spread", h, "home", d["home"])
            add("spread", h, "away", d["away"])
    return pd.DataFrame(rows)


def main() -> None:
    df = build()
    out = Path(__file__).resolve().parent / "fixtures" / "synth_snapshot.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    # ground-truth lambdas alongside, for the inversion test
    truth = pd.DataFrame(
        [{"match_id": 9_000_000 + i, "home": h, "away": a, "lam_home": lh, "lam_away": la}
         for i, (h, a, lh, la) in enumerate(MATCHES)]
    )
    truth.to_csv(out.parent / "synth_truth.csv", index=False)
    print(f"Wrote {len(df)} legs for {len(MATCHES)} matches -> {out}")
    print(f"Ground-truth lambdas -> {out.parent / 'synth_truth.csv'}")


if __name__ == "__main__":
    main()
