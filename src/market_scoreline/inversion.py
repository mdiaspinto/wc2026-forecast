"""Phase 3b — invert the de-vigged market into goal rates (lam_home, lam_away).

The closing line gives us many de-vigged probabilities per match (1X2, every
totals line, every Asian-handicap line). They over-determine a 2-parameter goal
model, so we solve:

    minimize_{lam_h, lam_a}  Σ_m  w_m * ( logit p_model_m(lam_h,lam_a; rho)
                                          - logit p_market_m )^2

over log-lambdas (unconstrained). `rho` (Dixon-Coles low-score dependence) is NOT
per-match identifiable from these aggregates, so it is a GLOBAL constant supplied
by the caller (calibrated once in Phase 4). The fit residual (RMSE in probability
space) is carried through as a per-match confidence flag — a thin or arbitrage-y
market shows up as a large residual.

Constraints used (each a probability in (0,1)):
    * 1X2:        P(home win), P(draw), P(away win)
    * totals k:   P(over k)            for every captured line
    * handicap h: P(home covers h)     for every captured line (home handicap)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from market_scoreline import matrix as mx

_EPS = 1e-6


def _logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


# --------------------------------------------------------------------------- #
# Per-match constraint bundle
# --------------------------------------------------------------------------- #
@dataclass
class MatchConstraints:
    match_id: int
    home: str
    away: str
    p_1x2: tuple[float, float, float] | None = None       # (home, draw, away)
    totals: list[tuple[float, float]] = field(default_factory=list)      # (line, p_over)
    handicaps: list[tuple[float, float]] = field(default_factory=list)   # (home_handicap, p_home_cover)

    def n_constraints(self) -> int:
        return (3 if self.p_1x2 else 0) + len(self.totals) + len(self.handicaps)


def constraints_from_devigged(g: pd.DataFrame) -> MatchConstraints:
    """Build a MatchConstraints from one match's de-vigged long rows (devig_ok)."""
    g = g[g["devig_ok"]]
    home, away = g["home"].iloc[0], g["away"].iloc[0]
    mid = int(g["match_id"].iloc[0])
    mc = MatchConstraints(mid, home, away)

    ml = g[g["market_type"] == "moneyline"]
    if len(ml) == 3:
        d = {r.side: r.prob for r in ml.itertuples()}
        if {"home", "draw", "away"} <= d.keys():
            mc.p_1x2 = (d["home"], d["draw"], d["away"])

    for line, sub in g[g["market_type"] == "total"].groupby("line"):
        over = sub[sub.side == "over"]
        if len(over):
            mc.totals.append((float(line), float(over["prob"].iloc[0])))

    for line, sub in g[g["market_type"] == "spread"].groupby("line"):
        hc = sub[sub.side == "home"]
        if len(hc):
            mc.handicaps.append((float(line), float(hc["prob"].iloc[0])))
    return mc


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
@dataclass
class InversionResult:
    match_id: int
    home: str
    away: str
    lam_home: float
    lam_away: float
    rho: float
    residual: float          # RMSE in probability space (confidence flag)
    n_constraints: int
    success: bool

    def matrix(self) -> np.ndarray:
        return mx.score_matrix(self.lam_home, self.lam_away, self.rho)


def _model_vector(lam_h: float, lam_a: float, rho: float,
                  mc: MatchConstraints) -> np.ndarray:
    m = mx.score_matrix(lam_h, lam_a, rho)
    vals = []
    if mc.p_1x2:
        vals.extend(mx.wdl(m))
    for line, _ in mc.totals:
        vals.append(mx.prob_over(m, line))
    for h, _ in mc.handicaps:
        vals.append(mx.prob_home_cover(m, h))
    return np.asarray(vals)


def _market_vector(mc: MatchConstraints) -> np.ndarray:
    vals = []
    if mc.p_1x2:
        vals.extend(mc.p_1x2)
    vals.extend(p for _, p in mc.totals)
    vals.extend(p for _, p in mc.handicaps)
    return np.asarray(vals)


def _weights(mc: MatchConstraints, scheme: str) -> np.ndarray:
    n3 = 3 if mc.p_1x2 else 0
    w = np.ones(mc.n_constraints())
    if scheme == "uniform":
        return w
    if scheme == "info":
        # p(1-p) emphasis: near-50/50 lines carry the most information.
        mkt = _market_vector(mc)
        return np.clip(mkt * (1.0 - mkt), 1e-3, None)
    raise ValueError(f"unknown weight scheme {scheme!r}")


def _init_lambdas(mc: MatchConstraints) -> tuple[float, float]:
    """Warm start: total from the totals line closest to 50/50, split from 1X2."""
    total = 2.6
    if mc.totals:
        line, p_over = min(mc.totals, key=lambda t: abs(t[1] - 0.5))
        total = float(line)                  # P(over line)~0.5 => mean total ~ line
    ratio = 0.5
    if mc.p_1x2:
        ph, _, pa = mc.p_1x2
        ratio = float(np.clip(ph / (ph + pa + _EPS), 0.15, 0.85))
    lam_h = max(0.2, total * ratio)
    lam_a = max(0.2, total * (1.0 - ratio))
    return lam_h, lam_a


def invert(mc: MatchConstraints, rho: float = 0.0,
           weight_scheme: str = "uniform") -> InversionResult:
    market = _market_vector(mc)
    w = _weights(mc, weight_scheme)
    target = _logit(market)

    def obj(theta):
        lam_h, lam_a = np.exp(theta)
        model = _model_vector(lam_h, lam_a, rho, mc)
        r = _logit(model) - target
        return float(np.sum(w * r * r))

    lh0, la0 = _init_lambdas(mc)
    res = minimize(obj, x0=np.log([lh0, la0]), method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 4000})
    lam_h, lam_a = (float(v) for v in np.exp(res.x))

    model = _model_vector(lam_h, lam_a, rho, mc)
    residual = float(np.sqrt(np.mean((model - market) ** 2)))
    return InversionResult(mc.match_id, mc.home, mc.away, lam_h, lam_a, rho,
                           residual, mc.n_constraints(), bool(res.success))


def invert_long(dv: pd.DataFrame, rho: float = 0.0,
                weight_scheme: str = "uniform") -> list[InversionResult]:
    """Invert every match in a de-vigged long table (from devig.devig_long)."""
    out = []
    for _, g in dv.groupby("match_id"):
        mc = constraints_from_devigged(g)
        if mc.n_constraints() < 2:
            continue
        out.append(invert(mc, rho, weight_scheme))
    return out


# --------------------------------------------------------------------------- #
# CLI / validation against the synthetic fixture
# --------------------------------------------------------------------------- #
def _validate_on_fixture() -> None:
    from market_scoreline.devig import devig_long

    base = Path(__file__).resolve().parent / "tests" / "fixtures"
    snap = pd.read_csv(base / "synth_snapshot.csv")
    truth = pd.read_csv(base / "synth_truth.csv").set_index("match_id")

    dv = devig_long(snap, method="multiplicative")   # fixture vig is multiplicative
    results = invert_long(dv, rho=0.0)

    print("Inversion vs ground-truth lambdas (fixture, rho=0):")
    print(f"  {'match':<22}{'lam_h (fit/true)':>22}{'lam_a (fit/true)':>22}{'resid':>9}")
    max_err = 0.0
    for r in results:
        t = truth.loc[r.match_id]
        eh, ea = abs(r.lam_home - t.lam_home), abs(r.lam_away - t.lam_away)
        max_err = max(max_err, eh, ea)
        print(f"  {r.home+' v '+r.away:<22}"
              f"{r.lam_home:>9.3f}/{t.lam_home:<6.3f}    "
              f"{r.lam_away:>9.3f}/{t.lam_away:<6.3f}  {r.residual:>8.5f}")
    print(f"\nmax |lambda error| = {max_err:.5f}  ({len(results)} matches)")
    assert max_err < 5e-3, f"lambda recovery too loose: {max_err}"
    print("PASS — inversion recovers the generating lambdas.")


if __name__ == "__main__":
    _validate_on_fixture()
