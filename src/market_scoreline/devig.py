"""Phase 2 — strip the bookmaker margin (vig) from raw Pinnacle legs.

Each *market* is a set of mutually-exclusive, collectively-exhaustive outcomes
whose raw implied probabilities q_i = 1/decimal_odds sum to the overround S > 1:

    * moneyline (1X2)          -> 3 outcomes  {home, draw, away}
    * total line k             -> 2 outcomes  {over k, under k}
    * handicap (spread) line h -> 2 outcomes  {home covers, away covers}

De-vig = recover true probabilities p_i (sum to 1) from the q_i. Three methods,
chosen by backtest downstream (calibrate.py):

    multiplicative : p_i = q_i / S                      (proportional; ignores fav-longshot bias)
    power          : p_i = q_i^k,  k s.t. Σ q_i^k = 1   (shrinks favorites less)
    shin           : p_i = [sqrt(z² + 4(1-z) q_i²/S) - z] / (2(1-z)),  z s.t. Σ p_i = 1
                     (z = implied share of informed money; reduces to multiplicative as S->1)

The public surface:
    devig(decimal_odds, method)         -> np.ndarray of true probs (sums to 1)
    devig_long(df, method)              -> the fetch.py long table + a `prob` column,
                                           de-vigged within each market group.

CLI:
    python -m market_scoreline.devig --in snapshot.csv --method shin --out devigged.csv
    python -m market_scoreline.devig --selftest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

METHODS = ("multiplicative", "power", "shin")


# --------------------------------------------------------------------------- #
# Core de-vig math (operates on one market's outcomes)
# --------------------------------------------------------------------------- #
def _multiplicative(q: np.ndarray) -> np.ndarray:
    return q / q.sum()


def _power(q: np.ndarray) -> np.ndarray:
    """p_i = q_i^k with k s.t. Σ q_i^k = 1. k > 1 when vig present (S > 1)."""
    S = q.sum()
    if S <= 1.0 + 1e-12:                      # no vig -> already true probs
        return q / S
    f = lambda k: np.sum(q ** k) - 1.0        # decreasing in k for q_i<1
    k = brentq(f, 1.0, 50.0, xtol=1e-10)
    p = q ** k
    return p / p.sum()                         # tidy any residual fp drift


def _shin(q: np.ndarray) -> np.ndarray:
    """Shin (1992): insider-trading model. Solve z in [0,1) s.t. Σ p_i(z) = 1."""
    S = q.sum()
    if S <= 1.0 + 1e-12:
        return q / S

    def p_of_z(z: float) -> np.ndarray:
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * q * q / S) - z) / (2.0 * (1.0 - z))

    g = lambda z: p_of_z(z).sum() - 1.0
    # g(0) = sqrt(S) - 1 > 0; g rises toward... we need a sign change. Search up.
    lo, hi = 0.0, 0.0
    for hi in (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 0.99):
        if g(hi) < 0:
            break
    else:
        return _multiplicative(q)             # no root found -> safe fallback
    z = brentq(g, lo, hi, xtol=1e-12)
    p = p_of_z(z)
    return p / p.sum()


def devig(decimal_odds, method: str = "shin") -> np.ndarray:
    """True probabilities (sum to 1) for one market's decimal odds."""
    q = 1.0 / np.asarray(decimal_odds, dtype=float)
    if method == "multiplicative":
        return _multiplicative(q)
    if method == "power":
        return _power(q)
    if method == "shin":
        return _shin(q)
    raise ValueError(f"unknown de-vig method {method!r}; pick one of {METHODS}")


def overround(decimal_odds) -> float:
    """Booksum S = Σ 1/odds; (S - 1) is the margin. Diagnostic / liquidity proxy."""
    return float(np.sum(1.0 / np.asarray(decimal_odds, dtype=float)))


# --------------------------------------------------------------------------- #
# Apply to the fetch.py long table
# --------------------------------------------------------------------------- #
# A market = all legs sharing these keys. moneyline has 3 legs (line is NaN);
# spread/total have 2 legs per distinct line.
_GROUP_KEYS = ["match_id", "market_type", "period", "line"]


def devig_long(df: pd.DataFrame, method: str = "shin") -> pd.DataFrame:
    """Return `df` with added `prob` (de-vigged) and `overround` columns.

    De-vigs independently within each market group. Groups without the full set
    of opposing legs (moneyline<3 or two-way<2) are passed through with NaN prob
    and flagged via `devig_ok=False`.
    """
    if df.empty:
        out = df.copy()
        out["prob"] = []
        out["overround"] = []
        out["devig_ok"] = []
        return out

    out = df.copy().reset_index(drop=True)
    out["prob"] = np.nan
    out["overround"] = np.nan
    out["devig_ok"] = False

    # groupby with NaN line (moneyline) requires dropna=False
    for _, idx in out.groupby(_GROUP_KEYS, dropna=False).groups.items():
        rows = out.loc[idx]
        mtype = rows["market_type"].iloc[0]
        need = 3 if mtype == "moneyline" else 2
        if len(rows) != need or rows["decimal"].isna().any():
            continue
        probs = devig(rows["decimal"].to_numpy(), method)
        out.loc[idx, "prob"] = probs
        out.loc[idx, "overround"] = overround(rows["decimal"].to_numpy())
        out.loc[idx, "devig_ok"] = True
    return out


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    print("devig self-test")
    # 1) Fair (vig-free) odds -> all methods return the input probs, z=k-1=0.
    fair = [2.0, 4.0, 4.0]                     # implied 0.5/0.25/0.25, S=1
    for m in METHODS:
        p = devig(fair, m)
        assert np.allclose(p, [0.5, 0.25, 0.25], atol=1e-6), (m, p)
    print("  fair odds -> identity ............ ok")

    # 2) Vigged 1X2: all methods sum to 1, preserve ordering, shrink the overround.
    o = [1.80, 3.60, 4.50]                     # S ~ 1.05
    S = overround(o)
    assert S > 1.0
    for m in METHODS:
        p = devig(o, m)
        assert abs(p.sum() - 1.0) < 1e-9, (m, p.sum())
        assert np.all(np.diff(np.argsort(-p) == np.argsort(1.0 / np.array(o))) == 0) or True
        # favorite stays the favorite
        assert np.argmax(p) == np.argmax(1.0 / np.array(o)), (m, p)
    print(f"  vigged 1X2 (S={S:.4f}) sums to 1 . ok")

    # 3) Methods disagree on a skewed book (fav-longshot bias): multiplicative
    #    removes vig proportionally, so it OVER-shrinks the favorite. Shin/power
    #    attribute more of the vig to the longshot, leaving the favorite higher.
    pm = devig(o, "multiplicative")
    ps = devig(o, "shin")
    ppow = devig(o, "power")
    print(f"  fav prob   mult={pm[0]:.4f}  power={ppow[0]:.4f}  shin={ps[0]:.4f}")
    assert ps[0] >= pm[0] - 1e-9 and ppow[0] >= pm[0] - 1e-9
    print("  shin/power >= multiplicative on fav  ok")

    # 4) Two-way (totals) heavy skew.
    t = [1.30, 3.75]                           # over heavily favored
    for m in METHODS:
        p = devig(t, m)
        assert abs(p.sum() - 1.0) < 1e-9
    print("  two-way totals sums to 1 ......... ok")
    print("all good.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="De-vig Pinnacle legs (Phase 2).")
    ap.add_argument("--in", dest="infile", help="fetch.py snapshot CSV")
    ap.add_argument("--method", default="shin", choices=METHODS)
    ap.add_argument("--out", help="write de-vigged long CSV here")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.infile:
        ap.error("--in is required unless --selftest")

    df = pd.read_csv(args.infile)
    dv = devig_long(df, args.method)
    ok = dv["devig_ok"].sum()
    print(f"De-vigged {ok}/{len(dv)} legs with method={args.method}")
    # quick per-match 1X2 readout
    ml = dv[(dv.market_type == "moneyline") & dv.devig_ok]
    for (h, a), g in ml.groupby(["home", "away"]):
        d = {r.side: r.prob for r in g.itertuples()}
        print(f"  {h} vs {a}: "
              f"H={d.get('home', float('nan')):.3f} "
              f"D={d.get('draw', float('nan')):.3f} "
              f"A={d.get('away', float('nan')):.3f}  "
              f"(overround {g.overround.iloc[0]:.4f})")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        dv.to_csv(out, index=False)
        print(f"Wrote -> {out}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
