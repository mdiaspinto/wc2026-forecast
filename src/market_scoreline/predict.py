"""Phase 5 — the product: most-likely scoreline for every match on a date.

End-to-end:  fetch (or load snapshot) -> devig -> invert -> matrix -> read-out.
Uses the de-vig method and global rho calibrated in Phase 4 (market_model.json).

Bets are placed ~45 min before kickoff, so in production this runs against a
T-45 snapshot (Phase 6 cron). For reproducibility it can also read a saved
snapshot CSV instead of hitting the live API.

    python -m market_scoreline.predict --date 2026-06-11 --league 2686
    python -m market_scoreline.predict --snapshot data/odds/snapshots/wc_2026-06-11.csv
    python -m market_scoreline.predict --date 2026-06-11 --out preds.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from market_scoreline import fetch, matrix as mx
from market_scoreline.devig import devig_long
from market_scoreline.inversion import constraints_from_devigged, invert

ROOT = Path(__file__).resolve().parents[2]
PARAM_PATH = ROOT / "data" / "processed" / "market_model.json"

# Fallback if Phase 4 hasn't been run yet. These mirror the locked calibration
# (1X2 + totals, 1500 matches): shin de-vig, mild negative rho. The method
# surface is flat, so rho carries the signal.
_DEFAULTS = {"devig_method": "shin", "rho": -0.06}


def load_model(path: Path = PARAM_PATH) -> dict:
    if path.exists():
        m = json.loads(path.read_text())
        return {"devig_method": m.get("devig_method", _DEFAULTS["devig_method"]),
                "rho": float(m.get("rho", _DEFAULTS["rho"]))}
    print(f"  (no {path.name}; using defaults {_DEFAULTS} — run Phase 4 to calibrate)")
    return dict(_DEFAULTS)


def predict_snapshot(df: pd.DataFrame, model: dict, top_n: int = 4) -> pd.DataFrame:
    """Tidy per-match predictions from a fetch.py long table."""
    dv = devig_long(df, model["devig_method"])
    rows = []
    for _, g in dv.groupby("match_id"):
        mc = constraints_from_devigged(g)
        if mc.n_constraints() < 2:
            continue
        r = invert(mc, model["rho"])
        m = r.matrix()
        i, j, p = mx.argmax_scoreline(m)
        ph, pd_, pa = mx.wdl(m)
        eh, ea = mx.expected_goals(m)
        tops = mx.top_scorelines(m, top_n)
        rows.append({
            "match_id": mc.match_id, "home": mc.home, "away": mc.away,
            "scoreline": f"{i}-{j}", "p_scoreline": round(p, 4),
            "p_home": round(ph, 4), "p_draw": round(pd_, 4), "p_away": round(pa, 4),
            "xg_home": round(eh, 2), "xg_away": round(ea, 2),
            "lam_home": round(r.lam_home, 3), "lam_away": round(r.lam_away, 3),
            "top_scorelines": "; ".join(f"{a}-{b} {q*100:.0f}%" for a, b, q in tops),
            "n_constraints": r.n_constraints,
            "residual": round(r.residual, 4),       # confidence flag (lower = better)
        })
    return pd.DataFrame(rows)


def _print_table(preds: pd.DataFrame, model: dict) -> None:
    if preds.empty:
        print("No predictions (no markets captured).")
        return
    print(f"\nMost-likely scorelines  (devig={model['devig_method']}, rho={model['rho']})")
    print("-" * 92)
    print(f"{'match':<34}{'score':>7}{'P':>7}{'  1X2 (H/D/A)':<18}{'E[goals]':>11}{'conf':>8}")
    print("-" * 92)
    for r in preds.itertuples():
        flag = "" if r.residual < 0.02 else ("  ~" if r.residual < 0.05 else "  !!")
        print(f"{r.home + ' v ' + r.away:<34}{r.scoreline:>7}{r.p_scoreline*100:>6.1f}%"
              f"  {r.p_home*100:>3.0f}/{r.p_draw*100:>2.0f}/{r.p_away*100:>3.0f}     "
              f"{r.xg_home:>4.2f}-{r.xg_away:<4.2f}{r.residual:>7.4f}{flag}")
    print("-" * 92)
    print("conf: residual = prob-space RMSE of the market fit; ~ / !! flag thin or "
          "inconsistent markets.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Most-likely scoreline per match (Phase 5).")
    ap.add_argument("--date", help="YYYY-MM-DD (tz-adjusted kickoff date)")
    ap.add_argument("--league", type=int, action="append", dest="leagues")
    ap.add_argument("--all-soccer", action="store_true")
    ap.add_argument("--tz-offset", type=float, default=0.0)
    ap.add_argument("--snapshot", help="read a saved fetch.py snapshot CSV instead of the API")
    ap.add_argument("--out", help="write the predictions table to CSV")
    args = ap.parse_args()

    model = load_model()

    if args.snapshot:
        df = pd.read_csv(args.snapshot)
    else:
        if not args.date:
            ap.error("--date is required unless --snapshot is given")
        leagues = None
        if args.all_soccer:
            leagues = [lg["id"] for lg in fetch.list_active_soccer_leagues()]
        elif args.leagues:
            leagues = args.leagues
        df = fetch.fetch_date(args.date, leagues, tz_offset_hours=args.tz_offset)

    preds = predict_snapshot(df, model)
    _print_table(preds, model)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(out, index=False)
        print(f"\nWrote {len(preds)} predictions -> {out}")


if __name__ == "__main__":
    main()
