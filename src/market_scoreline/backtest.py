"""Phase 4 — calibrate the model on historical Pinnacle CLOSING lines.

Uses football-data.co.uk, which carries closing Pinnacle prices + realized scores:
    PSCH/PSCD/PSCA      closing 1X2
    PC>2.5 / PC<2.5     closing Over/Under 2.5
    AHCh, PCAHH/PCAHA   closing main Asian-handicap line (home handicap)

Historical data only has the MAIN lines (not the full ladder the live product
captures), but 1X2 + totals(2.5) already over-determine the 2-parameter goal
model, which is all we need to:
    (AH is excluded -- football-data's handicap column convention corrupts the
     fit; see --with-ah note in main(). The live product uses Pinnacle's own AH.)
    1. SELECT the de-vig method (multiplicative / power / shin)
    2. CALIBRATE the global Dixon-Coles rho
by minimizing realized-scoreline log-loss, with exact-score hit-rate and 1X2
calibration reported alongside, against naive baselines.

Output: data/processed/market_model.json  {devig_method, rho, ...}

    python -m market_scoreline.backtest                 # 1X2 + totals (default)
    python -m market_scoreline.backtest --with-ah       # diagnostic: add historical AH
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from market_scoreline import matrix as mx
from market_scoreline.devig import devig, METHODS
from market_scoreline.inversion import MatchConstraints, invert

ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT / "data" / "odds" / "history"
OUT = ROOT / "data" / "processed"
PARAM_PATH = OUT / "market_model.json"

# football-data.co.uk league codes. Default: big-5 + a couple, recent seasons
# where closing Pinnacle columns exist (2019/20 onward).
DEFAULT_LEAGUES = ["E0", "D1", "I1", "SP1", "F1"]
DEFAULT_SEASONS = ["2021", "2122", "2223", "2324"]  # season "2324" == 2023/24

NEEDED = ["FTHG", "FTAG", "PSCH", "PSCD", "PSCA", "PC>2.5", "PC<2.5"]
AH_COLS = ["AHCh", "PCAHH", "PCAHA"]
# itertuples can't expose 'PC>2.5'/'PC<2.5' as attributes -> rename to safe ids.
RENAME = {"PC>2.5": "PCO25", "PC<2.5": "PCU25"}
RHO_GRID = np.round(np.arange(-0.22, 0.081, 0.02), 3)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _url(season: str, league: str) -> str:
    return f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"


def download(seasons, leagues, force=False) -> list[Path]:
    HIST.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in seasons:
        for lg in leagues:
            p = HIST / f"{lg}_{s}.csv"
            if force or not p.exists():
                try:
                    req = urllib.request.Request(_url(s, lg),
                                                 headers={"User-Agent": "Mozilla/5.0"})
                    p.write_bytes(urllib.request.urlopen(req, timeout=40).read())
                    print(f"  downloaded {lg}_{s}")
                except Exception as e:  # noqa: BLE001
                    print(f"  ! {lg}_{s}: {e}")
                    continue
            if p.exists():
                paths.append(p)
    return paths


def load_matches(paths, include_ah: bool) -> pd.DataFrame:
    frames = []
    cols = NEEDED + (AH_COLS if include_ah else [])
    for p in paths:
        try:
            df = pd.read_csv(p, encoding="latin-1")
        except Exception:  # noqa: BLE001
            continue
        if not set(NEEDED).issubset(df.columns):
            continue
        keep = [c for c in cols if c in df.columns]
        sub = df[keep].apply(pd.to_numeric, errors="coerce")
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=NEEDED).reset_index(drop=True)
    return out.rename(columns=RENAME)


# --------------------------------------------------------------------------- #
# Constraints from one historical row
# --------------------------------------------------------------------------- #
def _row_constraints(row, method: str, include_ah: bool, ah_sign: float) -> MatchConstraints:
    mc = MatchConstraints(0, "H", "A")
    p = devig([row.PSCH, row.PSCD, row.PSCA], method)
    mc.p_1x2 = (float(p[0]), float(p[1]), float(p[2]))

    pt = devig([row.PCO25, row.PCU25], method)
    mc.totals.append((2.5, float(pt[0])))

    if include_ah and not np.isnan(getattr(row, "AHCh", np.nan)):
        pah = devig([row.PCAHH, row.PCAHA], method)
        mc.handicaps.append((ah_sign * float(row.AHCh), float(pah[0])))
    return mc


def _detect_ah_sign(df: pd.DataFrame, method: str) -> float:
    """Home-cover prob should rise with realized margin. If the data's AHCh sign
    is opposite to our home-handicap convention, corr flips -> return -1."""
    margins, covers = [], []
    for row in df.itertuples():
        if np.isnan(getattr(row, "AHCh", np.nan)):
            continue
        pah = devig([row.PCAHH, row.PCAHA], method)
        covers.append(pah[0])
        margins.append(row.FTHG - row.FTAG)
    if len(covers) < 50:
        return 1.0
    c = np.corrcoef(covers, margins)[0, 1]
    return 1.0 if c >= 0 else -1.0


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _score_config(df: pd.DataFrame, method: str, rho: float,
                  include_ah: bool, ah_sign: float) -> dict:
    n = len(df)
    sl_ll = wdl_ll = exact = 0.0
    cal_pred, cal_obs = [], []      # 1X2 home-prob calibration
    cap = mx.CAP
    for row in df.itertuples():
        mc = _row_constraints(row, method, include_ah, ah_sign)
        r = invert(mc, rho)
        m = r.matrix()
        i, j = min(int(row.FTHG), cap), min(int(row.FTAG), cap)
        sl_ll += -np.log(max(m[i, j], 1e-12))
        ph, pd_, pa = mx.wdl(m)
        res_p = ph if row.FTHG > row.FTAG else (pd_ if row.FTHG == row.FTAG else pa)
        wdl_ll += -np.log(max(res_p, 1e-12))
        ai, aj, _ = mx.argmax_scoreline(m)
        exact += (ai == i and aj == j)
        cal_pred.append(ph)
        cal_obs.append(1.0 if row.FTHG > row.FTAG else 0.0)
    # calibration error: mean |pred - obs| over 10 prob bins (home win)
    cal_pred, cal_obs = np.array(cal_pred), np.array(cal_obs)
    bins = np.clip((cal_pred * 10).astype(int), 0, 9)
    ece = 0.0
    for b in range(10):
        msk = bins == b
        if msk.any():
            ece += msk.mean() * abs(cal_pred[msk].mean() - cal_obs[msk].mean())
    return {"method": method, "rho": float(rho),
            "scoreline_logloss": sl_ll / n, "wdl_logloss": wdl_ll / n,
            "exact_rate": exact / n, "home_ece": float(ece), "n": n}


def _baselines(df: pd.DataFrame) -> dict:
    """Naive references: empirical most-common scoreline + empirical score dist."""
    cap = mx.CAP
    hg = df.FTHG.clip(0, cap).astype(int)
    ag = df.FTAG.clip(0, cap).astype(int)
    counts = np.zeros((cap + 1, cap + 1))
    for i, j in zip(hg, ag):
        counts[i, j] += 1
    dist = counts / counts.sum()
    mode = np.unravel_index(int(np.argmax(counts)), counts.shape)
    sl_ll = -np.mean([np.log(max(dist[i, j], 1e-12)) for i, j in zip(hg, ag)])
    exact = np.mean([(i == mode[0] and j == mode[1]) for i, j in zip(hg, ag)])
    return {"mode_score": (int(mode[0]), int(mode[1])),
            "empirical_dist_logloss": float(sl_ll),
            "mode_exact_rate": float(exact)}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(seasons, leagues, include_ah, max_matches, save) -> dict:
    print("Downloading football-data.co.uk closing-Pinnacle CSVs...")
    paths = download(seasons, leagues)
    df = load_matches(paths, include_ah)
    if max_matches and len(df) > max_matches:
        df = df.sample(max_matches, random_state=1).reset_index(drop=True)
    print(f"Loaded {len(df)} matches with closing Pinnacle 1X2 + totals"
          + (" + AH" if include_ah else "") + ".")

    ah_sign = 1.0
    if include_ah:
        ah_sign = _detect_ah_sign(df, "multiplicative")
        print(f"AH sign auto-detect: {'OK (+1)' if ah_sign > 0 else 'FLIPPED (-1)'}")

    base = _baselines(df)
    print(f"\nBaseline: empirical-dist scoreline log-loss={base['empirical_dist_logloss']:.4f}, "
          f"mode {base['mode_score'][0]}-{base['mode_score'][1]} "
          f"exact-rate={base['mode_exact_rate']*100:.1f}%")

    print(f"\n{'method':<15}{'rho':>7}{'scoreLL':>10}{'wdlLL':>9}{'exact%':>9}{'homeECE':>9}")
    results = []
    for method in METHODS:
        for rho in RHO_GRID:
            r = _score_config(df, method, float(rho), include_ah, ah_sign)
            results.append(r)
            print(f"{method:<15}{rho:>7.2f}{r['scoreline_logloss']:>10.4f}"
                  f"{r['wdl_logloss']:>9.4f}{r['exact_rate']*100:>8.1f}%{r['home_ece']:>9.4f}")

    best = min(results, key=lambda r: r["scoreline_logloss"])
    print(f"\nBEST: method={best['method']} rho={best['rho']:.2f} "
          f"scoreLL={best['scoreline_logloss']:.4f} "
          f"(vs empirical baseline {base['empirical_dist_logloss']:.4f}), "
          f"exact={best['exact_rate']*100:.1f}%")

    model = {"devig_method": best["method"], "rho": best["rho"],
             "ah_sign": ah_sign if include_ah else None,
             "calibrated_on": {"leagues": leagues, "seasons": seasons, "n": best["n"]},
             "metrics": {k: best[k] for k in
                         ("scoreline_logloss", "wdl_logloss", "exact_rate", "home_ece")},
             "baseline": base}
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        PARAM_PATH.write_text(json.dumps(model, indent=2, default=str))
        print(f"\nSaved -> {PARAM_PATH}")
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate de-vig method + rho (Phase 4).")
    ap.add_argument("--seasons", nargs="*", default=DEFAULT_SEASONS)
    ap.add_argument("--leagues", nargs="*", default=DEFAULT_LEAGUES)
    # AH is OFF by default: football-data.co.uk's AHCh/PCAHH/PCAHA convention does
    # not cleanly map to a home-handicap (even after sign auto-detect it corrupts the
    # fit -- wdlLL 1.02 vs 0.97, homeECE 0.108 vs 0.036). The live product still uses
    # Pinnacle's full, correctly-signed AH ladder; this flag is diagnostic only.
    ap.add_argument("--with-ah", action="store_true",
                    help="(diagnostic) include the historical AH constraint")
    ap.add_argument("--max-matches", type=int, default=1500)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    run(args.seasons, args.leagues, args.with_ah, args.max_matches, not args.no_save)


if __name__ == "__main__":
    main()
