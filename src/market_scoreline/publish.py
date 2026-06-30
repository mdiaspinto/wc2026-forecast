"""Phase 7 — cloud publish tick. The entrypoint GitHub Actions runs on a cron.

Unlike `daily.py`'s long-lived watch loop, this is a STATELESS, idempotent tick
suited to short-lived CI runs: each invocation
  1. captures the closing line for any match now inside [kickoff-LEAD, kickoff]
     that hasn't been captured yet (the committed snapshot files are the state),
  2. (re)builds docs/predictions.json for the public page from the locked closing
     snapshot if present, else a fresh non-persisted PREVIEW capture,
and exits. The workflow then commits any changed snapshots + predictions.json.

All modeling is reused, not reimplemented:
    fetch.list_fixtures / fetch.capture_matchup      (market capture)
    snapshots.latest_snapshot / save_snapshot        (immutable closing store = state)
    devig_long / constraints_from_devigged / invert  (inversion)
    matrix.*                                         (scoreline read-out)
    predict.load_model / notify.confidence           (params + verdict)

    python -m market_scoreline.publish                      # capture-if-due + build JSON
    python -m market_scoreline.publish --no-capture         # preview-only (local test)
    python -m market_scoreline.publish --out /tmp/p.json --horizon-days 3
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from market_scoreline import fetch, notify, snapshots
from market_scoreline import matrix as mx
from market_scoreline import predict as P
from market_scoreline.devig import devig_long
from market_scoreline.inversion import constraints_from_devigged, invert

ROOT = Path(__file__).resolve().parents[2]

# A T-LEAD capture window only a few minutes wide can be missed by the coarse,
# jittery */5 GitHub Actions cron. So when a tick fires and a kickoff's capture
# instant (KO-lead) is imminent (within this many minutes), the job sleeps until
# exactly that instant, then captures — turning a 5-min scheduler into a precise
# T-lead capture. Sized just above one cron interval so some tick always catches it.
SLEEP_HORIZON_MIN = 8
DEFAULT_OUT = ROOT / "docs" / "predictions.json"


def _utc_dates(horizon_days: int) -> list[str]:
    """Today plus the next `horizon_days` days, as UTC YYYY-MM-DD strings."""
    today = datetime.now(timezone.utc).date()
    return [(today + timedelta(days=i)).isoformat() for i in range(horizon_days + 1)]


def _predict_one(df: pd.DataFrame, model: dict, top_n: int) -> dict | None:
    """One match's long table -> structured prediction (mirrors predict.predict_snapshot,
    but returns machine-readable top scorelines for the JSON). None if unfittable."""
    dv = devig_long(df, model["devig_method"])
    mc = constraints_from_devigged(dv)
    if mc.n_constraints() < 2:
        return None
    r = invert(mc, model["rho"])
    m = r.matrix()
    i, j, p = mx.argmax_scoreline(m)
    ph, pd_, pa = mx.wdl(m)
    eh, ea = mx.expected_goals(m)
    tops = mx.top_scorelines(m, top_n)
    return {
        "scoreline": f"{i}-{j}",
        "p_scoreline": round(float(p), 4),
        "p_home": round(float(ph), 4), "p_draw": round(float(pd_), 4),
        "p_away": round(float(pa), 4),
        "xg_home": round(float(eh), 2), "xg_away": round(float(ea), 2),
        "top_scorelines": [[f"{a}-{b}", round(float(q), 4)] for a, b, q in tops],
        "residual": round(float(r.residual), 4),
        "verdict": notify.confidence(float(r.residual))[0],
    }


def build_predictions(dates, model, lead, do_capture, league_ids, top_n=4) -> dict:
    """Assemble the predictions payload across `dates`, capturing closing lines when due."""
    now = datetime.now(timezone.utc)
    matches: list[dict] = []
    for date in dates:
        try:
            fixtures = fetch.list_fixtures(date, league_ids, verbose=False)
        except Exception as e:  # noqa: BLE001 — never let one date kill the run
            print(f"  ! list_fixtures {date}: {e}")
            continue
        for fx in fixtures:
            try:
                row = _one_fixture(fx, date, now, model, lead, do_capture, top_n)
                if row is not None:
                    matches.append(row)
            except Exception as e:  # noqa: BLE001 — isolate per-match failures
                print(f"  ! {fx.get('home')} v {fx.get('away')}: {e}")
    matches.sort(key=lambda r: (r["kickoff_utc"] or "9999", r["home"]))
    n_closing = sum(m["is_closing"] for m in matches)
    print(f"Built {len(matches)} match prediction(s); {n_closing} closing, "
          f"{len(matches) - n_closing} preview.")
    return {
        "generated_at": now.isoformat(),
        "model": {"devig_method": model["devig_method"], "rho": model["rho"]},
        "lead_minutes": lead,
        "matches": matches,
    }


def _one_fixture(fx, date, now, model, lead, do_capture, top_n) -> dict | None:
    ko = fx["start_dt"]
    mid, home, away = fx["match_id"], fx["home"], fx["away"]

    # State lives in the committed snapshot store: a saved snapshot == closing captured.
    snap = snapshots.latest_snapshot(date, mid, home, away)

    # Capture the closing line once, when first inside the lead window.
    if (do_capture and snap is None and ko is not None
            and (ko - timedelta(minutes=lead)) <= now < ko):
        cdf = fetch.capture_matchup(fx["matchup"])
        if not cdf.empty:
            snap = snapshots.save_snapshot(cdf, date, mid, home, away)
            print(f"  captured CLOSING {home} v {away} "
                  f"(T-{int((ko - now).total_seconds() / 60)})")

    is_closing = snap is not None
    if snap is not None:
        pdf = snapshots.load_snapshot(snap)
        captured_at = (str(pdf["captured_at"].iloc[0])
                       if "captured_at" in pdf.columns and len(pdf) else None)
    else:
        # Preview (non-persisted) so the page is populated before T-45. Skip games
        # that already kicked off with no closing snapshot — nothing trustworthy to show.
        if ko is not None and now >= ko:
            return None
        pdf = fetch.capture_matchup(fx["matchup"])
        captured_at = now.isoformat()

    if pdf is None or pdf.empty:
        return None
    row = _predict_one(pdf, model, top_n)
    if row is None:
        return None
    row.update({
        "date": date,
        "match_id": int(mid) if mid is not None else None,
        "home": home, "away": away,
        "kickoff_utc": ko.isoformat() if ko is not None else None,
        "is_closing": bool(is_closing),
        "captured_at": captured_at,
    })
    return row


def _wait_for_imminent_capture(dates, lead, league_ids) -> None:
    """If a game's capture instant (KO-lead) is imminent, sleep until exactly then,
    so the coarse */5 cron still lands the closing capture near T-lead."""
    now = datetime.now(timezone.utc)
    soonest = None
    for date in dates:
        try:
            fixtures = fetch.list_fixtures(date, league_ids, verbose=False)
        except Exception:  # noqa: BLE001 — listing is best-effort here
            continue
        for fx in fixtures:
            ko = fx["start_dt"]
            if ko is None:
                continue
            target = ko - timedelta(minutes=lead)           # desired capture instant
            if (now < target <= now + timedelta(minutes=SLEEP_HORIZON_MIN)
                    and snapshots.latest_snapshot(date, fx["match_id"],
                                                  fx["home"], fx["away"]) is None):
                soonest = target if soonest is None else min(soonest, target)
    if soonest is not None:
        wait_s = (soonest - datetime.now(timezone.utc)).total_seconds()
        if wait_s > 0:
            print(f"Imminent kickoff: sleeping {wait_s:.0f}s to capture at T-{lead}...")
            time.sleep(wait_s)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cloud publish tick (Phase 7).")
    ap.add_argument("--horizon-days", type=int, default=2,
                    help="Cover today plus this many days ahead (default 2).")
    ap.add_argument("--lead", type=int, default=5,
                    help="Closing-capture lead minutes (capture at KO-lead; default 5).")
    ap.add_argument("--league", type=int, action="append", dest="leagues",
                    help="League id (repeatable). Default: WC 2026 (2686).")
    ap.add_argument("--top-n", type=int, default=4)
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="predictions.json path.")
    ap.add_argument("--no-capture", action="store_true",
                    help="Preview only; do not persist closing snapshots (local testing).")
    args = ap.parse_args()

    model = P.load_model()
    league_ids = args.leagues or [fetch.WC_LEAGUE_ID]
    dates = _utc_dates(args.horizon_days)
    print(f"Model: devig={model['devig_method']}, rho={model['rho']} | "
          f"lead=T-{args.lead} | dates {dates[0]}..{dates[-1]}")

    if not args.no_capture:
        _wait_for_imminent_capture(dates, args.lead, league_ids)
    data = build_predictions(dates, model, args.lead, not args.no_capture, league_ids, args.top_n)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    print(f"Wrote {len(data['matches'])} predictions -> {out}")

    # Tier-2 mirror for the offline QuinielaTracker: outright winner odds + alive
    # teams, served from GitHub Pages so friends on betting-blocked networks still
    # get fresh odds/results (best-effort; never fail the tick).
    try:
        from market_scoreline import outright
        od = outright.write_outright(ROOT / "docs" / "odds.json")
        print(f"Wrote outright mirror: {len(od['oddsByEn'])} priced, "
              f"{len(od['aliveEn'])} alive -> docs/odds.json")
    except Exception as e:  # noqa: BLE001 — mirror is optional, never fatal
        print(f"  ! outright mirror failed: {e}")

    # Grade past closing predictions vs actual results (best-effort; never fatal).
    try:
        from market_scoreline import results as R
        R.build_results(model)
    except Exception as e:  # noqa: BLE001
        print(f"  ! results build failed: {e}")


if __name__ == "__main__":
    main()
