"""Phase 8 — grade closing-line predictions against actual final scores.

The closing predictions are re-derived from our COMMITTED snapshots (the lines we
captured at T-5), so they persist even after Pinnacle drops a played match from
its feed. Actual scores come from ESPN's free FIFA World Cup scoreboard. Output is
docs/results.json, consumed by the site's "Results" tab.

    python -m market_scoreline.results            # writes docs/results.json
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from market_scoreline import predict as P
from market_scoreline import snapshots
from market_scoreline.team_names import canon_team

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "docs" / "results.json"
ESPN_URL = ("https://site.api.espn.com/apis/site/v2/sports/soccer/"
            "fifa.world/scoreboard?dates={ymd}")
# First Round-of-32 kickoff. Everything on/after this date is a knockout match, so
# the `knockouts` feed below can never be polluted by group-stage results.
KO_START = "2026-06-28"
# ESPN display names that differ from our canonical (Pinnacle) spellings.
_FIX = {
    "Czechia": "Czech Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}


def _canon(name: str) -> str:
    c = canon_team(name)
    return _FIX.get(c, _FIX.get(name, c))


def _int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _espn_day(ymd: str) -> list[dict]:
    """Final/live scores ESPN lists for one UTC date (state: pre/in/post)."""
    try:
        req = urllib.request.Request(ESPN_URL.format(ymd=ymd),
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.load(r)
    except Exception:  # noqa: BLE001 — results are best-effort, never fatal
        return []
    out = []
    for e in d.get("events", []):
        comp = (e.get("competitions") or [{}])[0]
        state = comp.get("status", {}).get("type", {}).get("state")
        cs = comp.get("competitors", [])
        h = next((c for c in cs if c.get("homeAway") == "home"), None)
        a = next((c for c in cs if c.get("homeAway") == "away"), None)
        if not h or not a:
            continue
        out.append({"home": _canon(h["team"]["displayName"]),
                    "away": _canon(a["team"]["displayName"]),
                    "hg": _int(h.get("score")), "ag": _int(a.get("score")),
                    "hwin": bool(h.get("winner")), "awin": bool(a.get("winner")),
                    "state": state})
    return out


def _knockouts() -> list[dict]:
    """Every completed knockout-stage match with an EXPLICIT winner. ESPN's per-
    competitor `winner` flag decides penalty shootouts too (where the regulation
    score is a draw), which the graded `matches` table cannot express. Consumed by
    the QuinielaTracker to pin played matches into its bracket."""
    day = datetime.strptime(KO_START, "%Y-%m-%d").date()
    end = datetime.now(timezone.utc).date() + timedelta(days=1)   # tz slack
    out, seen = [], set()
    while day <= end:
        for ev in _espn_day(day.strftime("%Y%m%d")):
            if ev["state"] != "post":
                continue
            w = (ev["home"] if ev["hwin"] else ev["away"] if ev["awin"] else
                 # no flag: fall back to the score, decisive only
                 ev["home"] if (ev["hg"] or 0) > (ev["ag"] or 0) else
                 ev["away"] if (ev["ag"] or 0) > (ev["hg"] or 0) else None)
            key = frozenset((ev["home"], ev["away"]))   # KO pairs meet at most once
            if w is None or key in seen:
                continue
            seen.add(key)
            out.append({"date": day.isoformat(), "home": ev["home"], "away": ev["away"],
                        "score": f"{ev['hg']}-{ev['ag']}", "winner": w})
        day += timedelta(days=1)
    return out


def _espn_table(dates: list[str]) -> dict:
    """{frozenset(home,away): event} across each date ±1 day (tz slack)."""
    ymds = set()
    for d in dates:
        dt = datetime.strptime(d, "%Y-%m-%d")
        for off in (-1, 0, 1):
            ymds.add((dt + timedelta(days=off)).strftime("%Y%m%d"))
    table = {}
    for ymd in sorted(ymds):
        for ev in _espn_day(ymd):
            table[frozenset((ev["home"], ev["away"]))] = ev
    return table


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def build_results(model: dict, out_path: Path = DEFAULT_OUT) -> dict:
    dates = ([p.name for p in sorted(snapshots.SNAP_DIR.iterdir()) if p.is_dir()]
             if snapshots.SNAP_DIR.exists() else [])
    espn = _espn_table(dates) if dates else {}

    rows, graded, exact, reswin = [], 0, 0, 0
    for date in dates:
        for snap in snapshots.list_day(date):
            df = snapshots.load_snapshot(snap)
            if df.empty:
                continue
            home, away = str(df["home"].iloc[0]), str(df["away"].iloc[0])
            ko = str(df["start_time_utc"].iloc[0]) if "start_time_utc" in df.columns else None
            cap = str(df["captured_at"].iloc[0]) if "captured_at" in df.columns else None

            preds = P.predict_snapshot(df, model, top_n=1)
            if preds.empty:
                continue
            r = preds.iloc[0]
            pi, pj = (int(x) for x in r["scoreline"].split("-"))

            actual, status, exact_hit, res_hit = None, "pending", None, None
            ev = espn.get(frozenset((home, away)))
            if ev:
                status = ev["state"]
                if ev["state"] == "post" and ev["hg"] is not None and ev["ag"] is not None:
                    ah, aa = ((ev["hg"], ev["ag"]) if ev["home"] == home
                              else (ev["ag"], ev["hg"]))      # orient to our home/away
                    actual = f"{ah}-{aa}"
                    exact_hit = bool(pi == ah and pj == aa)
                    res_hit = bool(_sign(pi - pj) == _sign(ah - aa))
                    graded += 1
                    exact += exact_hit
                    reswin += res_hit
            rows.append({
                "date": date, "home": home, "away": away, "kickoff_utc": ko,
                "predicted": r["scoreline"], "p_predicted": round(float(r["p_scoreline"]), 4),
                "actual": actual, "status": status,
                "exact_hit": exact_hit, "result_hit": res_hit, "captured_at": cap,
            })

    rows.sort(key=lambda x: (x["kickoff_utc"] or "", x["home"]))
    summary = {
        "graded": graded,
        "exact": exact, "exact_pct": round(100 * exact / graded, 1) if graded else None,
        "result": reswin, "result_pct": round(100 * reswin / graded, 1) if graded else None,
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {"devig_method": model["devig_method"], "rho": model["rho"]},
        "summary": summary, "matches": rows,
        "knockouts": _knockouts(),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Graded {graded} completed match(es); exact {summary['exact_pct']}%, "
          f"1X2 {summary['result_pct']}% -> {out}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Grade closing predictions vs actual results (Phase 8).")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    build_results(P.load_model(), Path(args.out))


if __name__ == "__main__":
    main()
