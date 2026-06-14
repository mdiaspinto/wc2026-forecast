"""Phase 1 — pull Pinnacle MATCH-LEVEL markets for every game on a given date.

We capture the three sharp, high-limit markets that over-determine a 2-parameter
goal model:
    * moneyline  (1X2)            -> home / draw / away
    * spread     (Asian Handicap) -> full ladder of home-handicap lines
    * total      (Over / Under)   -> full ladder of total-goal lines

Output is a *tidy long table* (one row per priced leg), still carrying the vig.
De-vig and the lambda inversion happen downstream in devig.py / inversion.py —
this module only does I/O and parsing.

Source: Pinnacle's unauthenticated "guest" arcadia API (same endpoints the old
outright scraper used). All markets are the FULL-MATCH period (period == 0).

Usage:
    python -m market_scoreline.fetch --date 2026-06-11 --league 2686
    python -m market_scoreline.fetch --date 2026-06-11 --league 2686 --out data/odds/snapshots/wc_2026-06-11.csv
    python -m market_scoreline.fetch --mock tests/fixtures/markets.json    # offline parse

Note: we place bets ~45 min pre-kickoff, so the operational capture is a cron at
kickoff-45 (Phase 6). This module is the capture primitive that cron will call.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Self-contained import: works as `python -m market_scoreline.fetch`,
# as `python fetch.py`, and when imported as a package module.
try:
    from .team_names import canon_team
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from team_names import canon_team

# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
BASE = "https://guest.api.arcadia.pinnacle.com/0.1"
SOCCER_SPORT_ID = 29
WC_LEAGUE_ID = 2686  # FIFA World Cup 2026 (default league)
FULL_MATCH_PERIOD = 0

# Pinnacle's guest API needs an app-level "X-API-Key". It's the PUBLIC key shipped
# in Pinnacle's own website JS (not an account credential) — but we keep it OUT of
# the repo. Supply it via the PINNACLE_API_KEY env var: CI reads the Actions secret
# of the same name; locally run `export PINNACLE_API_KEY=...` once.
PINNACLE_API_KEY = os.environ.get("PINNACLE_API_KEY", "")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "X-API-Key": PINNACLE_API_KEY,
    "Referer": "https://www.pinnacle.com/",
    "Accept": "application/json",
}

LONG_COLUMNS = [
    "match_id", "league_id", "start_time_utc", "home", "away",
    "market_type", "period", "line", "side", "american", "decimal", "implied_raw",
]


# --------------------------------------------------------------------------- #
# Odds conversions
# --------------------------------------------------------------------------- #
def american_to_decimal(price: int) -> float:
    if price > 0:
        return 1.0 + price / 100.0
    return 1.0 + 100.0 / (-price)


def american_to_implied(price: int) -> float:
    """Per-leg implied probability (still includes the bookmaker margin)."""
    if price > 0:
        return 100.0 / (price + 100.0)
    return -price / (-price + 100.0)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _get(url: str, timeout: int = 20) -> object:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def fetch_league_matchups(league_id: int) -> list:
    """All matchups (games + specials) currently listed for a league."""
    return _get(f"{BASE}/leagues/{league_id}/matchups")


def fetch_straight_markets(matchup_id: int) -> list:
    """All 'straight' markets (moneyline/spread/total, every period & alternate)
    for one matchup."""
    return _get(f"{BASE}/matchups/{matchup_id}/markets/straight")


def list_active_soccer_leagues() -> list[dict]:
    """Soccer leagues that currently have at least one matchup listed.
    Lets the product sweep 'all matches on a date' beyond a single league."""
    leagues = _get(f"{BASE}/sports/{SOCCER_SPORT_ID}/leagues?brandId=0")
    return [lg for lg in leagues if lg.get("matchupCount", 0) > 0]


# --------------------------------------------------------------------------- #
# Matchup filtering / parsing
# --------------------------------------------------------------------------- #
def _is_game(m: dict) -> bool:
    """A real GOALS game matchup (two aligned teams), not a prop/special/outright
    and not a derivative market (Corners/Bookings/Shots/...).

    Derivatives carry a non-null `parent` pointing at the main matchup and a
    non-'Regular' `units` ('Corners', 'Bookings', ...). The goals game has
    parent is None and units in (None, 'Regular').
    """
    if m.get("special") is not None:
        return False
    if m.get("type") not in (None, "matchup"):
        return False
    if m.get("parent") is not None:
        return False
    if m.get("units") not in (None, "Regular"):
        return False
    parts = m.get("participants", []) or []
    aligns = {p.get("alignment") for p in parts}
    return "home" in aligns and "away" in aligns


def _teams(m: dict) -> tuple[str, str]:
    home = away = None
    for p in m.get("participants", []):
        if p.get("alignment") == "home":
            home = p.get("name")
        elif p.get("alignment") == "away":
            away = p.get("name")
    return canon_team(home), canon_team(away)


def _start_dt(m: dict) -> datetime | None:
    s = m.get("startTime")
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _on_date(m: dict, date_str: str, tz_offset_hours: float) -> bool:
    dt = _start_dt(m)
    if dt is None:
        return False
    local = dt.timestamp() + tz_offset_hours * 3600.0
    return datetime.fromtimestamp(local, tz=timezone.utc).strftime("%Y-%m-%d") == date_str


def parse_markets(matchup: dict, markets: list) -> list[dict]:
    """Flatten one matchup's straight markets into tidy long rows.

    Keeps only the full-match period. For moneyline, `line` is NaN and `side` is
    home/draw/away. For spread, `line` is the HOME handicap and `side` is
    home/away. For total, `line` is the goal line and `side` is over/under.
    """
    home, away = _teams(matchup)
    mid = matchup.get("id")
    league_id = (matchup.get("league") or {}).get("id")
    start = matchup.get("startTime")
    rows: list[dict] = []

    for mk in markets:
        if mk.get("period") != FULL_MATCH_PERIOD:
            continue
        mtype = mk.get("type")
        if mtype not in ("moneyline", "spread", "total"):
            continue
        for pr in mk.get("prices", []):
            price = pr.get("price")
            if price is None:
                continue
            desig = pr.get("designation")          # home/away/draw/over/under
            points = pr.get("points")              # handicap or total line; None for moneyline
            # Canonicalize the handicap to the HOME handicap so both legs of an AH
            # market share one `line` and group together. Pinnacle posts the away
            # leg with the opposite-signed points (home -1.5 <-> away +1.5).
            line = points
            if mtype == "spread" and points is not None and desig == "away":
                line = -points
            rows.append({
                "match_id": mid,
                "league_id": league_id,
                "start_time_utc": start,
                "home": home,
                "away": away,
                "market_type": mtype,
                "period": FULL_MATCH_PERIOD,
                "line": float(line) if line is not None else float("nan"),
                "side": desig,
                "american": int(price),
                "decimal": round(american_to_decimal(price), 5),
                "implied_raw": round(american_to_implied(price), 6),
            })
    return rows


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def fetch_date(date_str: str, league_ids: list[int] | None = None,
               tz_offset_hours: float = 0.0, pace: float = 0.12,
               verbose: bool = True) -> pd.DataFrame:
    """Tidy long table of all moneyline/spread/total legs for every game whose
    (tz-adjusted) kickoff falls on `date_str`, across the given leagues."""
    league_ids = league_ids or [WC_LEAGUE_ID]
    rows: list[dict] = []
    for lid in league_ids:
        try:
            matchups = fetch_league_matchups(lid)
        except Exception as e:  # noqa: BLE001 — network/availability, keep sweeping
            if verbose:
                print(f"  ! league {lid}: matchups fetch failed: {e}")
            continue
        games = [m for m in matchups if _is_game(m) and _on_date(m, date_str, tz_offset_hours)]
        if verbose:
            print(f"  league {lid}: {len(games)} game(s) on {date_str}")
        for m in games:
            try:
                markets = fetch_straight_markets(m["id"])
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"    ! markets fetch failed for {m['id']}: {e}")
                continue
            new = parse_markets(m, markets)
            rows.extend(new)
            if verbose:
                h, a = _teams(m)
                print(f"    {h} vs {a}: {len(new)} legs")
            time.sleep(pace + random.random() * pace)
    return pd.DataFrame(rows, columns=LONG_COLUMNS)


def list_fixtures(date_str: str, league_ids: list[int] | None = None,
                  tz_offset_hours: float = 0.0, verbose: bool = False) -> list[dict]:
    """The day's games as lightweight fixture records, sorted by kickoff.

    Each record is {match_id, league_id, home, away, start_dt (UTC datetime),
    matchup (raw dict)}. The raw matchup is retained so `capture_matchup` can
    snapshot a single game at its T-45 without re-listing the league. This is the
    primitive the Phase-6 scheduler iterates over to time per-match captures.
    """
    league_ids = league_ids or [WC_LEAGUE_ID]
    fixtures: list[dict] = []
    for lid in league_ids:
        try:
            matchups = fetch_league_matchups(lid)
        except Exception as e:  # noqa: BLE001 — keep sweeping other leagues
            if verbose:
                print(f"  ! league {lid}: matchups fetch failed: {e}")
            continue
        for m in matchups:
            if _is_game(m) and _on_date(m, date_str, tz_offset_hours):
                h, a = _teams(m)
                fixtures.append({
                    "match_id": m.get("id"),
                    "league_id": (m.get("league") or {}).get("id", lid),
                    "home": h, "away": a,
                    "start_dt": _start_dt(m),
                    "matchup": m,
                })
    _far = datetime.max.replace(tzinfo=timezone.utc)
    fixtures.sort(key=lambda f: f["start_dt"] or _far)
    return fixtures


def capture_matchup(matchup: dict) -> pd.DataFrame:
    """Tidy long table for ONE game — the single-match capture the cron fires at
    T-45. `matchup` is a raw dict from `list_fixtures` / `fetch_league_matchups`."""
    markets = fetch_straight_markets(matchup["id"])
    return pd.DataFrame(parse_markets(matchup, markets), columns=LONG_COLUMNS)


def parse_mock(path: str) -> pd.DataFrame:
    """Offline parse of a saved {matchup, markets} fixture (or list of them)."""
    blob = json.loads(Path(path).read_text())
    items = blob if isinstance(blob, list) else [blob]
    rows: list[dict] = []
    for it in items:
        rows.extend(parse_markets(it["matchup"], it["markets"]))
    return pd.DataFrame(rows, columns=LONG_COLUMNS)


def _summarize(df: pd.DataFrame) -> None:
    if df.empty:
        print("No legs captured.")
        return
    games = df.groupby(["match_id", "home", "away"])
    print(f"\nCaptured {len(games)} game(s), {len(df)} legs:")
    for (mid, h, a), g in games:
        n_ml = (g.market_type == "moneyline").sum()
        n_sp = g.loc[g.market_type == "spread", "line"].nunique()
        n_to = g.loc[g.market_type == "total", "line"].nunique()
        print(f"  {h} vs {a:<18} | 1X2 legs={n_ml}  AH lines={n_sp}  totals lines={n_to}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pinnacle match-market capture (Phase 1).")
    ap.add_argument("--date", help="YYYY-MM-DD (tz-adjusted kickoff date)")
    ap.add_argument("--league", type=int, action="append", dest="leagues",
                    help="League id (repeatable). Default: WC 2026 (2686).")
    ap.add_argument("--all-soccer", action="store_true",
                    help="Sweep every active soccer league instead of --league.")
    ap.add_argument("--tz-offset", type=float, default=0.0,
                    help="Hours to shift UTC kickoff before taking its date.")
    ap.add_argument("--out", help="Write the tidy long table to this CSV path.")
    ap.add_argument("--mock", help="Parse a saved fixture JSON instead of hitting the API.")
    args = ap.parse_args()

    if args.mock:
        df = parse_mock(args.mock)
    else:
        if not args.date:
            ap.error("--date is required unless --mock is used")
        leagues = None
        if args.all_soccer:
            print("Discovering active soccer leagues...")
            leagues = [lg["id"] for lg in list_active_soccer_leagues()]
            print(f"  {len(leagues)} active leagues")
        elif args.leagues:
            leagues = args.leagues
        df = fetch_date(args.date, leagues, tz_offset_hours=args.tz_offset)

    _summarize(df)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df["captured_at"] = datetime.now(timezone.utc).isoformat()
        df.to_csv(out, index=False)
        print(f"\nWrote {len(df)} legs -> {out}")


if __name__ == "__main__":
    main()
