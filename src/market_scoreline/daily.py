"""Phase 6 — the daily driver. Run this each match-day.

Bets go in ~45 min before kickoff, so "the closing line" is a per-match T-45
price, not one morning snapshot. This driver captures each game at its own T-45,
snapshots it immutably, predicts the scoreline, and alerts you to bet.

Three modes:

  watch (default)  Schedule a capture for every game on the date at kickoff-LEAD,
                   then sleep until each fires. Per-match: capture -> snapshot ->
                   predict -> alert. This is the turnkey daily ritual — start it
                   once, leave it running, get a ping per match at T-45.

  --preview        Morning look at the card on CURRENT (non-closing) lines: fetch
                   the whole day, predict, print the table. For planning only —
                   the number you bet is the T-45 capture from watch/--now.

  --now            Capture+predict+alert right now for every game kicking off
                   within --window minutes, then exit. The manual workflow: run
                   it ~45 min before a kickoff cluster and bet what it prints.

    python -m market_scoreline.daily --date 2026-06-14 --league 2686            # watch
    python -m market_scoreline.daily --date 2026-06-14 --preview                # plan
    python -m market_scoreline.daily --date 2026-06-14 --now                    # bet now
    python -m market_scoreline.daily --date 2026-06-14 --watch --dry-run        # show schedule

Reproducibility: predictions are computed from the saved snapshot, so the alert
reflects exactly the line that was captured. Re-run a past day from disk with
--replay (reads data/odds/snapshots/<date>/ instead of the API).
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import pandas as pd

from market_scoreline import fetch, notify, snapshots
from market_scoreline import predict as P

POLL_SECONDS = 60  # max sleep granularity so Ctrl-C stays responsive


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _hm(dt: datetime | None, tz_offset: float) -> str:
    if dt is None:
        return "  ?  "
    local = datetime.fromtimestamp(dt.timestamp() + tz_offset * 3600, tz=timezone.utc)
    return local.strftime("%H:%M")


# --------------------------------------------------------------------------- #
# One match: capture -> snapshot -> predict -> alert
# --------------------------------------------------------------------------- #
def capture_and_predict(fx: dict, date_str: str, model: dict, top_n: int,
                        lead_min: int) -> str | None:
    """Returns the verdict, or None if nothing was captured/fit."""
    home, away = fx["home"], fx["away"]
    try:
        df = fetch.capture_matchup(fx["matchup"])
    except Exception as e:  # noqa: BLE001
        notify.notify(f"{home} v {away}", f"capture FAILED: {e}")
        return None
    if df.empty:
        notify.notify(f"{home} v {away}", "no markets returned — SKIP")
        return None

    path = snapshots.save_snapshot(df, date_str, fx["match_id"], home, away)
    preds = P.predict_snapshot(df, model, top_n=top_n)
    if preds.empty:
        notify.notify(f"{home} v {away}", "markets too thin to fit — SKIP")
        return None

    row = preds.iloc[0].to_dict()
    return notify.alert_match(date_str, row, lead_min=lead_min, snapshot_path=path)


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_watch(date_str, fixtures, model, lead, top_n, tz_offset, dry_run) -> None:
    sched = []
    for fx in fixtures:
        ko = fx["start_dt"]
        cap = None if ko is None else ko - pd.Timedelta(minutes=lead).to_pytimedelta()
        sched.append({"fx": fx, "ko": ko, "cap": cap, "done": False})

    print(f"\nWatch schedule for {date_str}  (lead={lead} min, {len(sched)} games)")
    print("-" * 72)
    print(f"{'kickoff':>8}{'capture':>10}   match")
    print("-" * 72)
    for s in sched:
        print(f"{_hm(s['ko'], tz_offset):>8}{_hm(s['cap'], tz_offset):>10}   "
              f"{s['fx']['home']} v {s['fx']['away']}")
    print("-" * 72)
    if dry_run:
        print("(dry-run: schedule only, not capturing)")
        return
    if not sched:
        return

    while True:
        now = _now()
        pending = [s for s in sched if not s["done"]]
        if not pending:
            print("\nAll games captured. Done.")
            return
        for s in pending:
            ko, cap = s["ko"], s["cap"]
            if ko is not None and now >= ko:
                print(f"  kickoff passed, skipping {s['fx']['home']} v {s['fx']['away']}")
                s["done"] = True
                continue
            if cap is None or now >= cap:
                mins = int(round((ko - now).total_seconds() / 60)) if ko else lead
                print(f"\n[{now.strftime('%H:%M:%S')}Z] capturing "
                      f"{s['fx']['home']} v {s['fx']['away']} (T-{mins})")
                capture_and_predict(s["fx"], date_str, model, top_n, max(mins, 0))
                s["done"] = True
        pending = [s for s in sched if not s["done"]]
        if not pending:
            print("\nAll games captured. Done.")
            return
        # next event = soonest capture time still ahead (else its kickoff)
        nxt = min((s["cap"] if (s["cap"] and s["cap"] > now) else s["ko"])
                  for s in pending if s["ko"])
        sleep_s = max(1.0, min(POLL_SECONDS, (nxt - now).total_seconds()))
        time.sleep(sleep_s)


def run_now(date_str, fixtures, model, lead, top_n, window, tz_offset) -> None:
    now = _now()
    due = []
    for fx in fixtures:
        ko = fx["start_dt"]
        if ko is None:
            continue
        mins = (ko - now).total_seconds() / 60
        if 0 < mins <= window:
            due.append((fx, int(round(mins))))
    if not due:
        print(f"No games kicking off within {window} min "
              f"(of {len(fixtures)} on {date_str}). Nothing to capture.")
        return
    print(f"Capturing {len(due)} game(s) kicking off within {window} min:")
    for fx, mins in due:
        print(f"\n[T-{mins}] {fx['home']} v {fx['away']}")
        capture_and_predict(fx, date_str, model, top_n, mins)


def run_preview(date_str, leagues, model, tz_offset, replay) -> None:
    if replay:
        paths = snapshots.list_day(date_str)
        if not paths:
            print(f"No saved snapshots for {date_str} to replay.")
            return
        df = pd.concat([snapshots.load_snapshot(p) for p in paths], ignore_index=True)
        print(f"Replaying {len(paths)} saved snapshot(s) for {date_str}.")
    else:
        df = fetch.fetch_date(date_str, leagues, tz_offset_hours=tz_offset)
    preds = P.predict_snapshot(df, model)
    if not replay:
        print("\n** PREVIEW — current (non-closing) lines. Bet the T-45 capture. **")
    P._print_table(preds, model)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Daily closing-line scoreline driver (Phase 6).")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (tz-adjusted kickoff date)")
    ap.add_argument("--league", type=int, action="append", dest="leagues",
                    help="League id (repeatable). Default: WC 2026 (2686).")
    ap.add_argument("--all-soccer", action="store_true",
                    help="Sweep every active soccer league instead of --league.")
    ap.add_argument("--tz-offset", type=float, default=0.0,
                    help="Hours to shift UTC for date selection AND clock display.")
    ap.add_argument("--lead", type=int, default=5, help="Capture this many min before kickoff.")
    ap.add_argument("--top-n", type=int, default=4, help="Alternative scorelines to surface.")

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--watch", action="store_true", help="(default) schedule per-match T-45 captures")
    mode.add_argument("--preview", action="store_true", help="card on current non-closing lines")
    mode.add_argument("--now", action="store_true", help="capture games within --window now, then exit")
    ap.add_argument("--window", type=int, default=None,
                    help="--now horizon in minutes (default: --lead).")
    ap.add_argument("--dry-run", action="store_true", help="--watch: print schedule and exit")
    ap.add_argument("--replay", action="store_true", help="--preview: read saved snapshots, not the API")
    args = ap.parse_args()

    model = P.load_model()
    print(f"Model: devig={model['devig_method']}, rho={model['rho']}")

    leagues = None
    if args.all_soccer:
        print("Discovering active soccer leagues...")
        leagues = [lg["id"] for lg in fetch.list_active_soccer_leagues()]
        print(f"  {len(leagues)} active leagues")
    elif args.leagues:
        leagues = args.leagues

    if args.preview:
        run_preview(args.date, leagues, model, args.tz_offset, args.replay)
        return

    fixtures = fetch.list_fixtures(args.date, leagues, tz_offset_hours=args.tz_offset, verbose=True)
    print(f"{len(fixtures)} game(s) on {args.date}.")

    if args.now:
        window = args.window if args.window is not None else args.lead
        run_now(args.date, fixtures, model, args.lead, args.top_n, window, args.tz_offset)
    else:  # watch is the default
        run_watch(args.date, fixtures, model, args.lead, args.top_n,
                  args.tz_offset, args.dry_run)


if __name__ == "__main__":
    main()
