"""Phase 6 — immutable snapshot store for captured closing lines.

Every T-45 capture is written once, timestamped, and never overwritten, so you
can always answer "what was the line when I bet?" and so the captured WC closing
lines accumulate into a backtest corpus over the tournament.

Layout:
    data/odds/snapshots/<date>/<match_id>_<slug>/<captured_at>.csv

    date        tz-adjusted kickoff date (YYYY-MM-DD)
    slug        home-v-away, filesystem-safe
    captured_at UTC capture instant, e.g. 20260614T181503Z

The capture instant in the filename is the immutability guarantee: re-capturing
the same match writes a new file alongside the old, never on top of it.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SNAP_DIR = ROOT / "data" / "odds" / "snapshots"


def _slug(text: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "-", str(text)).strip("-")
    return s or "x"


def _stamp(dt: datetime | None = None) -> str:
    dt = dt or datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def match_dir(date_str: str, match_id, home: str, away: str) -> Path:
    return SNAP_DIR / date_str / f"{match_id}_{_slug(home)}-v-{_slug(away)}"


def save_snapshot(df: pd.DataFrame, date_str: str, match_id, home: str, away: str,
                  captured_at: datetime | None = None) -> Path:
    """Write one capture immutably; returns the file path. Stamps every row with
    the capture instant so a snapshot is self-describing once detached from its path."""
    captured_at = captured_at or datetime.now(timezone.utc)
    d = match_dir(date_str, match_id, home, away)
    d.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["captured_at"] = captured_at.astimezone(timezone.utc).isoformat()
    base = _stamp(captured_at)
    path = d / f"{base}.csv"
    n = 1  # never overwrite an existing capture (e.g. two within the same second)
    while path.exists():
        path = d / f"{base}-{n}.csv"
        n += 1
    out.to_csv(path, index=False)
    return path


def latest_snapshot(date_str: str, match_id, home: str, away: str) -> Path | None:
    """Most recent capture for a match (the one nearest kickoff == the closing line)."""
    d = match_dir(date_str, match_id, home, away)
    if not d.exists():
        return None
    files = sorted(d.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def load_snapshot(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)


def list_day(date_str: str) -> list[Path]:
    """Every match's latest snapshot for a date (for a reproducible whole-day re-run)."""
    day = SNAP_DIR / date_str
    if not day.exists():
        return []
    out = []
    for sub in sorted(day.iterdir()):
        if sub.is_dir():
            files = sorted(sub.glob("*.csv"), key=lambda p: p.stat().st_mtime)
            if files:
                out.append(files[-1])
    return out
