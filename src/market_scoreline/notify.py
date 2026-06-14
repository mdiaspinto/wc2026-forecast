"""Phase 6 — alerting. At each T-45 capture the scheduler calls `alert_match`,
which pushes you the most-likely scoreline right when you're about to bet.

Channels (best-effort, never fatal):
    * console   always — a one-line bet card + a terminal bell
    * macOS     desktop notification via osascript
    * webhook   optional POST to $MARKET_WEBHOOK_URL (e.g. a phone-push service),
                so the nudge reaches you away from the desk for a last-minute bet

A per-day digest markdown is appended at data/odds/snapshots/<date>/digest.md as a
durable log of every alert fired that day.

Confidence policy: residual is the market-fit RMSE (lower = tighter). We label
    < 0.02  clean  -> BET
    < 0.05  thin   -> CAUTION
    >= 0.05 noisy  -> SKIP (suppressed from "bet" framing)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from market_scoreline.snapshots import SNAP_DIR

CLEAN, THIN = 0.02, 0.05


def confidence(residual: float) -> tuple[str, str]:
    """(verdict, glyph) from the fit residual."""
    if residual < CLEAN:
        return "BET", "✓"
    if residual < THIN:
        return "CAUTION", "~"
    return "SKIP", "!!"


# --------------------------------------------------------------------------- #
# Channels
# --------------------------------------------------------------------------- #
def _macos(title: str, message: str) -> None:
    if sys.platform != "darwin":
        return
    try:
        safe = message.replace('"', "'")
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{safe}" with title "{title}"'],
            check=False, capture_output=True, timeout=5)
    except Exception:  # noqa: BLE001 — notifications are best-effort
        pass


def _webhook(payload: dict) -> None:
    url = os.environ.get("MARKET_WEBHOOK_URL")
    if not url:
        return
    try:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=8).read()
    except Exception:  # noqa: BLE001
        pass


def _digest_append(date_str: str, line: str) -> None:
    d = SNAP_DIR / date_str
    d.mkdir(parents=True, exist_ok=True)
    path = d / "digest.md"
    if not path.exists():
        path.write_text(f"# Closing-line predictions — {date_str}\n\n"
                        "| capture (UTC) | match | scoreline | P | 1X2 H/D/A | "
                        "conf | verdict |\n|---|---|---|---|---|---|---|\n")
    with path.open("a") as f:
        f.write(line + "\n")


# --------------------------------------------------------------------------- #
# Public
# --------------------------------------------------------------------------- #
def notify(title: str, message: str) -> None:
    """Generic push: console + macOS + webhook."""
    print(f"\a{title}: {message}")
    _macos(title, message)
    _webhook({"title": title, "message": message})


def alert_match(date_str: str, pred: dict, lead_min: int,
                snapshot_path: Path | str | None = None) -> str:
    """Fire the bet alert for one match. `pred` is one row of predict.predict_snapshot.
    Returns the verdict ('BET' / 'CAUTION' / 'SKIP')."""
    verdict, glyph = confidence(pred["residual"])
    home, away = pred["home"], pred["away"]
    score, p = pred["scoreline"], pred["p_scoreline"] * 100
    h, dr, a = pred["p_home"] * 100, pred["p_draw"] * 100, pred["p_away"] * 100

    title = f"T-{lead_min} {home} v {away}"
    body = (f"{score} ({p:.0f}%) | H/D/A {h:.0f}/{dr:.0f}/{a:.0f} "
            f"| {verdict} {glyph}")
    notify(title, body)
    if pred.get("top_scorelines"):
        print(f"     alts: {pred['top_scorelines']}")
    if snapshot_path:
        print(f"     line: {snapshot_path}")

    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    _digest_append(
        date_str,
        f"| {stamp} | {home} v {away} | {score} | {p:.0f}% | "
        f"{h:.0f}/{dr:.0f}/{a:.0f} | {pred['residual']:.4f} | {verdict} {glyph} |")
    return verdict
