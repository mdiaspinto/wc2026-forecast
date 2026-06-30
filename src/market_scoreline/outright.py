"""Outright 'World Cup 2026 Winner' snapshot — the Tier-2 mirror for the tracker.

QuinielaTracker_v3.html normally pulls Pinnacle's outright winner market and the
live fixture board directly in the browser to refresh odds + reflect decided
matches. On networks that block betting domains that browser fetch hangs, so the
tracker also accepts a Tier-2 mirror: this module reproduces the tracker's
in-browser `fetchLive` on the server (where Pinnacle is reachable) and writes
docs/odds.json, which GitHub Pages serves from a non-betting domain.

Names are emitted exactly as Pinnacle returns them (English). The tracker applies
its own English->Spanish mapping (`pinToApp` / `pinClean`), so name handling stays
in one place and the mirror path is byte-for-byte equivalent to the live path
downstream of the fetch.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from market_scoreline import fetch

OUTRIGHT_DESC = "FIFA - World Cup 2026 Winner"

# Mirrors the tracker's pinClean: strip prop-market suffixes like "France (Corners)".
_PROP_SUFFIX = re.compile(
    r"\s*\((Corners|Bookings|Cards|Shots[^)]*|Offsides|Throw[^)]*)\)\s*$", re.I)
# Mirrors the tracker's per-participant prop guard.
_PROP_PARTICIPANT = re.compile(r"\((Corners|Bookings|Cards|Shots|Offsides|Throw)", re.I)


def _clean(name: str) -> str:
    return _PROP_SUFFIX.sub("", name or "").strip()


def build_outright(league_id: int = fetch.WC_LEAGUE_ID) -> dict:
    """Outright winner prices + alive teams, keyed by Pinnacle English names.

    Same logic as the tracker's fetchLive: a team is alive if it is in the winner
    market OR still has a scheduled (non-prop) fixture, so a long-shot Pinnacle
    drops from the winner market while still in the tournament is never mistaken
    for eliminated.
    """
    matchups = fetch.fetch_league_matchups(league_id)
    out = next((m for m in matchups
                if m.get("special")
                and (m["special"] or {}).get("description") == OUTRIGHT_DESC), None)
    if out is None:
        raise RuntimeError("outright winner market not found")

    pid2name = {p["id"]: p["name"] for p in out.get("participants", [])}
    markets = fetch._get(f"{fetch.BASE}/matchups/{out['id']}/markets/related/straight")
    market = next((x for x in (markets or [])
                   if isinstance(x.get("prices"), list) and x["prices"]), {"prices": []})

    odds_by_en: dict[str, int] = {}
    for pr in market["prices"]:
        en = pid2name.get(pr.get("participantId"))
        price = pr.get("price")
        if en and price is not None:
            odds_by_en[en] = int(price)

    fixture_alive: set[str] = set()
    for m in matchups:
        if m.get("special"):
            continue
        ps = m.get("participants") or []
        if len(ps) != 2:
            continue
        if any(_PROP_PARTICIPANT.search(p.get("name", "") or "") for p in ps):
            continue
        for p in ps:
            name = _clean(p.get("name", ""))
            if name:
                fixture_alive.add(name)

    alive = sorted({*odds_by_en.keys(), *fixture_alive})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "pinnacle/outright",
        "league": league_id,
        "oddsByEn": odds_by_en,
        "aliveEn": alive,
    }


def write_outright(out_path: Path, league_id: int = fetch.WC_LEAGUE_ID) -> dict:
    data = build_outright(league_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))
    return data


if __name__ == "__main__":
    import sys
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/odds.json")
    d = write_outright(dest)
    print(f"Wrote {len(d['oddsByEn'])} priced, {len(d['aliveEn'])} alive -> {dest}")
