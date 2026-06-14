"""Minimal, self-contained team-name canonicalization.

Pinnacle uses a handful of non-standard spellings ('Turkiye', 'Curacao', 'USA').
We keep this tiny and dependency-free so the new package owns its own naming and
does not reach back into the (to-be-archived) features stack.
"""
from __future__ import annotations

# Pinnacle / sportsbook spelling -> canonical display name.
ALIASES: dict[str, str] = {
    "Turkiye": "Türkiye",
    "Curacao": "Curaçao",
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic",
}


def canon_team(name: str) -> str:
    if not isinstance(name, str):
        return name
    return ALIASES.get(name.strip(), name.strip())
