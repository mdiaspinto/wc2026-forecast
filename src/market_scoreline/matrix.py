"""Phase 3a — the scoreline matrix and the market quantities derived from it.

ONE object per match: a joint probability matrix P(home_goals, away_goals) built
from a double-Poisson with the Dixon-Coles low-score correction. Everything the
product reports (most-likely scoreline, 1X2, totals, handicaps, E[goals]) is read
off this single matrix.

    P(i,j) = Pois(i; lam_h) * Pois(j; lam_a) * tau(i,j; rho)   (normalized)

The `tau` correction (Dixon & Coles 1997) only perturbs the four lowest cells
(0-0, 0-1, 1-0, 1-1) to fix the well-known independent-Poisson draw mis-fit.
Ported from the pre-pivot single-model `scoreline.py`.

The market-quantity helpers (`prob_over`, `prob_home_cover`) compute the *fair
two-way probability* exactly the way a de-vigged Pinnacle line is defined: each
side's break-even implied prob from its payout structure (win / push-refund /
loss), renormalized across the two sides. This handles integer-line pushes and
quarter (split) lines correctly, so model and market are compared like-for-like.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

# Internal grid. Bigger than the 8x8 we *display* so totals/handicap tails aren't
# truncated when matching market lines. Argmax scoreline is unaffected.
CAP = 15
_N = CAP + 1
_G = np.arange(_N)
_MARGIN = _G[:, None] - _G[None, :]      # home_goals - away_goals
_TOTAL = _G[:, None] + _G[None, :]       # home_goals + away_goals


# --------------------------------------------------------------------------- #
# Matrix construction (ported)
# --------------------------------------------------------------------------- #
def _dc_tau(lh: float, la: float, rho: float) -> np.ndarray:
    tau = np.ones((_N, _N))
    tau[0, 0] = 1.0 - lh * la * rho
    tau[1, 0] = 1.0 + la * rho
    tau[0, 1] = 1.0 + lh * rho
    tau[1, 1] = 1.0 - rho
    return np.clip(tau, 1e-9, None)


def score_matrix(lam_home: float, lam_away: float, rho: float = 0.0) -> np.ndarray:
    """Normalized (CAP+1)x(CAP+1) joint scoreline matrix."""
    pmf_h = poisson.pmf(_G, lam_home)
    pmf_a = poisson.pmf(_G, lam_away)
    m = np.outer(pmf_h, pmf_a) * _dc_tau(lam_home, lam_away, rho)
    return m / m.sum()


# --------------------------------------------------------------------------- #
# Derived market quantities
# --------------------------------------------------------------------------- #
def wdl(m: np.ndarray) -> tuple[float, float, float]:
    """(P(home win), P(draw), P(away win)) — strict, no handicap."""
    return (float(m[_MARGIN > 0].sum()),
            float(m[_MARGIN == 0].sum()),
            float(m[_MARGIN < 0].sum()))


def expected_goals(m: np.ndarray) -> tuple[float, float]:
    return float((_G[:, None] * m).sum()), float((_G[None, :] * m).sum())


def _thresholds(line: float) -> list[float]:
    """A clean (integer/half) line -> [line]; a quarter line -> its two half-stakes."""
    frac = round(line - np.floor(line), 2)
    if frac in (0.25, 0.75):
        return [line - 0.25, line + 0.25]
    return [line]


def _fair_odds(stat: np.ndarray, probs: np.ndarray, line: float, direction: int) -> float:
    """Fair decimal odds for a leg that WINS when `stat` is on `direction` side of
    the line (push on equality refunds the stake). Splits quarter lines into two
    equal half-stakes. Returns inf if the leg can essentially never win."""
    ths = _thresholds(line)
    win = push = 0.0
    for t in ths:
        if direction > 0:
            win += probs[stat > t].sum()
        else:
            win += probs[stat < t].sum()
        push += probs[stat == t].sum()
    win /= len(ths)
    push /= len(ths)
    if win <= 1e-12:
        return np.inf
    return (1.0 - push) / win               # EV=0: o*win + push = 1


def _two_way_prob(stat: np.ndarray, probs: np.ndarray, line: float, direction: int) -> float:
    """De-vig-consistent fair probability of the `direction` side of a two-way
    line: normalize the two legs' raw implied (1/fair-odds)."""
    oa = _fair_odds(stat, probs, line, direction)
    ob = _fair_odds(stat, probs, line, -direction)
    qa, qb = 1.0 / oa, 1.0 / ob
    s = qa + qb
    return float(qa / s) if s > 0 else float("nan")


def prob_over(m: np.ndarray, line: float) -> float:
    """Fair P(Over `line` total goals), matching a de-vigged O/U line."""
    return _two_way_prob(_TOTAL, m, line, +1)


def prob_home_cover(m: np.ndarray, home_handicap: float) -> float:
    """Fair P(home covers the HOME handicap h): home wins the bet iff
    margin > -h. Matches a de-vigged Asian-handicap line."""
    return _two_way_prob(_MARGIN, m, -home_handicap, +1)


# --------------------------------------------------------------------------- #
# Scoreline read-outs (the product output)
# --------------------------------------------------------------------------- #
def argmax_scoreline(m: np.ndarray) -> tuple[int, int, float]:
    i, j = np.unravel_index(int(np.argmax(m)), m.shape)
    return int(i), int(j), float(m[i, j])


def top_scorelines(m: np.ndarray, n: int = 5) -> list[tuple[int, int, float]]:
    flat = np.argsort(m, axis=None)[::-1][:n]
    out = []
    for idx in flat:
        i, j = np.unravel_index(int(idx), m.shape)
        out.append((int(i), int(j), float(m[i, j])))
    return out
