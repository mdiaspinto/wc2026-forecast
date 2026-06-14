# Rebuild Plan — Pinnacle-Closing-Line Scoreline Predictor

**Date:** 2026-06-10
**Goal:** Replace the forward-looking WC bracket model with a product that, for all
matches on a given date, pulls Pinnacle odds (captured ~45 min before kickoff),
processes them, and outputs the **most-likely scoreline** per match.

---

## 1. Thesis shift

| | OLD (current repo) | NEW (target) |
|---|---|---|
| Primary signal | Elo + squad value + xG + edges | **Pinnacle closing line only** |
| Market role | 80% blend anchor (`market.py`, `w=0.80`) | **The entire model** |
| Output | Group scorelines + bracket + KO advancement | One scoreline matrix per match |
| Fundamentals data | Elo history, FBref, Understat, squads | **None used at inference** |
| Time anchor | Tournament-static | **T‑45min capture per match** |

Core principle: the Pinnacle closing line at T‑45 is the most efficient public
forecast available. We do not *predict* it; we **invert** it into a joint
scoreline distribution.

## 2. Why we synthesize, not read, the scoreline

Pinnacle's correct-score market is low-limit and stale relative to the main
markets. The sharp signal is in three high-limit markets that **over-determine**
a 2‑parameter goal model:

- **Asian Handicap ladder** → pins supremacy `λ_H − λ_A`.
- **Totals (O/U) ladder** → pins the sum `λ_H + λ_A`.
- **1X2 (moneyline)** → ties down draw mass / low-score dependence `ρ`.

With 2 free parameters (λ_H, λ_A) and a globally-calibrated ρ, these markets are
an over-determined system → robust least-squares inversion, and the fit residual
doubles as a data-quality / arbitrage sanity flag.

## 3. Methodology (the math)

### 3.1 De-vig
Pinnacle margin ≈ 2–2.5%. Per market, strip the overround:
- Two-way (each AH line, each totals line): `multiplicative` baseline; prefer
  `power` or `Shin` (handles informed-trader share).
- Three-way (1X2): favorite–longshot bias matters → `power`/`Shin` over naive
  proportional. **Final method chosen by backtest** (§5).

### 3.2 Inversion (market → λ)
Per match, solve:

```
minimize_{λ_H, λ_A}  Σ_m  w_m · D( p_model_m(λ_H, λ_A, ρ) , p_devig_m )
```

- `m` ranges over {P(H), P(D), P(A), P(over k) ∀ totals lines k,
  P(home covers h) ∀ AH lines h}.
- `D` = squared error in log-odds (recommended) or probability space.
- `w_m` = liquidity / proximity-to-main-line weight.
- `ρ` (Dixon-Coles low-score dependence) is **not** per-match identifiable from
  these aggregates → fix globally, calibrated once on historical scorelines (§5).
- Fast, convex-ish 2-D solve (`scipy.optimize.minimize`, L-BFGS / Nelder-Mead).
- **Fallback tiers:** if only 1X2 + one totals line are posted → 2 eqns / 2
  unknowns, solve exactly. If only 1X2 → fit λ_H,λ_A with a totals prior.

### 3.3 Scoreline matrix
Reuse the existing Dixon-Coles construction
(`src/single_model/scoreline.py::_dc_tau`, `score_matrix`):

```
P(i,j) = Pois(i; λ_H) · Pois(j; λ_A) · τ(i,j; ρ)   (normalized, 8×8, 7+ absorbing)
```

### 3.4 Outputs per match
- **Most-likely scoreline** = argmax cell.
- Full 8×8 matrix; top-N scorelines with probabilities.
- Derived P(1X2), E[goals], O/U — for a coherence check against the input market.
- **Fit residual** → confidence flag (large residual = thin/odd market, surface it).

## 4. Target package layout

New self-contained package (repurpose `src/single_model/` or new
`src/market_scoreline/`):

| Module | Responsibility | Reuse from current repo |
|---|---|---|
| `fetch.py` | Pinnacle arcadia client: per-match **straight** markets (moneyline + spread ladder + totals ladder) for a date/league | auth headers, `american_to_implied`, `fetch_matchups`/`fetch_prices_for`, `canon_team` from `data/odds/save_pinnacle.py` |
| `devig.py` | multiplicative / power / Shin for 2-way & 3-way | new |
| `inversion.py` | (λ_H, λ_A) optimizer over de-vigged constraints | new |
| `matrix.py` | DC matrix + derived quantities (argmax, top-N, 1X2, totals) | **lift `_dc_tau`, `score_matrix`** from `scoreline.py` |
| `predict.py` | CLI: date → fetch → devig → invert → matrix → table | rewrite |
| `backtest.py` | historical odds → pipeline → log-loss / exact-hit / calibration vs naive | new |
| `calibrate.py` | fit global ρ + select de-vig method, saved to JSON | adapt calibration pattern from `scoreline.py` |

Data:
- `data/odds/` — raw pulls + **T‑45 closing snapshots** (timestamped).
- `data/odds/history/` — backtest corpus (§5).
- `data/processed/` — `market_model.json` (ρ + de-vig choice), per-date predictions.

## 5. Backtest / validation corpus

Need historical **Pinnacle closing** odds + realized scores. Recommended source:
**football-data.co.uk** — provides `PSCH/PSCD/PSCA` (Pinnacle closing 1X2) plus
closing Asian Handicap & totals columns across many seasons/leagues. Large, free,
independent of the WC, ready-made for validating the inversion.

Backtest answers:
1. Does argmax-scoreline beat naive baselines (most-common 1-1 / 1-0)?
2. Calibration of de-vigged market probs vs realized (reliability curves,
   1X2 log-loss). Closing line should sit near the diagonal.
3. **Selects** the de-vig method and global ρ that minimize scoreline log-loss /
   maximize exact-score hit-rate.

Keep `data/matches/results_2010_plus.csv` **only** as the realized-score side of
the backtest. Everything else fundamentals-related is deleted.

## 6. What gets deleted (dead weight under the new thesis)

- **Elo:** `data/elo/`, point-in-time Elo joins, `compute_elo.py`.
- **Fundamentals/players/xG:** `data/players/`, `data/xg/`, `fbref_*.csv`,
  `scrape_fbref.py`, `data/scripts/scrape_fbref.py`,
  `src/features/{squad_strength,edge_features,build_*}.py`.
- **Tournament machinery:** `src/models/{simulator,bracket_picker*,ensemble,
  elo_anchored,gbm,calibrate,backtest*,predict_group_scores,...}.py`,
  `src/single_model/{ko,predict,fill_submission,strength,market,backtest_points}.py`,
  `data/contextual/` (schedule/groups/venues/slots).
- **WC deliverables:** `Bracket_*.md`, `Group_Score_Picks*.md`, `Final_Bracket.md`,
  `Model_Validation.md`, `*Submission*.xlsx`, `Step3–Step6` (forward-model design).
- **Outright market code:** `pinnacle_outright_odds.csv`, `save_pinnacle.py`'s
  outright/reach/group inversion (we now use *match* markets).

**Keep:** DC matrix math, Pinnacle API client scaffolding, `canon_team`,
historical scores (backtest only). Archive the rest under `_archive/` first commit,
delete after the new pipeline is green.

## 7. Phasing

- **Phase 0 — Scaffold.** Create new package; archive old tree; init git (repo is
  not currently versioned — do this first so deletions are recoverable).
- **Phase 1 — fetch.** Live Pinnacle match-market client → one date's raw ladder
  (moneyline + AH + totals). Mock fixtures until WC markets open.
- **Phase 2 — devig.** Implement + unit-test the three methods.
- **Phase 3 — matrix + inversion.** Port DC matrix; build optimizer; unit-test that
  inversion recovers known λ from synthetic de-vigged markets.
- **Phase 4 — backtest.** football-data.co.uk corpus → select de-vig + ρ → save
  `market_model.json`.
- **Phase 5 — predict CLI.** End-to-end `predict.py --date YYYY-MM-DD` → table of
  most-likely scorelines + confidence flags.
- **Phase 6 — ops.** Schedule the T‑45 capture per match, snapshot odds, alert on
  high fit-residual matches.

## 8. Locked decisions (2026-06-10)

1. **Markets to invert:** ✅ **Full ladder** — 1X2 + full AH ladder + full totals ladder.
2. **Dependence ρ:** ✅ **Global, calibrated** on historical scorelines; reused per match.
3. **Backtest corpus:** ✅ **Both** — calibrate de-vig + ρ on football-data.co.uk now,
   then re-validate/refit on accumulated WC closing lines as the tournament runs.
4. **De-vig method:** backtest-selected (Shin / power / multiplicative) on the above corpus.
5. **Build order:** ✅ git init → `fetch.py` (Phase 1) first.

Still to confirm later: T‑45 capture mechanism (cron at kickoff−45 vs continuous
logging with closing-line extraction).
