# Session summary 2026-06-08 — deliverables for human/Opus review

A long autonomous session (driven by `/loop` + the directives below). All work is leak-verified and sits
in the working tree (per the project's "commit done by human/Opus" convention). This is the single review
entry point; details in the linked docs/memories.

## 1. Website deep-reorganize — ✅ DEPLOYED LIVE
- Fixed a real data bug: hero rendered Calmar **5.16/7-names** contradicting its own "+USO PROPOSED" pill.
  Now sources the honest **6-name 4.62** book; USO is one upgrade pill. (`console/data.py` book_resolver.)
- Ended deploy drift: the live `:80` service was serving the **stale pre-refactor monolith** (1.6 MB) since
  2026-06-05; restarted → clean 10-section narrative build (175 KB). Verified (5 curls + full Flask smoke test).
- Reorg: narrative section order, USO oil-card re-sourced to the real leak-fixed crown, type-tokens, a11y,
  lazy-graph appendix, favicon note. Changes uncommitted (revert: `git checkout scripts/console reports/style.css
  reports/console.js && sudo systemctl restart autoresearch-reports`).

## 2. Wang transcripts — ✅ MINED OUT
~95% exhausted; the workflow's "live lever" (diff-strength trend_leg ensemble) was **already raced + dead** on
GLD (3.66<4.02, ledger). Only untested in-rule remainder = fracdiff/entropy feature-blocks (low EV). Doc:
`WANG_INVESTIGATION.md`.

## 3. New methods (Wang + internet) — 5 edge DISCARDs + 2 HONESTY-TOOLING WINS
- DISCARDs (all real learnings): EVT crash-feature (structurally nondeployable, trades cap ~44), Garleanu-Pedersen
  `aim` sizer (Calmar-negative, Sharpe/turnover-positive), dispersion-entropy (feature-crowds), path-signature
  (4th feature-add fail). The feature/sizer levers are CLOSED across all types.
- ✅ **MinBTL sufficiency gate** — wired live into `honest_audit.py` (UUP insufficient, leverage-trap quantified).
- ✅ **LOND online-FDR ledger** — `scripts/online_fdr_ledger.py`, simulation-validated; of 1210 rounds only 34
  discoveries survive (29=GLD) → GLD is the singular durable edge.
- Doc: `docs/research/NEW_METHODS_2026-06-08.md`.

## 4. Book re-validation — decay sweep
GLD **4.02** robust · USO **3.85** robust (crushes BH 0.92) · UUP decayed **1.30→0.44** (marginal) · IWM decayed
**0.665→0.47 < BH** (timing edge dead, already carried as always_long). The book's robust core is **GLD + USO**.

## 5. Universe screen → EQUITY SLEEVE (the session's main new result)
Screen confirmed the 3 booked mechanisms (trend/regime/revert) are the universe's edges. International/sector equity
yielded **3 real, permute-confirmed, non-redundant, decay-healthy** timing edges + a commodity-timing edge:
- **IXP** (global telecom trend) · **AAXJ** (Asia sadf) · **EWL** (Swiss ker) · **DJP** (commodity ker-timing, DBC-upgrade)
- Each clears EVERY gate (permute · decay-HOLDING · book-additive · split-half-OOS-stable · cost@10bp), but each is
  individually Bonferroni-marginal + idiosyncratic (seated-member tier); AAXJ/DJP are late-period-weighted.

**Book-upgrade MENU (Calmar²-weighted, real-OOS, net-of-cost-survivable):**
| Book | Calmar | Sharpe | MaxDD | Note |
|---|---|---|---|---|
| Current 6-name | 4.62 | 2.46 | 2.46% | baseline |
| **Aggressive: 10-name (DJP-replaces-DBC + sleeve)** | **6.48** | 3.51 | 1.84% | OOS-stable both halves; 6 marginal late-weighted edges |
| **Conservative: robust core (GLD/USO/IXP)** | ~5.1 early | — | — | best EARLY-half, regime-balanced, fewer overfit-prone edges |

⚠️ The high Calmars are OOS-stable but lean on marginal late-weighted edges — diversification benefit vs overfit
risk is the human's aggressive-vs-conservative call. Don't stack MORE marginal edges (diminishing robustness).

## 6. β200 mechanism-map + screen pre-filter
Fits cluster at modest-drift (β200≈0.5: IXP/AAXJ/DJP); US-equity NO-FITs are strong-drift (β200≫0.6: VTI/IVV 0.77).
Validates a **β200 pre-filter** (skip β≫0.65 strong-drift names → efficient future screens). β200 routes fit-ABILITY,
not mechanism (DJP β<0.5 but trends, not reverts — mechanism stays asset-intrinsic).

## 7. Spec-robustness (deep-v2 A3, `scripts/spec_curve.py`)
Across each crown's defensible specs (carrying axes × main sizers, deployable): **GLD is the singular
spec-ROBUST edge** (median Calmar 3.03 / 149 specs — most specs work; crown 4.02 is high-but-typical).
**USO (median 0.94/66), AAXJ (0.78/9), EWL (0.80/6) are "lone-max"** — only the exact crown spec works.
Nuance: across-all-labelers conflates *mechanism choice* with *tuning fragility*, so lone-max ≈
**mechanism-specific** (only oil-revert works on USO) — consistent with asset-intrinsic edges. Crown
takeaway: **GLD is the uniquely robust anchor; USO + sleeve are mechanism-specific single-spec edges**
→ reinforces the *conservative* book (GLD-anchored) over stacking lone-max members.

## 8. Cell-key correctness fix (2026-06-08, worth committing)
Found + fixed a real regression: the driver's infer cell-key (`run_autoresearch_round.py`) must mirror the
footer save-key `_PSUF` (`header.py.tmpl`) EXACTLY — for **features** (termstruct→`_ts`, evt/disp/sig→`_fx`)
and **calibration** (`_va`). A mismatch made infer read a nonexistent cell → false **0 trades**. This had
silently produced unreliable 0-trades "DISCARDs". After the fix, all 5 affected levers re-tested clean on GLD
and **all DISCARD** (cross-asset 3.29 · path-sig 3.61 · EVT 3.70 · Venn-Abers 3.92 · dispersion-entropy 3.98
vs champion 4.02) — **no masked edge; GLD champion robust with clean evidence.** New tooling also added:
`scripts/spec_curve.py` (NSE spec-robustness), `lo_sharpe_se_factor()` in `scripts/stats_rigor.py`, and a
(QC-platform-blocked, disabled) survival/AFT labeler. Lesson banked: a 0-trades result is usually a
plumbing/key bug, not a method failure — verify the key before concluding.

## 9. 2026-06-09 directives executed (per-ETF + new-methods + leak)
- **Leak audit (CLEAN):** found+fixed a cross-asset `.bfill()` future-seeding leak; 8-agent adversarial audit cleared the rest of the session's new code. No result invalidated.
- **Per-ETF module sweep (`per_etf_module_sweep.py` + `sweep_revalidate.py`):** re-ran all 36 `per_etf_best` champions × {cdf_overlay,dd_overlay,cdf_plain}×{corr,infogain}. **19 genuine, 10 stale/fake.** ⚠️**DROP 10:** QLD 4.06→0.22 (sliced_wasserstein artifact), SPXL/SSO/AGQ (leveraged), SOXX (leak-dead), ACWX/UUP (decayed), XME (vol fluke), XOP, BIL (cash). **ADOPT ~8 sizer improvements:** EWT 1.29→**5.13** (permute-REAL), DXJ→2.90, IDV→2.58, EPP→1.79, EWG→1.89, EWZ→1.49, EWY→2.19(marginal), AAXJ→2.44. Anchors GLD 4.02/USO 3.85 confirmed-optimal; sleeve AAXJ/EWL/IXP/DJP all re-validated + hold.
- **Data-snooping tooling (new, deep-v3):** `hansen_spa`+`stationary_bootstrap` in `stats_rigor.py` (self-tested); `certify_leaderboard.py` → **GLD edge data-snooping-robust (Hansen SPA p=0.017)** over the 16 book candidates.
- **New axis/labeler research (`NEW_AXES_LABELERS_2026-06-09.md`):** 12 new axes + 17 new labelers catalogued; raced the top of each — **dp_oracle** (cost-aware oracle labeler) + **vratio** (variance-ratio axis) both DISCARD → frontier confirmed (only logdollar/imbalance carry). Both registered + QC-feasible as tools.
- **New scripts to commit:** `per_etf_module_sweep.py`, `sweep_revalidate.py`, `certify_leaderboard.py`, `spec_curve.py`; `stats_rigor.py` (+Lo/SPA/bootstrap); `bar_builder.py` (+vratio); `labeler.py` (+dp_oracle, survival_aft); `NEW_AXES_LABELERS_2026-06-09.md`, `WANG_INTERNET_DEEP_V3_2026-06-09.md`.

## Human/Opus action items
1. **Crown decision on the book-upgrade menu** (aggressive 6.48 vs conservative robust core; or status-quo + USO).
2. **Reclassify** UUP/IWM (decayed timing → buy-hold diversifiers).
3. **Commit** the session's work (website + tooling + scripts) if desired.
4. **Reopen the frontier** with a new input: options IV/skew · COT/flows · VIX term-structure (the static-data
   frontier is comprehensively mapped + mechanistically explained — H₀: no more independent edges without new data).
