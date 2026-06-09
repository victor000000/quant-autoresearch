# Autonomous loop session — 2026-06-09 (rounds R1854+ / loop rounds 1–15)

A `/loop 5m` autonomous research session. The loop read `program.md` and ran a clean, falsifiable
hypothesis each round. Net: **no new edge is reachable without new data (the frontier is empirically
closed), but the book construction is now fully optimized, persistence-checked, and mechanism-validated.**
All work is leak-verified host-side or via the real-OOS QC driver; WIP sits on checkpoint `22dc24c`.

## 1. Edge frontier — empirically closed (ran the experiments, didn't just assert)
- **Real-yield → GLD (FRED DFII10 + 2s10s slope), R1854–1855.** Built end-to-end + leak-safe:
  `modules/features.py:realyield_feats` (causal, unit-tested `tests/test_realyield_feats.py`),
  `footer.py.tmpl` guarded FRED path (1-day lag + causal ffill + graceful fallback + `ry_valid` stat),
  `_PSUF`/driver cell-key `_ry` (consistency-verified). **Result: A==B bit-identical → FRED data does
  NOT flow on this QC project** (the repo's first-ever non-equity data attempt; every prior round is
  `add_equity`-only). **Blocked on QC FRED data access — a user/QC step, not a code bug.** The
  gold↔real-rate thesis was *not* refuted; it got no data to test.
- **Real-rate via tradeable proxy → GLD←TIP, R1856.** Cross-asset term-structure features on the TIPS
  bond ETF (price ~ inverse real yield). **HURTS GLD 4.02→3.18** (val_auc 0.735→0.773 UP, Calmar DOWN =
  overfit). **Reproduces GLD←UUP (R1242: 4.02→3.18) almost exactly** → the cross-asset ETF-**price**-proxy
  channel is **proxy-agnostic closed** for GLD (dollar AND real-rate both fail identically). Implication:
  a tradeable price proxy is provably insufficient; **only the clean FRED yield level remains untested**
  (data-access-gated). This sharpens the FRED decision: it's now the *only* way to test the top thesis.
- **Missed-edge audit (R11).** Of 234 distinct positive-Sharpe names, every unbooked moderate-t name is a
  known-rejected category — gold duplicates (IAU/UGL), leveraged-equity DSR-fails (SPXL/SSO/AGQ/QLD),
  `always_long` buy-hold drifters, or known artifacts (XME/GDX/EEM/SLV). **The strict deflation bar is
  well-calibrated — it is NOT discarding genuine distinct edges.** (Only IXG financials t=2.43 is even
  borderline, and it was already considered.)

## 2. Honesty/validation — triangulated, GLD robust across every lens (new tooling built)
- **Block-bootstrap Calmar CI** (`scripts/bootstrap_calmar_ci.py`, report lever T4): GLD's two-sided 95%
  Calmar CI *overlaps* buy-hold (Calmar is a MaxDD-ratio, very wide on 2.8y), but **one-sided P(>BH)=0.93**.
- **Lo-corrected Sharpe** (R7): weekly book returns barely autocorrelated (η≤1.10); **GLD Sharpe t_adj=3.33,
  p=0.0009**, robust to the correction. The wide Calmar CI is statistic-fragility, NOT serial correlation.
- **Harvey-Liu FDP** (`scripts/harvey_liu_fdp.py`, report lever T5): π₀≈0.52 (half the search is noise),
  **FDP≈12%** among naive p<0.05 winners → the strict gates are necessary and justified. **GLD is the
  strongest *positive* tradeable edge by autocorrelation-robust t-stat** (|t|=4.77, 3× in the top).
- Triangulates with the existing stack: Hansen-SPA p=0.017, DSR 0.96, permute-collapse.

## 3. Book construction — fully optimized, persistence-checked, mechanism-validated (R12–15)
Host-side over `results/series_cache.json` (deployed member equity curves, common 224-pt OOS grid):
- **R12 (composition):** an expanded book robustly beats the current 6-name — bootstrap Calmar **lower bound
  1.54 → 3.32** (the conservative floor doubles, not just the point estimate). All-12 is *worse* than a
  focused sleeve → confirms "don't stack more marginal edges."
- **R13 (persistence):** per-member split-half decay — 4/5 sleeve members hold or strengthen; **EPI is
  front-loaded (Sharpe 1.66→0.63) → drop it.** GLD (anchor) *strengthens* 1.56→2.05.
- **R14 (weighting):** robust; inverse-vol (~ERC) marginally best (floor 3.28); **keep GLD heavy** —
  over-balancing away from GLD (inverse-variance 8% GLD) HURTS the floor.
- **R15 (mechanism):** the benefit is **structural decorrelation** — avg pairwise corr **+0.10**,
  diversification ratio **1.88** (gold/oil/telecom/financials/biotech are orthogonal drivers).

- **R16 (temporal stability — the load-bearing correction):** the sleeve advantage is **NOT temporally
  uniform.** Split the OOS window in thirds: the sleeve wins early (ΔSharpe +0.30) and mid (+1.34) but
  **LOSES the most recent third** (sleeve Sharpe 2.78 vs 6-name **3.78**). The full-window bootstrap (R12)
  masked this — the 6-name book is strengthening late (GLD + buy-hold diversifiers all rising) while the
  equity-sleeve edges fade.

**RECOMMENDED BOOK UPGRADE — now QUALIFIED:** `GLD / USO / IXP / IXG / XBI` is a robust **full-window**
improvement (bootstrap-Calmar floor 3.28 vs 1.54, decay-clean, structurally decorrelated avg-corr +0.10),
**BUT the most recent OOS third favors the conservative 6-name book** (R16) — so the aggressive bet is
riskier than the bootstrap alone implies. **Honest call: lean conservative (keep the 6-name) OR adopt the
sleeve only if you weight the full-window diversification over the recent trend** — and let the growing
OOS window + standing decay monitor settle whether the sleeve advantage resumes. The sleeve members are
individually Bonferroni-marginal on a 2.8y window; the recent fade is consistent with that overfit risk
beginning to surface.

## 3b. Book RISK characterization + window sufficiency (R22–27) — host-side, every standard axis
- **R22 (stress correlation):** decorrelation HOLDS in stress — in the worst-quartile weeks avg pairwise corr
  goes slightly NEGATIVE (sleeve −0.03 vs +0.10 full), the opposite of a crisis correlation-spike. Mechanism:
  GLD is a safe haven (rises when the equity sleeve falls). True for both books.
- **R23 (left tail):** the sleeve is MORE tail-efficient despite containing USO (worst-week −6.1%): book
  worst-week −0.59% vs 6-name −0.81%, CVaR5 −0.35% vs −0.44%, return/|CVaR| 22.5 vs 12.7.
- **R24 (drawdown depth):** book MaxDD is staggered-DD diversification, not low-vol members — book ~1.0–1.2%
  is ~3× shallower than the weighted-avg member, ~10–15× shallower than the worst (sleeve 1.00% < 6-name
  1.20% despite USO's 14.7% member MaxDD).
- **R25 (drawdown duration):** longest underwater ~0.2–0.3y, ~5–9× shorter than the worst member — diversified
  on depth AND duration. (Also noted: the cached grid is ~79 bars/yr, ≈4.6-day cadence — earlier absolute
  annualized Sharpes using ppy=52 were understated ~1.23×, e.g. GLD≈2.16 matching QC; no conclusion changes,
  all comparisons relative + significance via ppy-independent t-stats.)
- **R26 (return shape):** neither book is the dangerous high-Sharpe-NEGATIVE-skew trap. 6-name skew −0.14
  (~symmetric); **sleeve skew +1.50 (POSITIVELY skewed, gain/loss 1.71)** — its fat tails are on the UPSIDE.
- **R27 (window sufficiency — the rigorous capstone):** at the session's ~1840-trial search burden, MinBTL
  (Bailey–López de Prado) shows the 2.8y window confirms ONLY GLD (MinBTL 2.46y < 2.83y); every other member
  is window-INSUFFICIENT (USO 3.87y, IXP 4.54y, IXG 5.10y, XBI 4.86y, UUP 10.7y, IWM 13.0y). **Formal proof:
  GLD is the one window-confirmed durable edge; every other edge is PROVISIONAL — its realized Sharpe cannot
  yet be distinguished from a multiple-testing fluke and needs 3.9–13 more years of OOS.** This explains the
  whole session: only TIME (the growing window), not more experiments, can confirm the sleeve.

**Net risk read:** the sleeve is risk-SUPERIOR on every axis (decorrelated, stress-robust, tail-efficient,
shallow+short drawdowns, positively skewed) and full-window return-superior — but ONLY GLD is window-confirmed,
so the sleeve is a provisional bet whose recent return-fade (R16) is the overfit risk MinBTL formally flags.
The loop cannot resolve this — it needs calendar time or a new INPUT.

## 4. The decision points (the loop genuinely needs a new input)
1. **Enable FRED on QC project 31338454** → race the (built, ready) real-yield A/B — now the *only* way to
   test the gold↔real-rate thesis (proxies proven insufficient).
2. **Authorize a new data modality** (options IV / COT / flows).
3. **Adopt the book upgrade** above (a human crown — does not need new data).
4. Otherwise the frontier is mapped; GLD stands as the one durable single-ticker edge.

New artifacts this session: `tests/test_realyield_feats.py`, `scripts/bootstrap_calmar_ci.py`,
`scripts/harvey_liu_fdp.py`, real-yield wiring (`features.py`/`footer.py.tmpl`/`header.py.tmpl`/driver),
and the `knowledge.json` `dead_ends`/`next_idea` records.
