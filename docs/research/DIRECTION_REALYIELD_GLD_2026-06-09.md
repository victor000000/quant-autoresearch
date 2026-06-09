# Final Direction: Real-Yield Feature on GLD — the Last Credible Untested Channel

*2026-06-09 · synthesized from a local-corpus + internet + adversarial-verification workflow (23 agents). Supersedes the vague "needs new data modality" frontier note with one concrete, runnable A/B.*

## Verdict in one line
On minute close+volume, the single-ticker **price/volume** frontier is closed; the only honest residual is **exogenous fundamental-macro features**, and the highest-EV move is one decisive A/B adding the **10-year TIPS real yield to GLD**.

## Why the obvious "new modalities" are mostly already closed
The internal position names options IV/skew, VIX term-structure, COT, flows and credit spreads as the "highest-leverage" reopeners. Stress-testing each against the repo's own logs deflates most of them:

- **Cross-asset ETF-price features are DEFINITIVELY CLOSED.** R1242 cleanly raced GLD←UUP (4.02→3.18/3.70), SPY←VXX (2.54→0.84) and USO←XOP (inert): exogenous slow scalars are either infogain-dropped or spuriously over-selected and **hurt OOS**.
- **IV-skew does not transfer to liquid ETFs.** The famous single-name skew alpha is largely a stock-borrow-fee artifact, falling ≥2/3 once high-fee names are excluded ([Muravyev-Pearson-Pollet 2025, JFE](https://www.sciencedirect.com/science/article/pii/S0304405X25001618)). SPY/GLD/USO are trivially shortable.
- **VIX-slope** is the same exogenous-vol channel that already scored net-negative on SPY, and SPY/QQQ have **no deployable edge** to lift (crash-veto is stuck below the 80-trade wall). Its quantified form (VRP) peaks at a **quarterly** horizon ([Bollerslev-Tauchen-Zhou, RFS 2009](https://academic.oup.com/rfs/article-abstract/22/11/4463/1565787)), mismatched to bar-level labels.
- **COT / flows / short-interest** are non-native on QC, cross-sectional in evidence, and cadence/leak-broken (weekly/bi-monthly with publication lag). ETF short-interest is ~64% operational shorting — dead for ETFs.
- **HY-OAS** fails OOS standalone ([Welch-Goyal, NBER w11468](https://www.nber.org/system/files/working_papers/w11468/w11468.pdf)) and the Aug-2023→Jun-2026 window has no credit-widening event for a gate to fire on.

The outside view confirms the price-only closure: [Harvey-Liu backtesting haircuts](https://people.duke.edu/~charvey/Research/Published_Papers/P120_Backtesting.PDF), TSMOM OOS-negative on ETFs, [vol-managed OOS failure (Cederburg et al.)](https://www.lehigh.edu/~xuy219/research/COWY.pdf), and [Gu-Kelly-Xiu](https://dachxiu.chicagobooth.edu/download/ML.pdf) microcap dependence.

## The one channel that survives scrutiny
R1242 closed cross-asset ETF **prices** — its note even concedes "NOT exhaustively closed." It never tested a **fundamental macro state variable**. Gold is mechanically a long-duration zero-coupon **real** asset, so the **real interest rate** is its dominant driver. The repo only ever used the **UUP dollar-ETF price** as a proxy (infogain-dropped, R1233) — but UUP and real rates decoupled repeatedly (e.g. 2022). The actual driver — **FRED DFII10 (10y TIPS real yield)** plus the 2s10s slope — is:

- **Free and QC-native daily** ([US Treasury Yield Curve, since 1990](https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/treasury-department/us-treasury-yield-curve); [FRED on QC](https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/fred/us-federal-reserve-(fred))),
- **Leak-clean** at a 1-day lag, numpy/XGBoost-only,
- **Single-ticker-legal** (exogenous feature, no pair), and
- Aimed at the **strongest surviving edge** (GLD, Calmar 4.02) — a win compounds on a real, deployed signal rather than a coin-flip on a dead name.

## First experiment (runnable A/B)
- **Ticker:** GLD.
- **B (champion):** `GLD_logdollar_trend_leg+regime_gmm_dd_overlay_t0.40` (Calmar 4.0218, val_auc 0.7348, ~710 OOS days).
- **A:** champion + `[DFII10 z-score, N-bar Δreal-yield, DGS10−DGS2 slope]` via the existing `spy_lc`/`termstruct` path (`modules/features.py`), `reduce=infogain`; lag 1 trading day, forward-fill onto event bars, mirror `add_data` into all deploy templates (footer/infer/infer_online/verify/live).
- **Gate:** beat Calmar 4.02 (a lift, not merely beat buy-hold), val_auc>0.52, **permuted-label collapse**, survive DSR/Bonferroni best-of-N at the ~2400-3500-trial burden, online==saved parity ≤1e-6.

## Honest assessment
The prior is **net-negative**: R1242's failure mechanism is delivery-agnostic and GLD←UUP failed exactly this way. Expect **~1-in-4** it lifts GLD; otherwise it **formally closes** the last credible new-modality feature channel on the best name. This is an edge-**lifter**, not a fourth mechanism. Run the honesty/validation tooling (block-bootstrap CIs, Lo-SE DSR fix, online-FDR) in parallel — it tightens claims but cannot raise performance. A genuine breakthrough still needs a new authorized INPUT ([options IV](https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/quantconnect/us-equity-option-universe) with provenance verified market-implied, COT/flows), which is a user decision — not something the loop can mine from current data.

## Ranked shortlist
1. **Real-yield (DFII10) + 2s10s slope → GLD** *(RECOMMENDED)* — free/native, leak-clean, cleanest fundamental theory, targets the strongest edge, distinct from the closed UUP-price proxy. One A/B either lifts GLD or completes closure on the best name.
2. **Variance Risk Premium (option IV² − own realized var) → GLD/USO** — weaker/sign-unstable theory, quarterly horizon mismatch, and QC option IV may be model-derived (verify provenance). Only if real-yield closes *and* options data is authorized.
3. **Honesty/validation tooling** (block-bootstrap Calmar CIs, Lo-SE DSR fix, Hansen SPA / Romano-Wolf StepM, online-FDR wiring) — high-confidence but tightening-only, no Calmar lift. Run in parallel as foundational correctness.
4. **Clock-anchored intraday momentum → SPY/QQQ** — under-tested but cost-fragile (~0.4 Sharpe net), and SPY/QQQ have no deployable edge. At most one rigorous A/B with realistic slippage.
5. **HY-OAS credit spread → HYG/equity sleeve** — fundamental + native, but fails OOS standalone and the OOS window has no credit event for the gate. Secondary only.
6. **IV-skew / VIX-TS / COT / flows / short-interest as standalone features** — refuted or dominated; do not spend a round here.

**Bottom line:** the frontier is mapped and effectively closed on price/volume; the real-yield→GLD A/B is the single highest-EV remaining move the loop can make without new authorized data.
