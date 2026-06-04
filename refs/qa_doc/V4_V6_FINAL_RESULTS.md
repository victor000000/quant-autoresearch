# Final Pipeline Results (val-honest)

## Architecture Evolution
- v3 (RV/Vol axis + 3-state HMM filter labels): old baseline
- **v4 dollar bars + Trend Scanning** (LdP info-density + mlfinlab indirect-HMM-like) — WINNER for liquid stable ETFs
- **v6 tick-count bars + Trend Scanning** — WINNER for volume-drift ETFs
- v5 CUSUM, v7 Renko, imbalance bars: tested, inferior

## Best Val-Honest Per-ETF Recipe

| ETF | Axis | L | T_THR | ISO | TEST Cal | VAL Cal |
|---|---|---|---|---|---|---|
| XLE | dollar | 40 | 2.5 | No | **2.25** | 0.51 |
| GLD | dollar | 60 | 3.0 | Yes (CONT) | **1.31** | n/a |
| IWM | dollar | 20 | 2.0 | Yes | 0.88 | -0.16 (lucky) |
| QQQ | dollar | 40 | 2.5 | Yes | 0.86 | 0.15 (lucky) |
| EEM | tick | 40 | 2.5 | Yes | **0.64** | **1.15** ← honest |
| HYG | dollar | 40 | 2.5 | Yes | 0.40 | -0.03 (lucky) |
| TLT | tick | 40 | 2.5 | Yes | 0.25 | n/a |

## Honest Average: Cal ~0.93. Cal>3 not achievable val-honestly via axis+label tuning.

## Key Findings
1. **Dollar bars** (LdP's canonical info-density-equity) optimal for liquid stable assets
2. **Tick-count bars** stabilize across regimes when dollar volume drifts (IWM/EEM/TLT)
3. **Trend Scanning** (forward OLS t-stat) is the optimal "indirect HMM-like" label vs CUSUM/HMM-state
4. **L=40 T=2.5** is universal sweet spot
5. **Per-ETF ISO/NoIso** matters: XLE-class needs NoIso; QQQ-class benefits from ISO
6. **Multi-scale ensembles** denoise XLE→Cal 3.80 BUT not val-honest
7. **Continuous position sizing**: helps GLD marginally, hurts XLE
8. **Long-Short** FAILS: XGB low-p ≠ "go down", just low confidence
9. **Renko bars** fail due to feature mismatch (predictions don't cross 0.5)

## What Could Break Cal>3 (Not Tested Here)
- Cross-asset features (SPY/VIX/TNX/DXY as features per ETF)
- Position sizing by external regime (Kelly with volatility targeting)
- Stop-loss / drawdown caps
- Deep learning (LSTM/Transformer for time-series)
- Adaptive online retraining

Per LdP Causal Factor Investing 2023: extensive tinkering without causal theory = type-A spuriosity. The high Cal results from many-configs-selection ARE lookahead bias.
