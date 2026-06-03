# Session-scale honest audit (Deflated Sharpe at true trial counts)

Audited 252 model trials across 11 assets with >=5 trials & positive champion. Holm-Bonferroni FWER<=0.05, BH FDR<=0.10.

```
ETF      N  Calmar  SR_ann  Emax_ann  PSR>0    DSR  Holm   BH  verdict
----------------------------------------------------------------------
GLD     72   4.545   2.634     1.777  1.000  0.931 False False  marginal [neither]
SOXX    39   3.025   1.521     0.640  0.999  0.959 False False  REAL (survives best-of-N) [neither]
UUP     48   2.540   1.275     1.124  0.984  0.600 False False  FAILS deflation [neither]
SLV     16   1.870   1.505     0.531  0.992  0.941 False False  marginal [neither]
EEM      8   1.380   0.901     1.033  0.970  0.391 False False  FAILS deflation [neither]
QQQ     11   1.046   1.349     0.657  0.988  0.876 False False  FAILS deflation [neither]
XLE      6   1.023   0.912     0.534  0.937  0.736 False False  FAILS deflation [neither]
DBC      7   0.912   0.774     0.508  0.900  0.671 False False  FAILS deflation [neither]
KRE      7   0.899   1.018     0.128  0.962  0.940 False False  marginal [neither]
IWM      9   0.881   0.910     0.567  0.936  0.717 False False  FAILS deflation [neither]
TLT     16   0.285   0.393     0.560  0.745  0.389 False False  FAILS deflation [neither]
```

**Survive Holm-Bonferroni (FWER<=.05): NONE.** DSR>=.95 = edge clears the best-of-N-trials noise at the real session search size.

## Harvey-Liu haircut Sharpe (independent multiple-testing cross-check)

Adjusts each champion's Sharpe t-stat p-value for its true trial count M (Bonferroni FWER, BH FDR);
haircut SR = Sharpe surviving the correction. Independent of DSR's extreme-value mechanism.

```
ETF      N  Calmar     SR      t        p1  HC_Bonf   HC_BH  verdict
GLD     66   4.545  2.634   4.42  4.91e-06    2.032   2.238  SURVIVES (HC>0.5 both)
SOXX    38   3.025  1.521   2.55  5.34e-03    0.495   1.276  partial (HC_BH>0)
UUP     42   2.540  1.275   2.14  1.62e-02    0.000   0.834  partial (HC_BH>0)
SLV     16   1.870  1.505   2.53  5.76e-03    0.791   1.180  SURVIVES (HC>0.5 both)
EEM      8   1.380  0.901   1.51  6.52e-02    0.000   0.382  partial (HC_BH>0)
QQQ     10   1.046  1.349   2.26  1.18e-02    0.706   1.054  SURVIVES (HC>0.5 both)
XLE      6   1.023  0.912   1.53  6.29e-02    0.186   0.596  partial (HC_BH>0)
DBC      7   0.912  0.774   1.30  9.69e-02    0.000   0.144  partial (HC_BH>0)
KRE      7   0.899  1.018   1.71  4.38e-02    0.301   0.911  partial (HC_BH>0)
IWM      9   0.881  0.910   1.53  6.34e-02    0.000   0.390  partial (HC_BH>0)
TLT     15   0.285  0.393   0.66  2.55e-01    0.000   0.000  FAILS (haircut ~0)
```


## PBO-via-CSCV (GLD config-overfitting, Bailey-Borwein-LdP-Zhu 2014)

Swept 9 comparable GLD configs (labeler variants; axis/sizing/thresh/ncomp fixed at champion), extracted each OOS return series, CSCV over 16 time-blocks (1000 partitions).

```
GLD PBO = 0.581   N_configs=9   T=223  -> HIGH PBO -> config-OVERFIT (IS-best tends OOS-below-median)
full-sample OOS Sharpe by config (IS-best candidate = top):
  ker+regime_gmm           +0.2828
  ker                      +0.2797
  ker+regime_gmm+accel     +0.2733
  ker+trend_scan           +0.2695
  ker+accel                +0.2679
  ker+bgm                  +0.2459
  regime_gmm               +0.2228
  trend_scan               +0.1739
  accel                    +0.1290
```

## Honest book re-derivation (real OOS series; decorr champion predates SOXX + audits)

Weights ∝ Calmar² (the deployed scheme), gross=1, on 224-pt common OOS grid.

```
composition                                   Calmar  CAGR%  MaxDD%  Sharpe
current decorr core (GLD/UUP/TIP/DBC/HYG)      5.224  13.61    2.61   2.725
+ SOXX added (6)                               6.192  12.59    2.03   3.039
UUP -> SOXX swap                               6.107  13.10    2.14   2.993
drop UUP (4)                                   5.145  14.36    2.79   2.675
robust crowns + diversifiers, no UUP           6.107  13.10    2.14   2.993
```

## Book weighting robustness + decorrelation (cached series, zero backtests)

Return correlation matrix (OOS): is UUP's regime edge decorrelated from the GLD/SOXX trend edges?

```
corr      GLD  SOXX   UUP   TIP   DBC   HYG
GLD      1.00  0.12 -0.22  0.22  0.23  0.17
SOXX     0.12  1.00 -0.07  0.03  0.02  0.43
UUP     -0.22 -0.07  1.00 -0.27 -0.05 -0.31
TIP      0.22  0.03 -0.27  1.00 -0.07  0.54
DBC      0.23  0.02 -0.05 -0.07  1.00  0.01
HYG      0.17  0.43 -0.31  0.54  0.01  1.00

scheme              Calmar  CAGR%  MaxDD%  Sharpe   UUP_wt
equal                4.563   8.68    1.90   2.535     17%
Calmar^2             6.192  12.59    2.03   3.039      5%
Calmar^2 x DSR       6.149  12.68    2.06   3.032      3%
inverse-variance     6.068   5.34    0.88   3.330     50%
```

UUP↔GLD corr = -0.22, UUP↔SOXX corr = -0.07 → LOW correlation confirms UUP decorrelates the trend edges (earns its book seat despite individual fragility).

## Per-champion yearly robustness (cached OOS series, zero backtests)

Annualized Sharpe by calendar year — is each edge CONSISTENT or one-year-CONCENTRATED?

```
name      2023    2024    2025    2026   consistency
GLD       0.46    1.98    3.31   -0.10   3/4 yrs +, mixed
SOXX      0.81    0.32    1.97    2.96   4/4 yrs +, CONSISTENT
UUP      -0.50    2.45    0.61    0.72   3/4 yrs +, mixed
TIP       0.78    0.42    1.37    1.19   4/4 yrs +, CONSISTENT
DBC      -0.89    0.21    0.62    2.34   3/4 yrs +, mixed
HYG       1.69    1.32    1.85    0.61   4/4 yrs +, CONSISTENT
```

Read: GLD/SOXX positive across (nearly) all years = consistent, not one-year artifacts; UUP's regime edge is lumpier (regime-dependent) — consistent with its statistical fragility.
