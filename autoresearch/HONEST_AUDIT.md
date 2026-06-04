# Session-scale honest audit (Deflated Sharpe at true trial counts)

Audited 269 model trials across 11 assets with >=5 trials & positive champion. Holm-Bonferroni FWER<=0.05, BH FDR<=0.10.

```
ETF      N  Calmar  SR_ann  Emax_ann  PSR>0    DSR  Holm   BH  verdict
----------------------------------------------------------------------
GLD     80   4.708   2.652     1.820  1.000  0.925 False False  marginal [neither]
SOXX    43   3.025   1.521     0.631  0.999  0.961 False False  REAL (survives best-of-N) [neither]
UUP     52   2.540   1.275     1.121  0.984  0.602 False False  FAILS deflation [neither]
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
GLD     74   4.708  2.652   4.45  4.25e-06    2.037   2.369  SURVIVES (HC>0.5 both)
SOXX    42   3.025  1.521   2.55  5.34e-03    0.451   1.281  partial (HC_BH>0)
UUP     46   2.540  1.275   2.14  1.62e-02    0.000   0.863  partial (HC_BH>0)
SLV     16   1.870  1.505   2.53  5.76e-03    0.791   1.180  SURVIVES (HC>0.5 both)
EEM      8   1.380  0.901   1.51  6.52e-02    0.000   0.382  partial (HC_BH>0)
QQQ     10   1.046  1.349   2.26  1.18e-02    0.706   1.054  SURVIVES (HC>0.5 both)
XLE      6   1.023  0.912   1.53  6.29e-02    0.186   0.596  partial (HC_BH>0)
DBC      7   0.912  0.774   1.30  9.69e-02    0.000   0.144  partial (HC_BH>0)
KRE      7   0.899  1.018   1.71  4.38e-02    0.301   0.911  partial (HC_BH>0)
IWM      9   0.881  0.910   1.53  6.34e-02    0.000   0.390  partial (HC_BH>0)
TLT     15   0.285  0.393   0.66  2.55e-01    0.000   0.000  FAILS (haircut ~0)
```

## Anytime-valid e-value monitor (peeking-robust; supersedes p-value/DSR re-checks)

H0: mean return <= 0 (edge dead). E-value >= 20 = significant at 0.05, VALID under continuous
monitoring (re-check anytime; merge re-validations by MULTIPLICATION). Testing-by-betting (WSR 2023).

```
champ    e-value  AV p=1/e        verdict  decay?
GLD         4.56    0.2193    weak (e>=1)  holding
SOXX        1.20    0.8362    weak (e>=1)  holding
UUP         1.31    0.7627    weak (e>=1)  holding
TIP         1.58    0.6324    weak (e>=1)  holding
DBC         1.26    0.7947    weak (e>=1)  holding
HYG         2.51    0.3982    weak (e>=1)  holding
```

Anytime-valid: unlike DSR/p-values, these e-values stay honest no matter how many times we re-check as the OOS window grows. Next re-validation just MULTIPLIES the new e-value in.

## Transaction-cost stress (explicit slippage; pipeline default = none)

Calmar after re-running each crown's infer (same decisions) with explicit per-fill slippage:

```
crown  default      5bp     10bp   erosion@10bp
GLD      3.472    2.857    2.258     35%
UUP      1.847    1.541    1.159     37%
```

## Honest book re-derivation (real OOS series; decorr champion predates SOXX + audits)

Weights ∝ Calmar² (the deployed scheme), gross=1, on 224-pt common OOS grid.

```
composition                                   Calmar  CAGR%  MaxDD%  Sharpe
prior core (GLD/UUP/TIP/DBC/HYG)               4.609  11.51    2.50   2.454
+ IWM added (6)                                4.617  11.36    2.46   2.460
drop UUP, keep IWM                             4.519  12.06    2.67   2.405
GLD + diversifiers only (no UUP/IWM)           4.511  12.24    2.71   2.399
both alpha names (GLD/UUP/IWM)                 5.061  12.46    2.46   2.254
```
