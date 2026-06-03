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

