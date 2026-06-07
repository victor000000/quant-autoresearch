# Session-scale honest audit (Deflated Sharpe at true trial counts)

Audited 2379 model trials across 105 assets with >=5 trials & positive champion. Holm-Bonferroni FWER<=0.05, BH FDR<=0.10.

```
ETF      N  Calmar  SR_ann  Emax_ann  PSR>0    DSR  Holm   BH  verdict
----------------------------------------------------------------------
BIL      6  30.430  13.653     1.627  1.000  1.000  True True  REAL (survives best-of-N) [Holm+BH]
SSO     57   6.960   0.840     1.400  0.998  0.030 False False  FAILS deflation [neither]
SPXL    81   5.638   0.733     1.276  0.997  0.020 False False  FAILS deflation [neither]
GLD    150   4.708   2.652     1.668  1.000  0.956 False False  REAL (survives best-of-N) [neither]
QLD     49   4.056   1.716     1.314  1.000  0.859 False False  FAILS deflation [neither]
USO     81   3.850   2.033     1.693  1.000  0.714 False False  FAILS deflation [neither]
UCO     58   3.506   1.712     1.123  0.998  0.836 False False  FAILS deflation [neither]
IAU     64   3.351   1.907     1.580  0.999  0.703 False False  FAILS deflation [neither]
TLT     48   3.315   0.949     0.838  1.000  0.656 False False  FAILS deflation [neither]
TNA     81   3.195   1.331     0.889  1.000  0.867 False False  FAILS deflation [neither]
SOXX    54   3.025   1.521     1.047  0.999  0.826 False False  FAILS deflation [neither]
IDV      6   2.676   1.712     0.856  1.000  0.963 False False  REAL (survives best-of-N) [neither]
UUP     84   2.540   1.275     1.370  0.984  0.436 False False  FAILS deflation [neither]
SPY     23   2.535   1.893     1.160  1.000  0.900 False False  marginal [neither]
AGQ     57   2.489   1.486     1.436  0.994  0.534 False False  FAILS deflation [neither]
GDX     64   2.478   1.642     1.108  0.997  0.812 False False  FAILS deflation [neither]
UGL     51   2.471   1.573     1.153  0.996  0.758 False False  FAILS deflation [neither]
ACWX     6   2.410   1.402     0.713  0.993  0.887 False False  FAILS deflation [neither]
DBA     11   2.382   1.524     0.726  0.996  0.914 False False  marginal [neither]
XME     54   2.370   1.884     1.184  0.999  0.888 False False  FAILS deflation [neither]
VT       6   2.017   1.601     0.915  0.996  0.875 False False  FAILS deflation [neither]
DJP     12   2.011   1.148     0.663  0.968  0.784 False False  FAILS deflation [neither]
DBC     23   1.927   1.222     0.889  0.981  0.713 False False  FAILS deflation [neither]
GSG     11   1.921   1.219     0.712  0.989  0.830 False False  FAILS deflation [neither]
VGK      6   1.920   1.490     0.919  1.000  0.982 False False  REAL (survives best-of-N) [neither]
EEM     12   1.908   1.659     1.289  0.999  0.746 False False  FAILS deflation [neither]
SLV     22   1.870   1.505     0.900  0.992  0.834 False False  FAILS deflation [neither]
EWY      6   1.806   1.483     0.597  0.992  0.925 False False  marginal [neither]
XOP     65   1.626   1.382     0.904  0.990  0.790 False False  FAILS deflation [neither]
SCZ      6   1.614   0.760     0.450  0.948  0.746 False False  FAILS deflation [neither]
EWG      5   1.562   1.216     1.393  0.998  0.340 False False  FAILS deflation [neither]
JNK      6   1.545   1.088     0.569  0.973  0.821 False False  FAILS deflation [neither]
VWO      7   1.528   0.996     0.734  0.968  0.687 False False  FAILS deflation [neither]
XLE     13   1.478   1.044     0.799  0.960  0.659 False False  FAILS deflation [neither]
XBI      8   1.475   1.175     0.786  0.995  0.802 False False  FAILS deflation [neither]
HYG     12   1.456   1.392     0.966  0.990  0.762 False False  FAILS deflation [neither]
EWZ      6   1.426   1.181     0.696  0.984  0.810 False False  FAILS deflation [neither]
EFA      6   1.385   1.192     0.594  0.977  0.841 False False  FAILS deflation [neither]
EWT      8   1.292   1.518     0.973  0.995  0.824 False False  FAILS deflation [neither]
QQQ     22   1.268   1.207     0.926  0.975  0.677 False False  FAILS deflation [neither]
IGE     12   1.244   1.150     0.663  0.967  0.782 False False  FAILS deflation [neither]
VYM      6   1.241   1.344     0.705  0.987  0.855 False False  FAILS deflation [neither]
VOX      6   1.153   1.211     0.670  0.991  0.855 False False  FAILS deflation [neither]
SPAB     6   1.130   0.972     0.808  0.997  0.677 False False  FAILS deflation [neither]
CMF      6   1.130   0.762     1.266  0.959  0.126 False False  FAILS deflation [neither]
FCG      9   1.059   0.826     0.496  0.917  0.710 False False  FAILS deflation [neither]
VEA      6   1.053   1.350     0.659  0.991  0.889 False False  FAILS deflation [neither]
SHV      6   1.050   1.351     0.717  0.990  0.863 False False  FAILS deflation [neither]
IYZ      6   1.040   0.788     0.441  0.940  0.753 False False  FAILS deflation [neither]
LQD      6   0.975   0.793     0.421  0.984  0.842 False False  FAILS deflation [neither]
SPDW     6   0.935   0.989     0.603  0.952  0.743 False False  FAILS deflation [neither]
ITB      6   0.906   0.834     0.501  0.924  0.716 False False  FAILS deflation [neither]
KRE     13   0.899   1.018     0.458  0.962  0.835 False False  FAILS deflation [neither]
IWM     32   0.881   0.910     0.672  0.936  0.655 False False  FAILS deflation [neither]
BIV      8   0.879   1.128     0.663  0.979  0.800 False False  FAILS deflation [neither]
IEO     11   0.875   0.784     0.814  0.902  0.480 False False  FAILS deflation [neither]
EWJ      6   0.875   0.773     0.911  0.902  0.408 False False  FAILS deflation [neither]
VPL      6   0.868   0.912     0.543  0.969  0.774 False False  FAILS deflation [neither]
DBB      7   0.867   0.583     0.779  0.841  0.368 False False  FAILS deflation [neither]
IOO      7   0.864   0.898     0.568  0.932  0.708 False False  FAILS deflation [neither]
ERX      5   0.858   0.801     0.447  0.906  0.720 False False  FAILS deflation [neither]
EZU      6   0.790   0.730     0.563  0.895  0.612 False False  FAILS deflation [neither]
TECL     5   0.771   1.067     0.766  0.964  0.694 False False  FAILS deflation [neither]
VEU      6   0.756   0.822     0.530  0.915  0.687 False False  FAILS deflation [neither]
UNG     53   0.729   0.671     0.770  0.876  0.433 False False  FAILS deflation [neither]
VNQ      7   0.698   0.639     0.445  0.905  0.655 False False  FAILS deflation [neither]
PEY      6   0.697   0.789     0.419  0.909  0.735 False False  FAILS deflation [neither]
SPXS    49   0.692   0.701     1.268  0.899  0.151 False False  FAILS deflation [neither]
TIP      7   0.680   0.787     0.544  0.913  0.663 False False  FAILS deflation [neither]
ACWI     5   0.649   0.465     0.379  0.794  0.560 False False  FAILS deflation [neither]
PSQ     51   0.602   0.735     1.129  0.973  0.151 False False  FAILS deflation [neither]
DDM      5   0.594   0.910     0.471  0.950  0.786 False False  FAILS deflation [neither]
FXF     12   0.586   0.623     0.350  0.858  0.680 False False  FAILS deflation [neither]
ICF      7   0.562   0.685     0.576  0.888  0.577 False False  FAILS deflation [neither]
MLN      5   0.545   0.398     1.675  0.781  0.006 False False  FAILS deflation [neither]
PFF      6   0.543   0.791     0.591  0.908  0.632 False False  FAILS deflation [neither]
FXE      8   0.518   0.576     0.317  0.836  0.670 False False  FAILS deflation [neither]
SPTL    52   0.516   0.445     1.163  0.771  0.115 False False  FAILS deflation [neither]
IYR      7   0.478   0.759     0.412  0.915  0.735 False False  FAILS deflation [neither]
BWX     10   0.465   0.594     0.893  0.852  0.299 False False  FAILS deflation [neither]
BSV      6   0.450   0.464     0.567  0.789  0.429 False False  FAILS deflation [neither]
REZ      6   0.446   0.548     0.350  0.814  0.626 False False  FAILS deflation [neither]
XES      8   0.425   0.548     0.296  0.819  0.662 False False  FAILS deflation [neither]
UWM      5   0.386   0.627     0.334  0.855  0.690 False False  FAILS deflation [neither]
SHM      5   0.326   0.444     0.235  0.770  0.636 False False  FAILS deflation [neither]
REM      6   0.289   0.336     0.327  0.714  0.506 False False  FAILS deflation [neither]
BLV      6   0.284   0.322     0.326  0.707  0.497 False False  FAILS deflation [neither]
PGX      6   0.273   0.454     0.801  0.779  0.278 False False  FAILS deflation [neither]
AGG      6   0.266   0.369     0.237  0.731  0.587 False False  FAILS deflation [neither]
VIXY     7   0.252   0.513     0.271  0.800  0.654 False False  FAILS deflation [neither]
FAS      5   0.250   0.289     0.163  0.687  0.584 False False  FAILS deflation [neither]
PGF      6   0.230   0.336     0.189  0.715  0.598 False False  FAILS deflation [neither]
RWR      6   0.219   0.312     0.228  0.703  0.557 False False  FAILS deflation [neither]
PCY      6   0.208   0.228     0.664  0.655  0.223 False False  FAILS deflation [neither]
EFV      5   0.203   0.213     0.220  0.639  0.496 False False  FAILS deflation [neither]
NLR     13   0.161   0.287     1.846  0.685  0.004 False False  FAILS deflation [neither]
RWO      7   0.155   0.273     0.537  0.676  0.329 False False  FAILS deflation [neither]
SPEM     6   0.147   0.153     0.083  0.601  0.547 False False  FAILS deflation [neither]
SH      51   0.144   0.237     1.097  0.660  0.067 False False  FAILS deflation [neither]
QID     49   0.122   0.147     1.306  0.598  0.025 False False  FAILS deflation [neither]
IGSB     6   0.115   0.175     0.093  0.621  0.557 False False  FAILS deflation [neither]
IEF      6   0.080   0.193     0.102  0.626  0.560 False False  FAILS deflation [neither]
MBB      6   0.065   0.089     0.048  0.560  0.528 False False  FAILS deflation [neither]
MOO     10   0.048   0.096     0.369  0.564  0.323 False False  FAILS deflation [neither]
SDS     49   0.023   0.064     0.842  0.543  0.093 False False  FAILS deflation [neither]
```

**Survive Holm-Bonferroni (FWER<=.05): BIL.** DSR>=.95 = edge clears the best-of-N-trials noise at the real session search size.

## Anytime-valid e-value monitor (peeking-robust; supersedes p-value/DSR re-checks)

H0: mean return <= 0 (edge dead). E-value >= 20 = significant at 0.05, VALID under continuous
monitoring (re-check anytime; merge re-validations by MULTIPLICATION). Testing-by-betting (WSR 2023).

```
champ    e-value  AV p=1/e        verdict  decay?
GLD         4.98    0.2010    weak (e>=1)  holding
UUP         1.31    0.7627    weak (e>=1)  holding
TIP         1.58    0.6324    weak (e>=1)  holding
DBC         1.26    0.7947    weak (e>=1)  holding
HYG         2.51    0.3982    weak (e>=1)  holding
```

Anytime-valid: unlike DSR/p-values, these e-values stay honest no matter how many times we re-check as the OOS window grows. Next re-validation just MULTIPLIES the new e-value in.
