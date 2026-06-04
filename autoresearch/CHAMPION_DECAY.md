# Champion alpha-decay / survival (formal, on REAL OOS return series)

Per-champion OOS equity extracted read-only from QC (`/backtests/chart/read`). Page-Hinkley
& CUSUM detect a DOWNWARD mean-shift (decay onset); early/late Sharpe is the half-window read.

```
GLD   npts= 223 ppy~ 78.9  early_SR=  1.84 late_SR=  2.30  PageHinkley=  5% in CUSUM=  5% in  -> HOLDING
```
- GLD: early→late Sharpe 1.84→2.30; downward change-point DETECTED.
```
UUP   npts= 223 ppy~ 78.9  early_SR=  2.67 late_SR=  0.74  PageHinkley= 10% in CUSUM= 10% in  -> STALE
```
- UUP: early→late Sharpe 2.67→0.74; downward change-point DETECTED.
```
IWM   npts= 223 ppy~ 78.9  early_SR=  0.63 late_SR=  1.31  PageHinkley=  6% in CUSUM=  6% in  -> HOLDING
```
- IWM: early→late Sharpe 0.63→1.31; downward change-point DETECTED.
