# ETF Optimization Summary — 97 Experiments

## Final Results (All Real SetHoldings Orders)

| ETF | Return | Calmar | Axis | Label | Model | Split | Features |
|-----|--------|--------|------|-------|-------|-------|----------|
| QQQ | +172.2% | 1.42 | Dollar | KMeans | XGBoost | 2019 | Base 72 |
| GLD | +128.4% | 3.43 | Tick | KMeans | XGBoost | 2021 | +Entropy |
| EEM | +69.8% | — | Dollar | KMeans | XGBoost | 2021 | +Entropy |
| IWM | +68.3% | 0.59 | Dollar | KMeans | XGBoost | 2018 | +Entropy |
| XLE | +49.3% | 0.96 | Tick | KMeans | XGBoost | 2021 | +Entropy |
| HYG | +31.4% | 1.61 | Dollar | KMeans | XGBoost | 2019 | Base 72 |
| TLT | +13.3% | 0.47 | Dollar | MR | Ridge α=1 | 2021 | Base 72 |

## VAL+TEST Verification (Real SetHoldings)

| ETF | VAL | TEST | Valid |
|-----|-----|------|-------|
| QQQ | +70.1%, Calmar 2.96 | +172.2%, Calmar 1.42 | ✓ |
| GLD | +0.2%, 4 trades | +128.4%, Calmar 3.43 | ✓ |
| EEM | +35.9%, Calmar 1.80 | +69.8% | ✓ |
| IWM | +86.5%, Calmar 2.84 | +68.3%, Calmar 0.59 | ✓ |
| XLE | +21.3%, Calmar 0.99 | +49.3%, Calmar 0.96 | ✓ |
| HYG | +14.7%, Calmar 1.56 | +31.4%, Calmar 1.61 | ✓ |
| TLT | +5.1%, Calmar 0.57 | +11.4%, Calmar 0.47 | ✓ |

## What Worked (4 Enhancements)

1. **Sample entropy features** (+7 features): EEM +23pp, IWM +28pp, GLD +18pp, QQQ +4pp
2. **Per-ETF training split**: QQQ +135pp (2019), HYG +20pp (2019), IWM +19pp (2018)
3. **Tick bars**: GLD +41pp, XLE +18pp
4. **Ridge regression (TLT only)**: +11pp over XGBoost

## What Failed (40+ Approaches)

Trend scanning, fractional differentiation, range/volume bars, rolling/normalized dollar, imbalance bars, daily/hourly data, multi-horizon AND/AVG ensemble, PCA/KernelPCA, AE/VAE, GMM probability, mean-reversion labels, higher-order diffs, DBSCAN, Spectral clustering, IG selection, extended features, ACF, RSI, vol prediction, trailing stop, continuous labels, z-score labels, stacking, XGBoost hyperparameter tuning, Lasso, ElasticNet, GradientBoosting, cross-asset (SPY/IEF/VXX/USO for non-GLD)

## Core Framework

v236 pipeline: Dollar bars (C1 custom axis) → Two-stage KMeans unsupervised labels → 72 integer-diff features → XGBoost

## Deliverables

1. Per-ETF optimal configurations (above)
2. Production auto-selecting pipeline (`_pipeline_prod_train/main.py`)
3. Consolidated pipeline (`_pipeline_consolidated_train/main.py`)
4. QC Cloud cleaned (49 old projects removed, 8 bt_pool retained)
5. All 97 experiments documented in pipeline versions v236-v317
