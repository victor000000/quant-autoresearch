# QC Cloud Python environment (authoritative, 2026-06-11)

Source: Lean `DockerfileLeanFoundation` (master) + QC docs + empirical probe. Python
**3.11.11**, CUDA 12.8 baked in. Same image for backtest/research/live. ~300 packages.

**Headlines for this project** (the "blocked on QC" assumption is dead — PROBE before
trusting any such claim):
- **DL**: torch 2.8.0 (+torchvision/lightning/pytorch-forecasting/torch-geometric),
  tensorflow 2.19.1, keras 3.13, jax 0.7.1, transformers 4.57, onnxruntime.
- **GBM zoo**: xgboost 3.0.5, lightgbm 4.6, catboost 1.2.8, ngboost 0.5.6.
- **Time-series**: statsmodels 0.14.6, **arch 8.0 (GARCH/FI-GARCH!)**, hmmlearn,
  **chronos-forecasting 2.2.2 (Amazon Chronos!)**, darts/gluonts/neuralforecast/
  statsforecast, **ruptures (changepoint)**, **stumpy 1.13 (matrix profile!)**,
  pymannkendall, EMD-signal, nolds, hurst, river (online learning).
- **Quant**: QuantLib 1.40, TA-Lib, pyportfolioopt, Riskfolio, QuantStats, py_vollib.
- **Topology/signal**: PyWavelets, scikit-tda/ripser/persim, **iisignature (path
  signatures)**, POT (optimal transport), KDEpy.
- **Causal/Bayes**: pymc, pyro, econml, causalml, tigramite, **copulas/copulae/
  pyvinecopulib**.
- **RL**: gymnasium, stable-baselines3 2.7, torchrl, ray[rllib].
- **Optimization**: cvxpy, optuna, hyperopt, deap, ray[tune].

**Nodes**: backtest B-MICRO(free 2c/8G) → B8-16 (8c/16G); **B4-16-GPU = 1/3 Tesla V100S**
($400/mo tier). Live L-MICRO..L8-16-GPU. Free tier: 200 backtests/day.

**Patterns that matter**: (1) train heavy models in Research (GPU), save weights to
ObjectStore, CPU-infer in backtest; (2) onnx-export encoders for fast CPU inference;
(3) version pinning is image-wide (no per-project pins; new-lib requests = 2-4 weeks);
(4) RAM (8-16G) is the binding constraint for in-backtest training.

Empirical probe (2026-06-11, our project): torch/lightgbm/catboost/ngboost/tensorflow/
shap/tsfresh/umap/cvxpy/statsmodels all import + torch tensor ops work (CPU).
