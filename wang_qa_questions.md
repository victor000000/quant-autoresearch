# 王老师 Q&A 准备

## 实验设置

19 只美股 ETF（QQQ/SPY/IWM/EFA/EEM/XLE/XLU/XLP/VNQ/AGG/TLT/SHY/TIP/HYG/GLD/DBC/UUP/VXX/EWJ），2009-08 至 2026-05 分钟数据。流水线按您课程 11 模块：

| Module | 实现 |
|---|---|
| 3 自定义轴 | LogDollar dollar bars |
| 4 标签（**无监督**） | sticky GaussianHMM-2 平滑后验 $\to$ 二元标签 |
| 5 特征工程 | 512 维（400 整数差分 + 32 frac-diff + 40 entropy + 40 freq/vol）|
| 6 数据降维 | VAE ($K{=}8$) |
| 7 模型训练（**有监督**） | XGBoost binary，early-stop on val |
| 10 OOS / 推理 | $\text{pos}[i] \cdot \log\text{ret}[i{+}1]$（leak-fix）|

train/val/oos $= 70/15/15$ 时序切分。

### 自定义轴（Module 3，LogDollar）

每分钟 $t$，$x_t = \log(P_t \cdot V_t)$，$w = 1950$ 滚动标准化：

$$z_t = \frac{x_t - \mu_t^{(w)}}{\sigma_t^{(w)}}, \quad C_t = \sum_{j \le t} z_j$$

发射 bar：$\max_j C_j - C_t \ge T$ 或 $C_t - \min_j C_j \ge T$；per-asset $T \in [13.2, 36.6]$；每 ETF 15-23k bars。

### 标签生成（Module 4，纯无监督）

label 阶段**只用无监督学习**生成标签——不涉及 features 或 supervised 部分。

TRAIN 上拟合 sticky GaussianHMM-2（$A_{kk} \ge 0.95$），观测序列 $o_t = \log\text{ret}_t$。EM 收敛后得参数 $\hat\theta_{\text{tr}} = (\hat\pi, \hat A, \hat\mu, \hat\Sigma)$。

将 $\hat\theta_{\text{tr}}$ 应用到全序列 $o_{1:N}$，做 forward-backward，得平滑后验：

$$\gamma_t^{\text{up}} \;=\; P\bigl(s_t = \text{up} \;\bigm|\; o_{1:N},\; \hat\theta_{\text{tr}}\bigr) \;=\; \frac{\alpha_t(\text{up}) \cdot \beta_t(\text{up})}{\sum_k \alpha_t(k) \beta_t(k)}$$

阈值 $\tau$ 给出二元 label：$y_t = \mathbb{1}\{\gamma_t^{\text{up}} > \tau\}$。

**Look-ahead 说明**：$\gamma_t^{\text{up}}$ 通过 $\beta_t$ 用到了 $o_{t+1:N}$（未来观测）。按您课上原话，**标签是训练目标，允许 look-ahead**；真正的交易信号是下游 supervised 模型在 OOS 上的因果输出（Module 7）。

### 单窗口 OOS 结果 (2024 — 2026)

| ETF | OOS AUC | Strat Cal | BH Cal | Alpha | | ETF | OOS AUC | Strat Cal | BH Cal | Alpha |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| GLD | 0.789 | **5.60** | 2.13 | **+3.48** | | TIP | 0.822 | 1.42 | 1.48 | $-0.06$ |
| HYG | 0.885 | 3.30 | 1.78 | +1.58 | | DBC | 0.784 | 1.06 | 1.10 | $-0.04$ |
| XLE | 0.857 | 2.00 | 0.92 | +1.05 | | EFA | 0.869 | 1.38 | 1.51 | $-0.13$ |
| TLT | 0.737 | 0.69 | $-0.13$ | +0.88 | | VNQ | 0.821 | 0.36 | 0.49 | $-0.13$ |
| EEM | 0.853 | 1.53 | 1.26 | +0.27 | | QQQ | 0.839 | 0.51 | 1.06 | $-0.55$ |
| IWM | 0.827 | 0.77 | 0.63 | +0.14 | | | | | | |

10 ETF 均值 alpha $\approx +0.66$；GLD 已达 Cal $> 5$。但 QQQ AUC 0.839 仍亏给 BH（alpha $-0.55$）；问题在 label/execution，不在 AUC。

\newpage

## Q1 (Module 3 自定义轴)：ETF peak\_frac 卡在 0.60-0.72，是 ETF 微观结构本质还是设计问题？

$\text{peak\_frac} := P(|r_{\text{bar}}| < 0.5\sigma)$；高斯基线 $= 0.383$。19 ETF LogDollar 测得：高流动 QQQ/SPY 0.60-0.65；中等 GLD/TLT/DBC 0.60-0.65；低流动 SHY/UUP/EWJ 0.65-0.72。8 种轴变体（LogDollar / PriceAction / RealVar / ZNorm + per-asset 校准）**没有一种**能同时让 peak\_frac $\approx 0.383$ 且 bar 数 $\ge 15$k。明显 ETF 微观结构：开盘/收盘集合竞价 + 做市商对冲 + 中间长时段 "calm-but-traded" 分钟。

请教：(1) 您框架对 calm-but-active 时段是否有专门处理？（时段过滤 / 价格-动量复合阈值） (2) 应追求 peak\_frac $\to 0.383$，还是接受 0.60+ 作为 ETF 本质？如果接受，下游 label 设计或特征生成需要怎么调整？

## Q2 (Module 4 标签 / 无监督)：HMM 平滑后验阈值 $\tau{=}0.5$ 在 drift-heavy ETF 上严重不平衡——平衡化又破坏 alpha——怎么解？

训练集 $n{=}200$-bar 远期回报正向频率 $\beta_n := P(r_n > 0)$ 与 HMM 标签均值 $\bar y_{tr}$：

| ETF | $\beta_{200}$ | $\bar y_{tr}^{\tau=0.5}$ | OOS Cal | BH Cal | Alpha |
|---|---:|---:|---:|---:|---:|
| QQQ | **0.74** | 0.73 | 0.51 | 1.06 | $-0.55$ |
| SPY | 0.74 | 0.72 | $-$ | 1.40 | $-$ |
| HYG | 0.71 | 0.65 | 3.30 | 1.78 | +1.58 |
| GLD | 0.53 | 0.55 | 5.60 | 2.13 | +3.48 |
| DBC | 0.52 | 0.52 | 1.06 | 1.10 | $-0.04$ |

形式化：设真实"上行 regime"$s_t \in \{0,1\}$，市场漂移 $\mu > 0$。则

$$\gamma_t^{\text{up}} \approx P(s_t{=}1) + \alpha_t \cdot \mathbb{E}[\text{sign drift} \mid \text{recent } o_t]$$

QQQ 上 $P(s_t{=}1) \approx 0.74$ 主导 $\Rightarrow$ $\gamma_t^{\text{up}} > 0.5$ 几乎处处 $\Rightarrow$ 标签严重 skewed $\Rightarrow$ "always 1" 基线本身就 73% accuracy。

**已测平衡化方案**（4 种，已验证）：(a) $\tau = $ median$_t \gamma_t^{\text{up,tr}}$ 强制 $\bar y_{tr} = 0.5$；(b) demean $o_t \leftarrow o_t - \hat\mu_{tr}$；(c) median-split forward $r_n$；(d) TB barrier 双向 k 校准。

**结果惨**：QQQ 平衡 $\tau$ 让 train AUC 0.92 但 OOS Cal 从 0.68 跌到 0.49；HYG 从 2.75 跌到 1.55；GLD 从 4.90 跌到 2.80。**所有平衡方案都 hurt**。

逻辑：在 bull regime 中，"mostly long" 本身就是 optimal；强制平衡标签反而让模型预测 short，破坏 drift capture。

请教：(1) 您 RB demo 5-9 阶 label 的 $\bar y_{tr}$ 大概多少？是否做 demean / 阈值平衡？(2) "label balance 约 0.5"（您原话）是 retraining 触发器还是 first-time training 的硬性要求？(3) drift-heavy ETF（$\beta_{200} > 0.70$）的 HMM regime label，您建议（a）保留 drift bias 但下游做 meta-label 过滤（您 Module 10）；（b）观测换成 $o_t = r_t - \hat\mu_{tr}$；（c）从 2-state 升到 3-state（up/range/down）；（d）其他？

## Q3 (Module 9 模型组合)：多尺度 HMM 简单平均在 ETF 上 hurts；您 RB 上 3.31 $\to$ 5.63 的简单 stacking 为何对 ETF 失效？

实现您 5/6/7/8/9 阶规范（ETF dollar-bar 上映射 $n \in \{50, 100, 200, 400, 800\}$）：每 $n$ 在 HMM 观测 $o_t^{(n)} = \log P_t - \log P_{t-n}$ 上独立训练（不同 $n$ 的 HMM 不同），各得 label，**独立** train 一个 XGBoost，OOS 概率简单平均 $\bar p_i = \frac{1}{5}\sum_n \hat p_i^{(n)}$。

GLD 上单尺度 vs ensemble：

| 尺度 $n$ | val AUC | OOS AUC | Cal |
|---:|---:|---:|---:|
| 50 | 0.899 | 0.916 | 2.97 |
| **100** | 0.647 | 0.614 | **4.76** |
| 200 | 0.597 | 0.516 | 0.40 |
| 400 | 0.552 | 0.331 | 1.94 |
| 800 | 0.609 | NaN | 0.80 |
| **简单 ens** | $-$ | $-$ | **2.23** |

$n{=}400$ AUC 跌到 0.33（**反预测**），$n{=}800$ 数据不足无法评估；低质量尺度纳入平均把 ens Cal 从 best 单尺度的 4.76 拖到 2.23。

您 RB 上 5 个尺度 Cal 都在 $[2.82, 3.31]$（同质 $\sigma_{\text{Cal}} \approx 0.18$）；ETF 跨度 $[0.40, 4.76]$（不同质 $\sigma_{\text{Cal}} \approx 1.6$）。

请教：(1) 您 RB 各尺度 OOS AUC 大约多少？同质度 $\sigma_{\text{AUC}}{<}0.05$ 吗？(2) 尺度质量异质时您 stacking 是简单平均，还是按 val AUC / val alpha 加权？(3) ETF 上"等价"的 5-9 阶选错了吗？您 RB 5-9 阶对应 1-9 天，ETF 是不是要更短（dollar-bar 的 $\{5, 10, 20, 40, 80\}$，对应小时-日级）？
