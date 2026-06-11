# Systematic Review — Improving autoresearch Efficiency (2026-06-02)

Backbone = Wang's philosophy pipeline (resample off the clock → label unsupervised → rich features then reduce → combine across scales → bet-size; aim Calmar>3, reproducible, deployable). This review investigates how to run that pipeline *more efficiently and more honestly*, drawing on four research streams + our own codebase.

---

## 0. The headline: three of four streams independently converge on the same #1 gap

Our recent debugging session empirically rediscovered that **stored Calmars go stale and weak edges collapse** (EEM 4.03→−0.02). The literature says this is the *central* failure mode of a data-mining research loop, and names the exact fixes — which we are only ~70% implementing:

- **Multiple-testing control under selection** (Deflated Sharpe, PBO) — flagged independently by the *experiment-selection* stream AND the *medicine/biostatistics* stream.
- **Alpha decay / time-to-death** — survival analysis (medicine) + concept-drift detection (ML) + `supersedes` edges (graph). All three describe our staleness finding.
- **karpathy/autoresearch** — our loop *is* a re-implementation of it; the ecosystem around it tells us what to add (git-keyed ledger, anti-cheating, dedup memory).

So the roadmap below is not speculative — it's the formal version of lessons we already paid for.

---

## 1. "Can I treat the causal graph as my experiment log?" — Yes, but call it a *provenance graph*

**Established pattern, two mature lineages we are fusing:**
- *MLOps provenance/lineage graphs* (production-grade): **ML Metadata / MLMD** (TFX) — nodes `Artifact`/`Execution`/`Context`, edges `Event`/`Attribution`/`Association`, with "recurse to upstream inputs" traversal; **W&B artifact graph** (`logged_by`/`used_by`); **DVC/MLflow/Neptune/Aim** lineage DAGs; **W3C PROV-O** standard verbs (`used`, `wasGeneratedBy`, `wasDerivedFrom`, `wasAttributedTo`).
- *Scientific knowledge graphs + autonomous discovery* (research frontier): **ORKG** (annotatable triples for problems/methods/results + auto "Comparisons" = our leaderboard); **SciAgents** (33k-node KG, agents sample paths → hypotheses); **AI Scientist v2** (tree-of-experiments + experiment-manager next-step selection); **Robot Scientist Adam/Eve/Genesis** (the original closed loop with a formal-logic experiment log).

**The terminology trap (fix this):** our edges encode *"hypothesis derived from finding"* = **provenance/derivation (`prov:wasDerivedFrom`)**, NOT a Pearl causal effect. A real causal DAG has strict semantics (nodes = random variables, arrows = population causal effects). Mislabeling invites the classic correlation→causation error. **Action:** rename to *provenance/derivation graph* in code/docs; reserve a separate `causal_effect` edge only for intervention-established (backtest-proven) relationships.

**What our graph is missing vs best practice** (we have `{id,type,phase,label}` nodes + `{src,dst,label}` edges, 188/298):
1. **Typed schema** (MLMD types + PROV verbs) instead of free-form `label` edges → enables real queries.
2. **Signed, weighted `supports`/`refutes` edges** carrying effect size + OOS flag + sample size (ORKG annotatable edges) → a hypothesis's status = aggregate of weighted incoming evidence.
3. **`supersedes` edges + `as_of_round`/`status`** → staleness first-class; filter superseded / in-sample-only nodes *before* reasoning (directly fixes the stale-leaderboard problem).
4. **Node embeddings + subgraph retrieval** (G-Retriever: cosine-retrieve seed nodes, expand along edges) instead of feeding the whole graph to the LLM.
5. **Dual representation:** keep `knowledge.json`/`round_results.csv` flat table as source-of-truth for ranking (where MLflow/W&B win), project the graph view for lineage + reasoning (the MLflow2PROV pattern).

**Deriving the next experiment — stack three layers:** graph query (surface candidates: open hypotheses, stale champions, frontier leaves) → LLM-over-subgraph with an **adversarial Critic** (SciAgents Ontologist→Scientist→Critic; co-scientist Reflection) → **information-gain ranking** of candidates (causal active-learning: pick the experiment that most reduces frontier uncertainty).

Pitfalls: graph bloat (compress/cap retrieval), stale nodes (the `supersedes` fix), edge-soup (closed PROV vocabulary), over-trusting graph for ranking (GraphRAG often loses to flat RAG — A/B it).

---

## 2. The autoresearch landscape — our loop is `karpathy/autoresearch`; here's what to borrow

- **`karpathy/autoresearch`** (Mar 2026): ~100-line keep-or-`git reset` loop, LLM edits one file, fixed **5-min compute box**, reads one metric, humans steer only via `program.md`. **This is exactly our design** (program.md, per-round keep/discard, per_etf_best). Validates us.
- **`autoresearch-cli`** (Rust): append-only **`experiments.jsonl`** keyed to git commit (run#, metric, OOS metric, status, delta), 14-point `doctor` pre-flight, **metric parsed by the harness so the agent can't fabricate it**. → Adopt the ledger + anti-fabrication.
- **AutoRA** (`AutoResearch/autora`): formal **experiment-selection samplers** — Novelty (under-sampled), Uncertainty (high variance), Model-Disagreement, Falsification (probe boundaries), Leverage (max info-gain), Mixture. → This is the principled replacement for our manual "enumerate + race 2."
- **AI Scientist v2**: best-first **tree search** + experiment-manager; **staged budgets** (21/12/12/12 nodes), **max-debug-depth=3**, **1-hr/node cap**, **dedup memory** of tried configs, replication+aggregation nodes (mean±std, not point estimates).
- **Google AI co-scientist**: 7 agents; **Elo tournament** of pairwise hypothesis debates; *majority of compute spent verifying, not generating*. → Add a verification/Critic budget.
- **AlphaEvolve / MAP-Elites**: keep a **diversity-preserving archive** (best-per-behavior-bin), island model, resurface old elites → matches our "diversification beats concentration / keep weak names."
- **DSPy MIPROv2 / AutoML**: **minibatch screen → full eval**; meta-learned warm starts; ensembling beats single-best.
- **Anti-cheating** (Cerebras): accept only if metric ≥ baseline else revert; one experiment per call; isolated dir per run; stricter validation when the agent drifts. *"Infrastructure and task framing determined whether the agent explored productively or spiraled."*

---

## 3. Most efficient experiment-selection methods (target architecture)

**One-line target:** *LHS-seed → TPE-propose on a **deflated** objective → ASHA-race at low fidelity → full-eval survivors → DSR/PBO crown gate → CUSUM decay monitor + EI-based stop/switch.* (= BOHB + López-de-Prado multiplicity control + drift monitoring.)

- **Multi-fidelity triage [highest ROI]:** add a `fidelity` knob (near-OOS-window / fewer folds / shallower model) → race 4–6 cheap candidates, full-eval only survivors. ~halves compute/round. **Guardrail (our scar tissue): triage cheap, but CROWN only on the full grown window** (short windows lie — EEM proved it).
- **TPE surrogate (Optuna)** warm-started from `round_results.csv`, per-asset (recipes don't transfer per our `frontier-mapped`), conditional space (meta cells only under `triple_barrier*`). TPE/SMAC over GP because our space is categorical/conditional.
- **ASHA racing** (async, fits 2 nodes) along the OOS-window/fold rungs, promoting on our existing `val_auc>0.52` + interim-Calmar gates.
- **Search-stopping / method-switch:** when expected-improvement over untried cells collapses or after K DISCARDs, auto-switch module family (makes our "don't grind / new method" lesson mechanical).
- **LHS/Sobol seeding** when a new axis/labeler is added, so the surrogate isn't blind to unexplored regions.

**Tools:** Optuna (TPE + Hyperband/ASHA pruners), Ray Tune (async ASHA), Ax/BoTorch (batch q-EI), irace (F-race), mlfinlab (PSR/DSR/PBO/CSCV), statsmodels (Holm/BH/BY), scipy.stats.qmc (LHS/Sobol).

---

## 4. Multiple-testing & decay control — the mandatory finance upgrade (our `assess_dsr.py` is ~70% there)

- **Deflated Sharpe Ratio (Bailey & López de Prado 2014):** PSR with the benchmark = *expected max Sharpe of N independent trials*. Our `assess_dsr.py` does PSR+Bonferroni only because it lacks per-trial Sharpes — **start logging each trial's daily Sharpe to `round_results.csv`** → compute true DSR (tighter + correct for correlated trials).
- **Probability of Backtest Overfitting / CSCV (Bailey-Borwein-LdP-Zhu 2014):** is the selected champion likely in-sample-overfit? **PBO would have flagged EEM-4.03 ex ante.** Refuse to deploy if PBO > ~0.5.
- **Holm (free upgrade over Bonferroni)** for the deploy gate; **Benjamini-Hochberg FDR** to rank *proposals*. **Harvey-Liu-Zhu 2016:** a new edge needs **t > ~3**, not 2.
- **MinBTL:** given N trials, is our OOS even long enough to trust the max Sharpe?

---

## 5. Wang's "read medicine papers" habit — why it works, and the transferable methods

**Why (all high-confidence):** Gentner structure-mapping (transfer *relational structure*, not surface), Dunbar (distant analogies drive breakthroughs), **Jeppesen-Lakhani 2010** (outsiders solve what experts can't — solution probability *rises* with field distance; 29–30% of stumped-R&D problems solved by outsiders), exaptation (Gould-Vrba), consilience (Wilson). Trading and clinical trials share the *same structural pains*: expensive samples, repeated looks, multiplicity, confounding, publication/selection bias.

**Directly transferable methods → our loop:**
| Method | Source | Our use |
|---|---|---|
| **SPRT / e-values / anytime-valid inference** | Wald 1947; Shafer 2021; Ramdas et al. 2023; Johari 2022 | Peek on rolling Sharpe every bar without inflating error; stop a variant the instant evidence is conclusive; combine evidence by multiplying e-values |
| **Group-sequential boundaries** | O'Brien-Fleming 1979; Lan-DeMets α-spending | Pre-set OOS interim-look thresholds so an early lucky stretch ≠ "deploy" |
| **FDR / e-BH; DSR/PBO; t>3** | Benjamini-Hochberg 1995; Wang-Ramdas 2022; Bailey-LdP 2014; Harvey-Liu-Zhu 2016 | §4 — the crown/deploy gate |
| **Survival analysis (Kaplan-Meier, Cox)** | Kaplan-Meier 1958; Cox 1972; decay: McLean-Pontiff 2016 | Model strategy **alpha half-life**; Cox covariates (turnover, crowding, vol) predict *which* champions decay first → targeted re-validation (our staleness finding, formalized) |
| **Random-effects meta-analysis + funnel plots** | DerSimonian-Laird 1986; Duval-Tweedie 2000 | Pool per-asset Sharpe with shrinkage (vs cherry-picking the best); funnel-plot our own trials to self-diagnose selection bias |
| **Causal inference (DAGs, propensity, target-trial emulation)** | Pearl 2009; Rosenbaum-Rubin 1983; Hernán-Robins 2016 | Draw the DAG before a backtest (avoid conditioning on colliders); target-trial emulation disciplines against look-ahead/immortal-time bias (our leak history) |
| **Bayesian adaptive / RAR / n-of-1** | Berry 2010; Platform-of-1 2025 | Bandit/RAR allocation of compute toward rising-posterior arms; n-of-1 = per-asset within-subject on/off testing |
| **Pre-registration / forking-paths** | Simmons 2011; Gelman-Loken 2014 | Timestamp hypothesis+rules+stopping-rule before each backtest |

**Systematic scanning routine (the habit, mechanized):** (1) abstract our problem to its relational structure; (2) keep a ranked donor-field map (clinical trials, epidemiology, reproducibility science, survival demography, reliability/QC, online learning); (3) standing biweekly TOC scan of *Statistics in Medicine, Biometrics, Biometrika, Am. J. Epidemiology, Annals of Statistics* + anchor-author citations (Ramdas, Hernán, Benjamini, Berry, López de Prado); (4) score a candidate (does the structure truly match? what assumption does it need? what failure mode does it kill?); (5) **A/B vs champion before adoption** (we already do this — sample-uniqueness HURT, meta-labeling won); (6) log the import + its A/B outcome to the graph.

---

## 6. Prioritized roadmap for OUR system (ROI × low-cost × fit)

**Tier 1 — Rigor (stop fake alpha; the staleness lesson made mechanical):**
1. `assess_dsr.py`: log per-trial daily Sharpe → **DSR** (deflate by n_trials) + **PBO/CSCV** crown gate + **Holm** + **BH-FDR** for proposals. (`mlfinlab` has all four.)
2. **Decay monitor:** CUSUM/Page-Hinkley on stored champion OOS returns (+ Kaplan-Meier alpha half-life) → re-validate only flagged champions, not all.
3. **Pre-registration:** stamp hypothesis+config+metric+stopping-rule before each backtest (anti forking-paths).

**Tier 2 — Efficiency (more hypotheses per compute):**
4. **Multi-fidelity triage** (`fidelity` knob) + race 4–6 → full-eval survivors. Crown only on full window.
5. **TPE surrogate (Optuna)** warm-started from `round_results.csv`, optimizing the deflated objective → replaces manual enumerate-and-race in `target_next.py`.
6. **Search-stop / method-switch trigger** when EI collapses or K DISCARDs.

**Tier 3 — Graph as a real experiment log + reasoning substrate:**
7. Rename → provenance graph; typed nodes (MLMD) + PROV-O edge verbs; signed/weighted `supports`/`refutes`; `supersedes` + `as_of_round`.
8. Embeddings + subgraph retrieval + adversarial **Critic** step + info-gain candidate ranking (AutoRA samplers).
9. Append-only git-keyed `experiments.jsonl` ledger; dedup memory; MAP-Elites diversity archive binned by edge-type/asset.

**Tier 4 — Wang's cross-domain habit:** standing donor-field scan + A/B-vs-champion gate for every import (§5).

**Sources:** MLMD/PROV-O; ORKG (orkg.org); SciAgents (PMC12138853); AI Scientist v2 (arXiv 2504.08066); Robot Scientist (Nature 02236, arXiv 2408.10689); karpathy/autoresearch + autoresearch-cli; AutoRA (JOSS 10.21105/joss.06839); AlphaEvolve; DSPy MIPROv2; Optuna/Hyperband/ASHA/BOHB (Falkner 2018); Bailey & López de Prado DSR 2014 / PBO 2014; Harvey-Liu-Zhu 2016; Benjamini-Hochberg 1995; Wald 1947; Shafer 2021 / Ramdas 2023; O'Brien-Fleming 1979; Kaplan-Meier 1958 / Cox 1972; DerSimonian-Laird 1986; Pearl 2009 / Hernán-Robins 2016; Berry 2010; Gentner 1983 / Jeppesen-Lakhani 2010 / Hope-Kittur PNAS 2017.
