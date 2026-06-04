# Research Review v2 — Simplest High-Leverage Upgrades to the autoresearch Loop (2026-06-02)

Backbone (unchanged): **Wang's pipeline** — resample off the clock → label unsupervised → rich
features then reduce → combine across scales → bet-size; aim Calmar > 3, reproducible, deployable.

This v2 **builds on `RESEARCH_REVIEW.md`** (read that for the full literature map) and **corrects** it
with primary-source verification of the autoresearch ecosystem. The standing directive is **SIMPLE IS
BEST**: the prior review proposed a sprawling stack (TPE + ASHA + BOHB + MAP-Elites + embeddings +
Critic + AutoRA samplers + 9-edge typed graph). After verifying the actual implementations, most of that
is **over-engineering for a single-operator QC backtest loop**. The honest finding from our own work —
*durable single-asset ML alpha is scarce; only GLD trend_scan and UUP imbalance_bgm survive a
deflated-Calmar audit; the rest is decorrelated buy-hold (book Calmar ~3.3)* — tells us the bottleneck is
**not search throughput. It is fooling ourselves.** The highest-leverage changes therefore harden
honesty, not horsepower.

---

## TL;DR — the 3 changes that matter (do these; skip the rest)

1. **Harness extracts the metric; the agent never writes a number.** (Anti-fabrication.) ~50 lines.
2. **True Deflated Sharpe + PBO crown gate, computed from columns we already log.** Stops fake alpha. ~80 lines.
3. **Append-only, git-keyed `experiments.jsonl` with auto-computed delta vs last-kept.** One source of truth. ~40 lines.

Two optional, cheap follow-ons (Tier 2): a **20-line novelty/uncertainty picker** over `round_results.csv`
to replace manual enumerate-and-race, and a **CUSUM staleness flag** so we re-validate only suspect
champions. Everything else in v1 (embeddings, MAP-Elites archive, the AutoRA framework, the Rust CLI, the
9-edge typed graph, TPE+ASHA+BOHB) is **deferred or dropped** — see §6.

---

## 1. Verified mechanics of the autoresearch ecosystem (what's real, what to copy)

All repos below were checked against primary sources (GitHub trees, raw Rust files, READMEs). They exist
exactly as named and the mechanics are as described. v1's central thesis was right: **the real
implementations are more concrete and more copyable than v1 implied.** Corrections to v1 are flagged.

### 1a. `karpathy/autoresearch` (github.com/karpathy/autoresearch) — we already ARE this
Verified verbatim from the README. The loop: *"It modifies the code, trains for 5 minutes, checks if the
result improved, keeps or discards, and repeats."* `program.md` is *"essentially a super lightweight
'skill'."* The agent **touches exactly one file** (`train.py`); `prepare.py` is never modified. Metric =
`val_bpb` (lower is better, vocab-independent). Fixed **5-minute wall-clock box** (excludes
startup/compile). Throughput: *"~12 experiments/hour and approx 100 experiments while you sleep."*

This validates our design (per-round keep/discard, `program.md` steering, `per_etf_best`). **Correction to
v1:** the README contains **no numeric results** — the often-quoted nanochat figures (2.02h→1.80h, ~11%
TTGPT2, val_bpb −2.82%, AdamW-betas/depth-9/RoPE tweaks) are **secondary journalism** (Agent Wars, etc.),
not a verified benchmark. Do not cite them as a benchmark. **Correction to v1:** the canonical throughput
is **~100 experiments overnight (primary README)**, not "126" (that's a downstream CLI-blog figure).

### 1b. `autoresearch-cli` (github.com/199-biotechnologies/autoresearch-cli) — copy two ideas, not the binary
Source verified: `src/cmd/record.rs`, `src/git.rs`, and 17 command modules exist.
- **The ledger (`record.rs`):** append-only `.autoresearch/experiments.jsonl`. Each record holds run #,
  full git SHA + short hash, metric, **auto-computed delta** (`current_metric − prev_kept/baseline`), a
  canonicalized **status ∈ {baseline, kept, discarded}** (raw input like "keep"/"discard" is normalized),
  a summary, and a timestamp. **Correction to v1:** v1 listed 8 record fields; there is a 9th
  (`normalized_status`) — minor.
- **Anti-fabrication is a pipeline property, not magic:** the `doctor` pre-flight **runs the eval command
  and requires exit 0** (check #6), then **confirms the output contains a parseable number** (check #7,
  extracts the last whitespace token). The harness extracts the metric — *the agent cannot hand-write it.*
  **Correction to v1:** the README markets a "14-point" doctor, but the implemented `doctor.rs` has ~11–12
  checks, and anti-fabrication is **two checks (#6 runs eval, #7 parses metric)**, not one "check #6."
- **Skip:** the `flock(LOCK_EX)` parallel-agent race-safety exists for *many concurrent agents*. We race
  at most 2 nodes and `knowledge.json` already serializes writes. Do **not** port it — it solves a problem
  we don't have. Likewise skip the Rust binary, `cargo install`, and the 17 command modules; we already
  have `knowledge.json` + `round_results.csv`.

### 1c. AutoRA (`AutoResearch/autora`) — borrow the *idea*, not the framework
Verified: maintained by the Autonomous Empirical Research Group (Brown U / Schmidt Science Fellows),
published in JOSS vol 9 issue 104 (Dec 2024, DOI 10.21105/joss.06839). It ships Novelty, Uncertainty,
Model-Disagreement, Falsification, Leverage, and Mixture **experimentalist samplers** (plus Random/Grid/
Bandit). These are elegant and real. **But installing the framework is over-engineering for us.** The
transferable idea — *pick the next config by novelty/uncertainty instead of manually enumerating and
racing two* — is a **~20-line heuristic over `round_results.csv`** (§3, Tier 2). Do not adopt the package.

---

## 2. "Is the causal graph my experiment log?" — Yes, as a *provenance* log. Keep it minimal.

Short answer: **yes, our graph is a legitimate experiment/provenance log**, and this is an established
pattern (MLMD/TFX `Artifact`/`Execution`/`Context`; W&B `logged_by`/`used_by`; DVC/MLflow lineage; W3C
PROV-O verbs `used`/`wasGeneratedBy`/`wasDerivedFrom`; ORKG annotatable result triples; SciAgents/AI
Scientist experiment trees). v1's §1 covers the full lineage.

**The one correction worth acting on: the name.** Our edges encode *"hypothesis derived from finding"* =
**provenance/derivation (`prov:wasDerivedFrom`)**, NOT a Pearl causal effect (random-variable nodes,
arrows = population causal effects). Calling it a "causal graph" invites the exact correlation→causation
error our whole rigor program fights. **Rename to "provenance graph" / "derivation graph"** in docs and
code; reserve any literal causal claim for backtest-*intervention*-established relationships.

**Apply SIMPLE IS BEST to the graph itself.** v1 proposed a 5-part upgrade (typed MLMD schema + PROV
verbs + signed/weighted edges + `supersedes`/`as_of_round` + embeddings + subgraph retrieval). For a
188-node graph that one human + one agent reads, **only one of those earns its keep now**:

- **DO add `supersedes` edges + an `as_of_round` / `status` field on champion nodes.** This makes
  staleness first-class so we can **filter superseded / in-sample-only nodes before reasoning** — a direct,
  mechanical fix for the exact failure we paid for (EEM 4.03 → −0.02 was a stale record the loop kept
  trusting). Cheap, high-leverage.
- **DEFER** node embeddings + G-Retriever subgraph retrieval (a 188-node graph fits in context — retrieval
  infrastructure is premature), the full MLMD/PROV typed schema (nice, not load-bearing yet), and signed
  weighted `supports`/`refutes` edges (only worth it once we have enough conflicting evidence to aggregate).
- **Keep the dual representation we already have:** `round_results.csv` / `knowledge.json` flat table is
  the **source of truth for ranking**; the graph is the **lineage/reasoning view**. Do not move ranking
  into the graph — GraphRAG frequently loses to flat retrieval; our CSV already ranks correctly.

---

## 3. The three high-leverage changes, concretely

### Change 1 — Anti-fabrication: the harness extracts the metric (Tier 1, ~50 lines)
**Why:** the single most important transfer from the whole ecosystem. `karpathy/autoresearch` and
`autoresearch-cli` both enforce that **the loop runs the eval and parses the number; the agent never
writes it.** Our scorer is already LOCKED and deterministic, so we are close — but the discipline must be
explicit: a round's `real_calmar` (and `val_auc`, `trades`) must be **produced by the harness from the
backtest output**, and any agent-asserted metric is ignored. Adopt `autoresearch-cli`'s pattern: a
pre-flight that (a) runs infer and requires success, and (b) parses the metric from the run artifact,
refusing the round if either fails. This makes "you cannot fool the scorer" a structural guarantee, not a
convention.

### Change 2 — True Deflated Sharpe + PBO crown gate, from columns we already log (Tier 1, ~80 lines)
**Why:** this is the formal version of our hardest-won lesson. PBO would have **flagged EEM-4.03 ex ante.**
`scripts/assess_dsr.py` currently does PSR + Bonferroni-by-N_trials because v1 believed we lacked per-trial
Sharpes. **We don't.** `round_results.csv` already logs **`real_sharpe`, `real_skew`, `real_kurt`,
`n_days`, and `val_auc` per trial**, plus `ticker` for grouping. That is enough to compute:
- **Deflated Sharpe Ratio (Bailey & López de Prado 2014):** deflate by the expected-max-Sharpe of N
  *correlated* trials (use the variance of the logged per-trial Sharpes per asset) — tighter and more
  correct than Bonferroni.
- **PBO / CSCV (Bailey-Borwein-LdP-Zhu 2014):** is the crowned champion likely in-sample-overfit? **Refuse
  to deploy if PBO > ~0.5.**
- **Holm** (a free upgrade over Bonferroni) for the deploy gate; **Benjamini-Hochberg FDR** to rank
  proposals; **Harvey-Liu-Zhu t > ~3** (not 2) for a new edge.
`mlfinlab` ships PSR/DSR/PBO/CSCV; `statsmodels` ships Holm/BH. This is the change that most directly
prevents "stored Calmars go stale → we deploy noise."

### Change 3 — Append-only, git-keyed `experiments.jsonl` ledger with auto-delta (Tier 1, ~40 lines)
**Why:** `autoresearch-cli`'s `record.rs` is the right shape and trivially portable. Append each round as
one JSON line keyed to the **git SHA**, with run #, the cell (axis/labeler/thresh/sizing/ticker), the
harness-extracted metric (`real_calmar`, plus `val_auc`/`trades`), **auto-computed delta vs the asset's
last-kept**, and a canonical `status ∈ {baseline, kept, discarded}`. We already have `round_results.csv`;
this adds (a) immutable append-only history, (b) git-commit linkage so every record is reproducible to an
exact code state, and (c) auto-delta so "did it beat the prior best" is computed, not asserted. **Skip the
`flock` locking** (single operator) — just append under our existing serialized write.

---

## 4. Most-efficient AI + human methods (verified, and de-scoped to what fits)

The efficient-research literature (AI Scientist v2 tree search + staged budgets + dedup memory + max-debug
depth; Google AI co-scientist's Elo tournament where *the majority of compute is spent verifying, not
generating*; AlphaEvolve/MAP-Elites diversity archive; DSPy MIPROv2 minibatch-screen-then-full-eval;
Cerebras anti-cheating: *"infrastructure and task framing determined whether the agent explored
productively or spiraled"*) all point one way: **for a search this small, spend marginal compute on
verification and honest selection, not on a bigger optimizer.** Our own audit agrees — throughput isn't
the bottleneck; self-deception is.

Concretely, the most efficient AI+human division of labor for our loop:
- **Human (via `program.md`):** sets the objective, the deploy gate, and *which donor field to raid next*
  (§5). Steers rarely. This is exactly `karpathy/autoresearch`'s "humans steer only via program.md."
- **Harness (deterministic, un-foolable):** runs the backtest, extracts the metric, computes
  DSR/PBO/delta, refuses fabricated or overfit keeps. (Changes 1–3.)
- **Agent:** proposes the next cell and edits one module. Make proposal selection *slightly* smarter than
  manual enumeration with a **~20-line novelty/uncertainty picker** (Tier 2): over `round_results.csv`,
  prefer cells that are (a) under-sampled for the weakest asset (novelty) or (b) near the current frontier
  with high outcome variance (uncertainty). This captures AutoRA's best idea without the framework, and
  AI-Scientist's **dedup memory** for free (don't re-propose a tried cell).

**De-scoped (per SIMPLE IS BEST):** the full BOHB stack (TPE + ASHA + BOHB), MAP-Elites archive, Elo
tournament of hypotheses, and AutoRA package. Our space is ~tens of cells per asset and recipes don't
transfer across assets (`frontier-mapped`), so a surrogate optimizer's payoff is marginal versus its
maintenance cost. Revisit only if the search space grows by an order of magnitude.

---

## 5. Wang's "read medicine papers" habit — systematized, but kept lightweight

**Why it works (high-confidence, unchanged from v1):** Gentner structure-mapping (transfer *relational
structure*, not surface), Dunbar (distant analogies drive breakthroughs), Jeppesen-Lakhani 2010 (outsiders
solve what experts can't — 29–30% of stumped-R&D problems solved from outside the field), exaptation,
consilience. Trading and clinical trials share the same structural pains: **expensive samples, repeated
looks, multiplicity, confounding, selection/publication bias.** This is *why* the medicine reflex keeps
paying off — those fields professionalized exactly the rigor we keep rediscovering by hand.

**The habit, mechanized (5 cheap steps — this is the systematization):**
1. **Abstract** our current pain to its relational structure (e.g. "repeated peeks at a growing OOS window
   inflate false positives").
2. **Map to a ranked donor field** (clinical trials, epidemiology, reproducibility science, survival
   demography, online learning) and its anchor authors (Ramdas, Hernán, Benjamini, Berry, López de Prado).
3. **Score the candidate import:** does the *structure* truly match? what assumption does it need? what
   failure mode does it kill?
4. **A/B vs the champion before adoption** — we already do this and it works (sample-uniqueness HURT and
   was reverted; meta-labeling WON). This is the gate that keeps imports honest.
5. **Log the import + its A/B outcome** to the provenance graph.

The single most relevant unimported method for our exact staleness pain is **anytime-valid inference /
SPRT / e-values** (Wald 1947; Shafer 2021; Ramdas 2023) + **group-sequential α-spending** (O'Brien-Fleming
1979; Lan-DeMets): they let us **peek at rolling OOS Sharpe every bar without inflating error** and stop a
variant the instant evidence is conclusive — the principled cure for "the OOS window grows and stored
Calmars go stale." Pair with **survival analysis (Kaplan-Meier/Cox; McLean-Pontiff 2016 alpha decay)** to
model champion **half-life** and re-validate only the champions a Cox model flags (turnover/crowding/vol
covariates) — feeding Tier-2 Change below. Keep the rest of v1's donor table as a reference menu, not a
build list.

---

## 6. How to improve our efficiency — prioritized, ruthless about simplicity

**Tier 1 — Honesty (do now; these are the leverage):**
1. **Anti-fabrication:** harness runs eval + parses metric; agent-asserted numbers ignored. (§3, Change 1)
2. **True DSR + PBO crown gate** from the per-trial Sharpe/skew/kurt/n_days already in
   `round_results.csv`; Holm deploy gate; BH-FDR proposal ranking; refuse deploy if PBO > 0.5. (§3,
   Change 2)
3. **Git-keyed append-only `experiments.jsonl`** with auto-delta vs last-kept. (§3, Change 3)
4. **`supersedes` edges + `as_of_round`/`status`** so stale/in-sample nodes are filtered before reasoning;
   **rename "causal graph" → "provenance graph."** (§2)

**Tier 2 — Cheap efficiency (do if Tier 1 lands clean):**
5. **20-line novelty/uncertainty picker** over `round_results.csv` (+ dedup memory) to replace manual
   enumerate-and-race. (§4)
6. **CUSUM / Page-Hinkley staleness flag** on stored champion OOS returns (+ a simple Kaplan-Meier
   half-life) → re-validate only flagged champions, not all of them. (§5)
7. **Pre-registration stamp:** record hypothesis + config + metric + stopping-rule *before* each backtest
   (anti forking-paths). One line appended to the ledger.

**Tier 3 — Deferred / dropped as over-engineering (explicitly NOT now):**
- TPE + ASHA + BOHB surrogate optimizer; MAP-Elites diversity archive; Elo hypothesis tournament.
- AutoRA framework install; the autoresearch-cli Rust binary; `flock`/LOCK_EX race-safety.
- Node embeddings + G-Retriever subgraph retrieval; full MLMD typed schema; signed/weighted
  `supports`/`refutes` edges; multi-fidelity `fidelity`-knob racing.
  (All real and sound; none is the bottleneck for a single-operator loop of this size. Revisit only if the
  search space grows ~10× or a second operator/agent joins.)

**Guardrail (our scar tissue, retained):** triage cheap if you ever add multi-fidelity, but **CROWN only
on the full grown OOS window** — short windows lie (EEM proved it). And only real OOS Calmar > 0 that
beats buy-hold *and* `val_auc > 0.52` crowns a winner.

---

## Corrections to v1 (so they aren't lost)
- karpathy README has **no numeric results**; nanochat figures are secondary journalism — don't cite as a benchmark.
- Canonical throughput is **~100 experiments overnight** (primary README), not 126.
- `record.rs` has **9** fields (the 9th is `normalized_status`); v1 listed 8.
- `doctor` is **~11–12 checks**, not 14; anti-fabrication is **two checks (#6 runs eval, #7 parses
  metric)**, not one.
- We **already log per-trial Sharpe/skew/kurt/n_days** in `round_results.csv`, so true DSR needs no new
  data capture — v1 assumed otherwise.

## Sources (primary-verified where marked)
karpathy/autoresearch (README, raw-verified); 199-biotechnologies/autoresearch-cli (`src/cmd/record.rs`,
`src/git.rs`, `doctor.rs`, source-verified); AutoRA — AutoResearch/autora (JOSS DOI 10.21105/joss.06839,
verified); MLMD/TFX; W3C PROV-O; ORKG (orkg.org); SciAgents; AI Scientist v2 (arXiv 2504.08066); Google AI
co-scientist; AlphaEvolve / MAP-Elites; DSPy MIPROv2; Bailey & López de Prado DSR 2014 / PBO-CSCV 2014;
Harvey-Liu-Zhu 2016; Benjamini-Hochberg 1995; Wald 1947 / Shafer 2021 / Ramdas 2023; O'Brien-Fleming 1979 /
Lan-DeMets; Kaplan-Meier 1958 / Cox 1972 / McLean-Pontiff 2016; Gentner 1983 / Jeppesen-Lakhani 2010.
Our artifacts: `autoresearch/program.md`, `autoresearch/results/round_results.csv`,
`scripts/assess_dsr.py`, `autoresearch/knowledge.json`.
