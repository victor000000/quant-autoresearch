# Autoresearch efficiency — systematic review v3 (2026-06-03)

Second deep review (re-run, deeper). Synthesized from a fan-out web-research harness: **110 primary-sourced
claims, 75 adversarial-verification verdicts (72 upheld / 3 refuted)**. Prioritizes what is NEW or supersedes
the prior two reviews and what we have already built. Ranked by **leverage ÷ implementation cost**.

## TL;DR — the one genuinely new high-leverage upgrade
**E-VALUES / anytime-valid inference for our continuous re-validation.** We re-test champions every few rounds
as the OOS window grows (decay checks, re-validation, "records go stale"). That is **optional stopping / data
peeking**, which *invalidates* p-value- and DSR-based gates (the reproducibility crisis is largely blamed on
exactly this). E-values/e-processes maintain Type-I validity under continuous monitoring (Ville's inequality),
are the *only* admissible method for anytime-valid inference + combining dependent tests, and **merge by simple
multiplication** with optional continuation. This is the formal fix for the staleness/peeking problem the prior
review flagged but did not solve.

---

## THREAD 1 — Causal vs provenance graph as the experiment ledger

**Verdict: "causal graph as log" is a TERMINOLOGY TRAP; our ledger is correctly a PROVENANCE graph.** (Confirms
the prior review.) Evidence:
- A causal DAG encodes *qualitative subject-matter assumptions about causal structure*, NOT statistical
  dependencies derived from data; its nodes are *random variables over a population*, not realized values or
  derived statistics. Treating a FINDING summary or a round-result as a "causal node" is a category error
  (primary: causal-inference methodology sources).
- Provenance/OPM graphs *can* be formally interpreted as deterministic structural causal models (Halpern-Pearl),
  but the authors explicitly do **not** claim they capture data-learned causality — the bridge is deterministic
  only. So the "causal" label overreaches; "provenance/derivation graph" is the honest term.
- Real tooling for this is provenance, not causal: **MLMD** (ML Metadata — run ledger, Events linking
  artifacts↔executions, recursive backward/forward lineage, Contexts for experiment-scoping, renders a DAG);
  **MLflow2PROV / PROV-AGENT** (W3C PROV, PROV-N, queryable via Neo4J/SPARQL, mergeable across projects;
  PROV-AGENT extends PROV for *agent decisions* in near-real-time).

**NEW, actionable — the executable-Knowledge-Graph (xKG) principle.** A research-agent ledger of Paper/Technique/
Code nodes performs best when **a technique is admitted ONLY if it is grounded in retrievable, executable code**;
ablation shows the **Code nodes are the single most load-bearing component** (removing them drops the score more
than removing Paper or doc nodes). Crucially: **semantic similarity is a *deceptive* retrieval signal** — agents
are misled by well-formatted, semantically-similar but technically-irrelevant knowledge.
→ *Apply:* keep our findings/methods grounded in runnable code (we do — every method is a registered module). And
distrust my own "this is like X" analogical retrieval: verify by *running*, not by resemblance. Benefit is
task-dependent (one paper +24%, another +2.6%), so don't over-invest in graph machinery uniformly.

## THREAD 2 — Efficient AI+human experiment selection

- **Multi-fidelity wins for fixed compute: BOHB** (Bayesian-opt × Hyperband) reaches a target ~**20× faster than
  Hyperband and ~55× faster than vanilla BO** on a 6-HP task, scales near-linearly with parallel workers, and uses
  a robust **TPE-style KDE** model (not a GP) for scalability/mixed spaces. → If we ever do return to HP search,
  this is the method — but see Thread 4 on why more search is rarely the bottleneck for us.
- **CONTRARIAN, supersedes "multi-agent is better": a SINGLE context-aware agent (Operand Quant) set the highest
  MLE-Bench medal rate (0.396), beating multi-agent/orchestrated systems under identical constraints.** A linear,
  non-blocking single agent consolidating explore→model→experiment→deploy. → **Validates our single-loop design;
  resist over-engineering multi-agent tournaments for the core loop** (reserve fan-out only for breadth tasks like
  this very review). The AI co-scientist (multi-agent generate-debate-evolve + tournament, test-time-compute
  scaling) and POPPER (agentic falsification with strict sequential Type-I control, matched experts at 10× less
  time) are useful *patterns*, not mandatory architecture.
- **CRITICAL caution on LLM hypothesis generation (Stanford 100+-reviewer study):** LLM ideas are judged *more
  novel* than expert humans (5.64 vs 4.84, p<0.05) — **but after execution the ranking FLIPS** (humans win);
  novelty-without-execution is unreliable; **LLMs cannot reliably evaluate ideas** (~53% agreement < human 56%);
  and **LLM ideation lacks diversity (~5% of 4000 ideas non-duplicate)**. → Two hard lessons for our loop: (1)
  **execution (real OOS Calmar) is the only arbiter — never trust my own novelty/quality judgment of a hypothesis;**
  (2) there's a low ceiling on distinct hypotheses I can generate per round, so don't equate "many ideas" with
  progress. Our design (A/B vs champion, gate on real backtest) is exactly right.

## THREAD 3 — Analogical transfer from clinical-trials / biostatistics

Strong support for what we built, plus two genuinely new imports:
- **Already-built & confirmed:** DSR (jointly corrects selection-bias-under-multiple-testing AND non-normality;
  E[max Sharpe] under N null trials is positive and grows with N; **the trial count N is "the single most
  important and universally omitted" backtest statistic** — our `honest_audit.py` uses the true session N).
  CPCV/Purged-CV is the most robust OOS method (purging prevents leakage) — we use PBO-via-CSCV. Harvey-Liu's
  **t>3.0 not 2.0** hurdle and **sequential orthogonalized selection** (test each new signal for *incremental*
  predictability) — we do Bonferroni + the haircut.
- **NEW #1 — E-VALUES / anytime-valid inference** (see TL;DR). The medical sequential-trials literature's modern
  successor to alpha-spending. Lets us monitor champions continuously without alpha inflation; e-values from
  follow-up re-validations **multiply** into a running evidence measure. *This is the formal cure for our
  "re-validate as the window grows" peeking.*
- **NEW #2 — Alpha-spending functions** (group-sequential): does **not** require pre-specifying the number or
  timing of interim looks; **data-driven peeking does not meaningfully inflate alpha** under common spending
  functions (O'Brien-Fleming spends little alpha early). → A lighter alternative to e-values for scheduling our
  re-validation gate; either one supersedes the current ad-hoc decay re-checks.

## THREAD 4 — The honest bottleneck: throughput or self-deception?

**Verdict: self-deception, decisively — and AI honesty harnesses are weak, so EXTERNAL ground truth is the only
reliable defense.** Evidence:
- **Automated AI-research-evaluated-by-AI loops are a demonstrated vulnerability.** LLM reviewers *detect*
  integrity problems yet still assign acceptance scores ("concern-acceptance conflict"); anti-fabrication
  detection harnesses **barely exceed random chance**; even with provably-sound score aggregation, integrity
  checking systematically fails → **layered defense-in-depth is required, correct math is not enough.** Real-world
  proof: 100 AI-hallucinated citations evaded peer review across 53 NeurIPS 2025 papers; 100% used *compound*
  deception; 66% were total fabrications. → **Our architecture is the right answer:** the QuantConnect backtest is
  *external, non-LLM ground truth*; we never let an LLM self-grade an edge. Keep it that way; never replace the
  real backtest with a model's self-assessment.
- **LLM-evolved trading-component studies** independently re-derive our playbook: **cap tunable parameters per
  candidate (~15-20)** as an anti-overfitting safeguard; require **multiple independent honesty tests** (sizing-
  decomposition vs pure-rescaling counterfactuals, scale-invariant metrics, structural-novelty) — mirrors our
  permute-control + replication + DSR; **overfitting worsens with task complexity** (joint feature+strategy >
  component-level); and **even successful evolved features yield tiny absolute predictive power (R²≈0.003-0.005)** —
  confirming durable single-asset alpha is intrinsically scarce. Search throughput is NOT the bottleneck.

---

## PRIORITIZED UPGRADES (leverage ÷ cost)

1. **[HIGH / MED] E-values (or alpha-spending) for continuous re-validation.** Replace ad-hoc "re-validate as the
   window grows" with an anytime-valid e-process per champion; accumulate evidence by multiplication; flag decay
   when the e-value crosses 1/α. Fixes the peeking that invalidates our current p-value/DSR re-checks. *The one
   genuinely new formal upgrade.*
2. **[HIGH / ZERO] Keep the single-agent loop; do not multi-agent-orchestrate the core.** Operand Quant + our own
   convergence say a single context-aware loop is best; reserve fan-out for breadth (reviews), not the edge hunt.
3. **[HIGH / ZERO] Execution is the only arbiter; distrust LLM novelty/quality judgments.** Stanford study: LLM
   novelty flips after execution; LLMs can't evaluate ideas; idea diversity is capped. Our gate-on-real-Calmar is
   correct — never crown on a model's say-so; expect few genuinely-distinct hypotheses per round.
4. **[HIGH / HAVE] External ground truth > AI honesty harness.** AI self-checks barely beat chance; the real QC
   backtest is our defense-in-depth. Never let an LLM self-grade an edge. (Already our architecture — protect it.)
5. **[MED / LOW] Executable-grounding + anti-semantic-similarity discipline.** Admit a method only if it's runnable
   code (we do); verify analogies by running, not by resemblance (semantic similarity is a deceptive signal).
6. **[MED / LOW] Audit config dimensionality (≤15-20 tunable params).** Cap the searchable CONFIG surface; we have
   ~9 dims (axis/label/thresh/sizing/depth/ncomp/permute/rebal_band/...) — near the safe ceiling, watch it.
7. **[LOW / LOW] Sequential orthogonalized selection.** When adding a new signal, test its *incremental*
   predictability over the held set (Harvey-Liu) — a refinement of our Bonferroni gate.

**Supersedes prior reviews:** (a) the "causal graph" framing is a confirmed misnomer → call it a provenance graph;
(b) e-values/anytime-valid inference is the new formal cure for staleness/peeking (prior review only named DSR/PBO);
(c) single-agent ≥ multi-agent for the core loop (don't build tournaments); (d) AI honesty harnesses are
empirically weak → lean entirely on external (backtest) ground truth.
