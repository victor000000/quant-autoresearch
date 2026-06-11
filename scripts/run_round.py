#!/usr/bin/env python3
"""Tournament driver (v2) — per-ETF, two-hypothesis, 2-node parallel.

THE IMPROVED LOOP (per the user). The 7 ETFs are naturally different, so each
round attacks the WEAKEST link instead of sweeping everything:

  1. read each ETF's current BEST real Calmar (active, G2-passing preferred)
     from autoresearch/knowledge.json (per_etf_best; seeded from `cells` if absent);
  2. rank the 7 ETFs and pick the LOWEST-performing one (the weakest link);
  3. take TWO competing hypotheses for that ETF and RUN THEM IN PARALLEL on the
     2 QC nodes (one each), then infer each cell;
  4. keep the winner iff it beats that ETF's current best on REAL OOS Calmar with
     DA reported and G2 trades>80; log BOTH;
  5. update knowledge.json per_etf_best, append results/round_results.csv, PRINT
     a clear comparison.

A HYPOTHESIS is a FULLY render-time-specified config (NO code edits between
hypotheses). It is the rendered QC script's header CONFIG:

    CONFIG = {"ticker": str, "axis": str, "labeler": str,
              "thresh": float, "sizing": str}

  axis    in bar_builder.AXES      (dollar,tick,vol,range,logdollar,entropy)
  labeler in labeler.LABELERS      (kmeans2stage,carry,tertile,bgm,
                                    agglomerative,triple_barrier,multi_horizon,
                                    hmm,always_long)
  thresh  per-ETF entry threshold  (e.g. 0.10-0.55)
  sizing  in {ramp, binary, cdf_plain, cdf_overlay}:
    ramp        = min(1,(p-thresh)*200) if p>thresh else 0          (legacy)
    binary      = 1.0 if p>thresh else 0.0       (most label-responsive: FLAT
                                                  when the model says not-up)
    cdf_plain   = clip(2*Phi((p-thresh)/sqrt(p(1-p)))-1,0,1) if p>thresh else 0
    cdf_overlay = cdf_plain * clip(std_slow/std_fast, 0.6, 1.0)  (vol-targeted)

CONFIG is injected purely at RENDER TIME by orchestrator.render_train_config,
which substitutes the header's five placeholders (no source edits). The footer
then enters HYPOTHESIS MODE — builds ONLY CONFIG['axis'], runs ONLY
CONFIG['labeler'] as ONE cell (no full sweep => fast, reproducible), sizes on VAL
via _size(CONFIG['sizing'], CONFIG['thresh']), and SAVES that sizing+thresh into
the cell payload. orchestrator.render_infer_cell renders infer bound to that
cell key; infer reads sizing+thresh back from the payload, so VAL (trainer) and
OOS (infer) use the IDENTICAL rule. Causality unchanged: fit on TRAIN only;
labels may use the future; features may not.

USAGE:
    # explicit (two JSON configs on argv):
    python3 scripts/run_round.py \
        '{"ticker":"TLT","axis":"vol","labeler":"tertile","thresh":0.45,"sizing":"binary"}' \
        '{"ticker":"TLT","axis":"range","labeler":"hmm","thresh":0.50,"sizing":"cdf_plain"}'

    # auto (pick weakest ETF, read its two hypotheses from the queue file):
    python3 scripts/run_round.py
      -> reads autoresearch/hypotheses.json = {"<TICKER>": [cfgA, cfgB], ...}

This driver does NOT write the HTML report and does NOT git-commit (the human/
opus does that per round). This codegen module executes NO backtest at import.

Do NOT run concurrently with the serial/axis drivers — two coordinators on one
QC project would race on main.py.
"""
import os
import sys
import csv
import json
import time
from datetime import datetime

# (1) qc_client lifecycle + orchestrator render functions (render a SINGLE-config
# train script given a CONFIG, plus render infer with the cell key). The package
# is installed (pip install -e .), so no sys.path manipulation is needed.
from lb.harness.qc_client import (submit_backtest, read_backtest_status, read_backtest,
                                  delete_backtest, is_done)
from lb.harness.orchestrator import render_train_config, render_infer_cell
from lb.harness.psuf import cell_suffix  # ONE canonical cell-key suffix (also injected into the QC header)
from lb.paths import (ROOT as PROJECT_ROOT, KNOWLEDGE_JSON, HYPOTHESES_JSON,
                      RESULTS_DIR, ROUND_RESULTS_CSV, STATUS_JSON)

from lb.describe import describe_cfg  # config -> plain-English hypothesis

CORE_7 = ["QQQ", "IWM", "EEM", "XLE", "HYG", "TLT", "GLD",
          "VIXY",  # NEW MECHANISM-CLASS (2026-06-04): VIX short-term futures ETF — structural variance-risk-premium CARRY (roll decay), a STRUCTURALLY DIFFERENT edge than trend/regime drift (FRONTIER re-opener). Short-capable (ls_overlay); permute control distinguishes carry-BETA from timing-ALPHA.
          "EFA",   # universe expansion: developed-intl equity (correlated w/ equity sleeve — excluded from book, R109)
          "DBC",   # broad commodities — DECORRELATED; improved the book 3.60->4.22 (R111)
          "UUP",   # US dollar — NEGATIVELY correlated w/ risk assets (risk-off hedge); decorrelation candidate
          "TIP",   # TIPS / inflation-linked bonds — real-rate/inflation exposure, decorrelated book member (R124 lever)
          "SLV",   # silver
          "SPY",   # S&P 500 — broad large-cap up-drifter (user request; equity-index complex w/ QQQ/IWM)
          "SOXX",  # semiconductors — CONFIRMED EDGE (ker+trend_scan 1.92, Bonferroni-significant): momentum-cyclical drawdowns are trend-predictable
          "XBI",   # biotech — event-driven (FDA) jumps -> trend labels can't time -> NO edge
          "KRE",   # regional banks — rate-trend edge but Bonferroni-FAIL -> provisional, NOT deployable
          "ITB",   # homebuilders — rate-cyclical, mechanism shows (DA 9x lower) but Bonferroni-FAIL -> provisional
          "SMH",   # VanEck semis — pre-registered REPLICATION of the SOXX semis edge (different fund, same sector)
          "XME",   # metals & miners — cyclical commodity producers; DIFFERENT sector test of the SOXX trend-predictable-drawdown mechanism (not a semis sibling)
          "FXY"]   # Japanese yen — NEW MECHANISM CLASS (FX policy-driven regime; Aug-2024 carry unwind): does UUP dollar-regime edge generalize to a distinct currency?

# KNOWLEDGE_JSON / HYPOTHESES_JSON / RESULTS_DIR / ROUND_RESULTS_CSV / STATUS_JSON
# are imported from lb.paths above (single source of truth, pathlib.Path objects).

# Universe-screening (2026-06-04, user "explore the 311 ETFs, find which fit"): allow any
# QC-CONFIRMED ETF (results/etf_qc_confirmed_pre2009.csv), not just CORE_7. Still a whitelist
# (typo-safe, QC-data-confirmed), just the full 311-name universe instead of the curated core.
def _confirmed_etfs():
    try:
        import csv as _csv
        p = os.path.join(RESULTS_DIR, "etf_qc_confirmed_pre2009.csv")
        return {r["Ticker"].strip() for r in _csv.DictReader(open(p)) if r.get("Ticker", "").strip()}
    except Exception:
        return set()
CONFIRMED_ETFS = _confirmed_etfs()


def _write_status(**kw):
    """Merge-update reports/status.json so the dashboard's live poller shows real
    phase progress. Preserves keys set elsewhere (e.g. round#) and overlays kw."""
    try:
        cur = json.load(open(STATUS_JSON)) if os.path.exists(STATUS_JSON) else {}
    except Exception:
        cur = {}
    cur.update(kw)
    try:
        json.dump(cur, open(STATUS_JSON, "w"))
    except Exception:
        pass


def _round_count():
    """Number of rounds already logged (distinct timestamps in round_results.csv). The DISPLAYED
    round number = this + 1 — incremented EVERY round (not just on KEEP, the old last_round bug that
    froze the dashboard at the last crown's round)."""
    try:
        seen = set()
        with open(ROUND_RESULTS_CSV) as f:
            for line in f:
                ts = line.split(",", 1)[0]
                if ts[:2] == "20":
                    seen.add(ts)
        return len(seen)
    except Exception:
        return 0

VALID_AXES = ["dollar", "tick", "vol", "range", "logdollar", "entropy", "imbalance", "tickimb", "volumeimb", "fracdiff", "dc", "zcusum", "kyle", "run", "spectral", "vpin", "jump", "volofvol", "wavelet", "amihud", "ddonset",
              # 2026-06-06 new-methods-backlog axes (mirror bar_builder._AXES_ORDER tail):
              "semivar", "chl", "diurnal", "kalman", "newma", "signedjumpvar",
              "vratio",   # new-methods 2026-06-09 A3: variance-ratio / price-efficiency clock
              "logdollar_rc",  # 2026-06-10 Wang de-scaled rolling-causal threshold (bar_ext.py)
              "sess2",  # 2026-06-10 session-anchored 2-bars/day clock (bar_ext.py, Wang frontier #4)
              "gapflow"]  # 2026-06-10 overnight-gap-weighted variance clock (bar_ext.py, invention round)
VALID_LABELERS = ["kmeans2stage", "tertile", "bgm", "agglomerative",  # carry disabled: QC runtime error, needs traceback to fix
                  "triple_barrier", "triple_barrier_tight", "triple_barrier_meta",
                  "triple_barrier_tight_meta", "triple_barrier_ae", "multi_horizon",
                  "regime_gmm", "cusum_regime", "jump_model", "dc_trend", "dc_reversal", "crash_ahead",
                  "trend_scan", "ker", "trend_leg", "accel", "sharpe_scan", "ofsc", "bde_cusum", "changepoint",
                  "tleg_fast", "tleg_mid", "tleg_slow", "ker_fast", "ker_mid", "ker_slow",
                  "calmar_scan", "sadf_explosive", "hurst_persist", "sliced_wasserstein", "sortino_scan", "transfer_entropy_dir", "visgraph", "mfe_mae", "revert", "turn_scan", "perment",
                  # 2026-06-06 new-methods-backlog labelers (mirror labeler.LABELERS):
                  "kllt", "diurnal_anomaly", "rskew", "icss_var", "bocpd_label", "setar", "tlb_reversal", "moe_law",
                  "dp_oracle",   # new-methods 2026-06-09 L1: cost-aware perfect-foresight oracle labeler
                  # "survival_aft" — deep-v2 B1 (built: labeler + footer survival:aft branch) but DISABLED:
                  # xgb.train(objective='survival:aft') C-CRASHES QC's XGBoost runtime (uncatchable "Runtime
                  # Error", empty logs), like the platform-blocked extratrees branch. Native model-objective
                  # innovations are QC-blocked; the code is kept (documented) but removed here to prevent crash-racing.
                  "hmm", "sticky_hmm", "always_long"]
VALID_SIZING = ["ramp", "binary", "cdf_plain", "cdf_overlay", "dd_overlay", "longshort", "ls_cdf", "ls_overlay", "crashveto", "ddbreaker", "cond_es"]
# aim/aim_dd/aim_cdf REMOVED from the raceable set (2026-06-10 leak-review LOW): the footer
# can SCORE them on VAL but infer.py has no aim branch -> a winner would be unreplayable
# (silent VAL/OOS rule mismatch). Zero trials ever raced them. Re-add only WITH the infer
# branch + aim_a persisted in the cell payload.

# 2-node pool params (reused from run_axis_label_parallel.py).
TIMEOUT = 300          # hard 5-min cap per backtest
POLL = 15              # seconds between status sweeps
MAX_INFLIGHT = 2       # QC node count

G2_MIN_TRADES = 80     # every DEPLOYABLE config must trade actively


def _now():
    return datetime.now().strftime("%H:%M:%S")


def _f(x, d=0.0):
    """Tolerant float parse ('12.3%', ' 4 ', None -> d)."""
    try:
        return float(str(x).replace("%", "").strip())
    except (ValueError, TypeError):
        return d


def _calmar_from_stats(st):
    cagr = _f(st.get("Compounding Annual Return", "0%"))
    mdd = _f(st.get("Drawdown", "0%"))
    return round(cagr / mdd, 4) if abs(mdd) > 0.01 else 0.0


def _cagr_from_stats(st):
    return round(_f(st.get("Compounding Annual Return", "0%")), 4)


def _mdd_from_stats(st):
    return round(_f(st.get("Drawdown", "0%")), 4)


def _count_trials(ticker):
    """How many configs have been tried for this ETF (multiple-testing N)."""
    try:
        import csv as _csv
        return sum(1 for r in _csv.DictReader(open(ROUND_RESULTS_CSV)) if r.get("ticker") == ticker)
    except Exception:
        return 1


def _survives_deflation(ticker, winner):
    """A SEARCHED edge must clear the best-of-N-trials Calmar noise (Bailey-LdP deflation):
    with N configs tried, the max is upward-biased, so the winner must exceed E[max] of N
    iid-noise trials. always_long baselines carry no selection bias -> exempt. <3 trials -> allow.
    Returns (ok, benchmark, n)."""
    if (winner.get("labeler") or "").startswith("always_long"):
        return True, 0.0, 0
    try:
        import csv as _csv
        cals = [float(r["real_calmar"]) for r in _csv.DictReader(open(ROUND_RESULTS_CSV))
                if r.get("ticker") == ticker and r.get("real_calmar") not in ("", None)]
    except Exception:
        cals = []
    cals = cals + [_f(winner.get("real_calmar", 0.0))]      # include this round
    n = len(cals)
    if n < 3:
        return True, 0.0, n
    mean = sum(cals) / n
    var = sum((c - mean) ** 2 for c in cals) / (n - 1)
    try:
        from stats_rigor import expected_max_sharpe
        bench = expected_max_sharpe(var, n)
    except Exception:
        return True, 0.0, n
    return (_f(winner.get("real_calmar", 0.0)) > bench), round(bench, 4), n


def _psr_significance(sharpe, skew, kurt, n_days, n_trials):
    """PSR(SR>0) with skew/kurt correction + a Bonferroni-by-N_trials threshold.
    Returns (psr, significant). The trials-adjusted 'is this edge real?' test."""
    import math
    if not (n_days and n_days > 3 and sharpe):
        return None, None
    srd = float(sharpe) / math.sqrt(252.0)                  # per-observation Sharpe
    denom = math.sqrt(max(1e-9, 1.0 - float(skew) * srd + (float(kurt) - 1.0) / 4.0 * srd * srd))
    psr = 0.5 * (1.0 + math.erf((srd * math.sqrt(n_days - 1) / denom) / math.sqrt(2.0)))
    thr = 1.0 - 0.05 / max(1, int(n_trials))
    return round(psr, 4), bool(psr > thr)


def _trades_from_stats(st):
    raw = str(st.get("Total Orders", "0")).strip()
    try:
        return int(float(raw))
    except (ValueError, TypeError):
        return 0


# ===========================================================================
# 2-NODE POOL (reused from run_axis_label_parallel.py)
# ===========================================================================
def run_pool(jobs):
    """jobs: list of (label, code) OR (label, code, extra_files). Keeps <=MAX_INFLIGHT
    backtests RUNNING at once. submit_backtest blocks through create (serializes main.py
    safely); polling then overlaps the run phase across both nodes. 'no spare nodes' is
    TRANSIENT — the job is re-queued and we wait for a node to free (300s cap per backtest).
    extra_files (e.g. a separate bar_builder.py) are uploaded alongside main.py.
    Returns {label: backtest_result_dict}."""
    results, inflight, pending, retries = {}, {}, list(jobs), {}
    while pending or inflight:
        while pending and len(inflight) < MAX_INFLIGHT:
            job = pending.pop(0)
            label, code = job[0], job[1]
            extra = job[2] if len(job) > 2 else None
            print(f"[{_now()}] SUBMIT {label} (inflight={len(inflight)+1}/{MAX_INFLIGHT}, pending={len(pending)})")
            try:
                bid = submit_backtest(code, label, extra_files=extra)   # upload->compile->create (serial, safe)
                inflight[label] = (bid, time.time())
            except Exception as e:
                msg = str(e)
                low = msg.lower()
                # TRANSIENT submit failures -> re-queue + wait (bounded). 'spare node'
                # (no free node) and 'compile id not found' (QC compile-cache miss at
                # create) are both flaky and clear on retry.
                transient = (("spare node" in low) or ("compile id not found" in low)
                             or ("could not find a part of the path" in low))  # QC build-cache miss
                if transient and retries.get(label, 0) < 4:
                    retries[label] = retries.get(label, 0) + 1
                    pending.insert(0, job)
                    print(f"[{_now()}]   {label} transient submit error (retry {retries[label]}/4): {msg[:90]} — re-queue + wait")
                    time.sleep(8)
                    break
                results[label] = {"status": "crash", "error": msg}
                print(f"[{_now()}]   {label} submit CRASH: {msg[:160]}")
        done = []
        for label, (bid, t0) in inflight.items():
            status, _, _ = read_backtest_status(bid)
            if is_done(status):
                results[label] = read_backtest(bid) if status.startswith("Completed") else {"status": status}
                print(f"[{_now()}]   {label} -> {status}")
                done.append(label)
            elif time.time() - t0 > TIMEOUT:
                delete_backtest(bid)
                results[label] = {"status": "timeout"}
                print(f"[{_now()}]   {label} -> TIMEOUT (deleted)")
                done.append(label)
        for label in done:
            del inflight[label]
        if inflight or pending:
            time.sleep(POLL)
    return results


# ===========================================================================
# KNOWLEDGE: per-ETF best (seed from existing cells if absent), rank, pick lowest
# ===========================================================================
def _load_knowledge():
    if os.path.exists(KNOWLEDGE_JSON):
        with open(KNOWLEDGE_JSON) as f:
            return json.load(f)
    return {}


def _save_knowledge(k):
    with open(KNOWLEDGE_JSON, "w") as f:
        json.dump(k, f, indent=2, default=str)


def _append_causal_round_node(target, winner, kept, prev_cal):
    """Append ONE round node (+ one edge from the prior round) to
    knowledge.json.causal_graph so the graph keeps pace with the CSV ledger.
    Idempotent on id; fully guarded — never raises into the round.

    The node id is r<n> where n = the chronological round number == count of
    distinct timestamps in round_results.csv (the same numbering the live ledger
    uses). The richer hand-curated FINDING hubs are still added by a human; this
    just keeps the round-trail current automatically."""
    try:
        # round number = distinct timestamps already logged (this round is logged
        # immediately before this call, so the latest ts is included).
        seen = []
        try:
            with open(ROUND_RESULTS_CSV, newline="") as f:
                for r in csv.DictReader(f):
                    ts = r.get("timestamp", "")
                    if ts and ts not in seen:
                        seen.append(ts)
        except Exception:
            pass
        if not seen:
            return
        with open(KNOWLEDGE_JSON) as f:
            k = json.load(f)
        cg = k.setdefault("causal_graph", {})
        cg.setdefault("nodes", [])
        cg.setdefault("edges", [])
        ids = {nd.get("id") for nd in cg["nodes"]}
        # collision-proof monotonic id: continue PAST the highest existing rN node
        # (the CSV round count is a different universe than the historical html numbering).
        rnums = [int(i[1:]) for nd in cg["nodes"]
                 for i in [str(nd.get("id", ""))] if i.startswith("r") and i[1:].isdigit()]
        n = (max(rnums) + 1) if rnums else len(seen)
        nid = f"r{n}"
        if nid in ids:
            return                                  # idempotent — already present
        verdict = "KEEP" if kept else "DISCARD"
        wc = (f'{winner["real_calmar"]:+.4f}' if winner else "—")
        recipe = describe_cfg(winner) if winner else ""
        label = (f"R{n} {target} {verdict}: {recipe} -> Calmar {wc} "
                 f"(prev best {prev_cal:+.4f}). Auto-logged from round_results.csv.")
        node = {"id": nid, "type": ("milestone" if kept else "round"),
                "phase": target, "label": label}
        cg["nodes"].append(node)
        # link from the immediately-prior round node if one exists
        prev_id = f"r{n-1}"
        if prev_id in ids:
            cg["edges"].append({"src": prev_id, "dst": nid,
                                "label": ("new best" if kept else "no improvement")})
        with open(KNOWLEDGE_JSON, "w") as f:
            json.dump(k, f, indent=2, default=str)
    except Exception as e:
        print(f"[warn] causal-graph round node not appended: {e}")


def _refresh_report():
    """Regenerate the static reports/index.html via render_index.build_html() so
    the dashboard is current the moment a round finishes (the Flask app also renders
    live per-request, but this keeps the on-disk file fresh for static hosting).
    Fully guarded — a render failure must NEVER fail a round."""
    try:
        from lb.console import render_index
        html = render_index.build_html()
        out = os.path.join(PROJECT_ROOT, "reports", "index.html")
        with open(out, "w") as f:
            f.write(html)
        print(f"[{_now()}] report refreshed -> {out} ({len(html)} bytes)")
    except Exception as e:
        print(f"[warn] report auto-refresh skipped: {e}")


def _seed_per_etf_best(knowledge):
    """Derive a per-ETF best from the legacy `cells` dict when per_etf_best is
    absent. PREFER an active (trades>80) G2-passing cell; if none, fall back to
    the highest-Calmar cell of any kind (buy-hold ceilings included, marked).

    A cell key is '<TICKER>_<rest>' (e.g. GLD_vol_carry, QQQ_dollar_R5)."""
    cells = knowledge.get("cells", {}) or {}
    by_etf = {t: [] for t in CORE_7}
    for key, c in cells.items():
        tk = key.split("_", 1)[0]
        if tk in by_etf and isinstance(c, dict):
            by_etf[tk].append((key, c))

    per = {}
    for tk in CORE_7:
        rows = by_etf[tk]
        if not rows:
            per[tk] = {"cell": None, "real_calmar": 0.0, "real_da": None,
                       "trades": 0, "g2_pass": False, "round": 0, "source": "none"}
            continue
        active = [(k, c) for (k, c) in rows if int(c.get("trades", 0) or 0) > G2_MIN_TRADES]
        pool = active if active else rows
        bk, bc = max(pool, key=lambda kc: _f(kc[1].get("real_calmar", 0.0)))
        per[tk] = {
            "cell": bk,
            "real_calmar": _f(bc.get("real_calmar", 0.0)),
            "real_da": bc.get("real_da"),
            "trades": int(bc.get("trades", 0) or 0),
            "g2_pass": int(bc.get("trades", 0) or 0) > G2_MIN_TRADES,
            "round": bc.get("round", 0),
            "source": "active_seed" if active else "buyhold_seed",
        }
    return per


def get_per_etf_best(knowledge):
    """Return per_etf_best (seeding into knowledge if absent)."""
    per = knowledge.get("per_etf_best")
    if not per:
        per = _seed_per_etf_best(knowledge)
        knowledge["per_etf_best"] = per
    return per


def pick_weakest(per_etf_best):
    """Rank the 7 ETFs by current best Calmar (then by G2-pass) and return the
    LOWEST (the weakest link). A buy-hold ceiling (G2 fail) is not deployable, so
    an equal-Calmar ETF lacking an active best sorts as weaker."""
    def rank_key(tk):
        b = per_etf_best.get(tk, {})
        return (_f(b.get("real_calmar", 0.0)), 1 if b.get("g2_pass") else 0)
    ranked = sorted(CORE_7, key=rank_key)
    return ranked[0], ranked


# ===========================================================================
# HYPOTHESIS loading + validation
# ===========================================================================
def _validate_cfg(cfg):
    req = ["ticker", "axis", "labeler", "thresh", "sizing"]
    for k in req:
        if k not in cfg:
            raise ValueError(f"CONFIG missing key '{k}': {cfg}")
    if cfg["ticker"] not in CORE_7 and cfg["ticker"] not in CONFIRMED_ETFS:
        raise ValueError(f"ticker {cfg['ticker']!r} not in CORE_7 nor the {len(CONFIRMED_ETFS)} QC-confirmed ETFs")
    if cfg["axis"] not in VALID_AXES:
        raise ValueError(f"axis {cfg['axis']!r} not in {VALID_AXES}")
    # labeler may be a single name OR a "+"-joined ENSEMBLE (⑦), e.g. "triple_barrier+bgm".
    for _lp in str(cfg["labeler"]).split("+"):
        if _lp not in VALID_LABELERS:
            raise ValueError(f"labeler part {_lp!r} (of {cfg['labeler']!r}) not in {VALID_LABELERS}")
    if cfg["sizing"] not in VALID_SIZING:
        raise ValueError(f"sizing {cfg['sizing']!r} not in {VALID_SIZING}")
    if cfg.get("horizons") is not None:   # optional INTRADAY-holding override (forward-label bars)
        hz = cfg["horizons"]
        # h=1 is the naive next-bar label Wang warns against on NOISE clocks — but on the
        # session-anchored sess2 clock (2 bars/day) 1 bar = a 30-min structural segment
        # (the last-half-hour session-momentum target), so allow it there only.
        _hmin = 1 if cfg.get("axis") == "sess2" else 2
        if not (isinstance(hz, list) and 1 <= len(hz) <= 4 and all(isinstance(h, int) and _hmin <= h <= 400 for h in hz)):
            raise ValueError(f"horizons {hz!r} must be a list of 1-4 ints in [{_hmin},400] (forward-label bars)")
    cfg["thresh"] = float(cfg["thresh"])
    if not (0.0 < cfg["thresh"] < 1.0):
        raise ValueError(f"thresh {cfg['thresh']} must be in (0,1)")
    cfg["max_depth"] = int(cfg.get("max_depth", 3))   # optional model-capacity override; default 3
    if not (2 <= cfg["max_depth"] <= 8):
        raise ValueError(f"max_depth {cfg['max_depth']} must be in [2,8]")
    cfg["permute_labels"] = bool(cfg.get("permute_labels", False))   # optional falsification control
    cfg["n_components"] = int(cfg.get("n_components", 20))            # optional reducer-width lever (default 20)
    if not (5 <= cfg["n_components"] <= 60):
        raise ValueError(f"n_components {cfg['n_components']} must be in [5,60]")
    cfg["reduce"] = str(cfg.get("reduce", "correlation"))            # Wang ④ dim-reduce lever (default correlation)
    if cfg["reduce"] not in ("correlation", "infogain", "variance", "autoencoder", "mrmr", "pca", "ae_np"):
        raise ValueError(f"reduce {cfg['reduce']!r} must be correlation|infogain|variance|autoencoder|mrmr")
    cfg["rebal_band"] = float(cfg.get("rebal_band", 0.01))           # optional net-of-cost dead-band lever (default 0.01)
    if not (0.0 <= cfg["rebal_band"] <= 0.20):
        raise ValueError(f"rebal_band {cfg['rebal_band']} must be in [0.0,0.20]")
    cfg["features"] = str(cfg.get("features", "base"))               # feature-set lever (base|rich|termstruct|evt|disp|sig|realyield)
    if cfg["features"] not in ("base", "rich", "termstruct", "evt", "disp", "sig", "realyield", "wangrich", "oilbasis"):
        raise ValueError(f"features {cfg['features']!r} must be base|rich|termstruct|evt|disp|sig|realyield|wangrich|oilbasis")
    return cfg


def load_hypotheses(argv, weakest):
    """TWO hypothesis CONFIGs from argv (two JSON strings) OR the per-ETF queue
    file autoresearch/hypotheses.json = {"<TICKER>": [cfgA, cfgB], ...}.

    `weakest` is the auto-picked ETF; reading the queue file uses its two configs.
    With argv we trust each config's own 'ticker' (and require BOTH to name the
    same ETF — a round pits two hypotheses for ONE ETF)."""
    if len(argv) >= 2:
        a = _validate_cfg(json.loads(argv[0]))
        b = _validate_cfg(json.loads(argv[1]))
        if a["ticker"] != b["ticker"]:
            raise ValueError(f"both hypotheses must target ONE ETF; got {a['ticker']} vs {b['ticker']}")
        return a["ticker"], a, b
    if os.path.exists(HYPOTHESES_JSON):
        with open(HYPOTHESES_JSON) as f:
            q = json.load(f)
        if weakest not in q or len(q[weakest]) < 2:
            raise ValueError(f"hypotheses.json has no >=2 configs for weakest ETF {weakest}; "
                             f"keys={list(q.keys())}")
        a = _validate_cfg(dict(q[weakest][0], ticker=weakest))
        b = _validate_cfg(dict(q[weakest][1], ticker=weakest))
        return weakest, a, b
    raise ValueError("no hypotheses: pass two JSON configs on argv OR create "
                     f"{HYPOTHESES_JSON} = {{'<TICKER>': [cfgA, cfgB]}}")


# ===========================================================================
# RESULTS extraction
# ===========================================================================
def _extract_result(name, train_bt, infer_bt, cfg):
    """Combine train (synth/AUC) + infer (REAL OOS Calmar/DA/trades) into one row."""
    rt_t = (train_bt.get("runtimeStatistics", {}) or {}) if isinstance(train_bt, dict) else {}
    train_status = "completed" if str(train_bt.get("status", "")).startswith("Completed") else train_bt.get("status", "?")

    st = (infer_bt.get("statistics", {}) or {}) if isinstance(infer_bt, dict) else {}
    rt_i = (infer_bt.get("runtimeStatistics", {}) or {}) if isinstance(infer_bt, dict) else {}
    infer_status = "completed" if str(infer_bt.get("status", "")).startswith("Completed") else infer_bt.get("status", "?")

    real_calmar = _calmar_from_stats(st) if st else 0.0
    real_cagr = _cagr_from_stats(st) if st else 0.0      # Compounding Annual Return (%)
    real_mdd = _mdd_from_stats(st) if st else 0.0        # Max Drawdown (%)
    trades = _trades_from_stats(st) if st else 0
    # REAL OOS DA: infer echoes da_oos as a runtime stat (fallback val_da).
    real_da = _f(rt_i.get("da_oos", rt_i.get("val_da", 0.0)))
    da_present = ("da_oos" in rt_i) or ("val_da" in rt_i)
    # OOS daily Sharpe + higher moments (for PSR / Deflated-Sharpe trials-adjustment).
    real_sharpe = _f(rt_i.get("sharpe_oos", 0.0))
    real_skew = _f(rt_i.get("skew_oos", 0.0))
    real_kurt = _f(rt_i.get("kurt_oos", 0.0))
    n_days = int(_f(rt_i.get("n_days", 0.0)))

    return {
        "name": name,
        "ticker": cfg["ticker"],
        "axis": cfg["axis"],
        "labeler": cfg["labeler"],
        "thresh": cfg["thresh"],
        "sizing": cfg["sizing"],
        # optional levers carried through so a CROWN records its full reproducible config
        **{k: cfg[k] for k in ("max_depth", "n_components", "rebal_band", "reduce",
                               "permute_labels", "horizons", "features") if k in cfg},
        "real_calmar": real_calmar,
        "real_cagr": real_cagr,
        "real_mdd": real_mdd,
        "real_da": real_da,
        "real_sharpe": real_sharpe,
        "real_skew": real_skew,
        "real_kurt": real_kurt,
        "n_days": n_days,
        "da_present": da_present,
        "trades": trades,
        "g2_pass": trades > G2_MIN_TRADES,
        "synth_cal": _f(rt_t.get("best_cal", rt_i.get("synth_cal", 0.0))),
        "train_auc": _f(rt_t.get("train_auc", rt_i.get("train_auc", 0.0))),
        "val_auc": _f(rt_t.get("val_auc", rt_i.get("val_auc", 0.0))),
        "train_status": train_status,
        "infer_status": infer_status,
    }


def _is_deployable(row):
    """A result counts as a real candidate only if both legs completed, DA was
    reported, and it passes G2 (trades>80)."""
    return (row["train_status"] == "completed" and row["infer_status"] == "completed"
            and row["da_present"] and row["g2_pass"])


def _append_round_results(rows, weakest, prev_best, winner, kept):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cols = ["timestamp", "weakest_etf", "prev_best_calmar", "prev_best_cell",
            "name", "ticker", "axis", "labeler", "thresh", "sizing",
            "real_calmar", "real_da", "trades", "g2_pass", "deployable",
            "synth_cal", "train_auc", "val_auc",
            "real_sharpe", "real_skew", "real_kurt", "n_days",  # per-trial OOS moments -> enables exact Deflated Sharpe (RESEARCH_REVIEW Tier-1)
            "is_winner", "kept_as_new_best", "train_status", "infer_status"]
    new = not os.path.exists(ROUND_RESULTS_CSV) or os.path.getsize(ROUND_RESULTS_CSV) == 0
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    with open(ROUND_RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({
                "timestamp": ts,
                "weakest_etf": weakest,
                "prev_best_calmar": round(_f(prev_best.get("real_calmar", 0.0)), 4),
                "prev_best_cell": prev_best.get("cell"),
                "name": r["name"], "ticker": r["ticker"], "axis": r["axis"],
                "labeler": r["labeler"], "thresh": r["thresh"], "sizing": r["sizing"],
                "real_calmar": round(r["real_calmar"], 4), "real_da": round(r["real_da"], 4),
                "trades": r["trades"], "g2_pass": r["g2_pass"], "deployable": _is_deployable(r),
                "synth_cal": round(r["synth_cal"], 4), "train_auc": round(r["train_auc"], 4),
                "val_auc": round(r["val_auc"], 4),
                "real_sharpe": round(_f(r.get("real_sharpe", 0.0)), 4),
                "real_skew": round(_f(r.get("real_skew", 0.0)), 4),
                "real_kurt": round(_f(r.get("real_kurt", 0.0)), 4),
                "n_days": int(_f(r.get("n_days", 0.0))),
                "is_winner": (winner is not None and r["name"] == winner["name"]),
                "kept_as_new_best": (kept and winner is not None and r["name"] == winner["name"]),
                "train_status": r["train_status"], "infer_status": r["infer_status"],
            })


def _cell_key(cfg):
    """The footer's ObjectStore cell key = fixed prefix + canonical suffix. The suffix
    MUST mirror templates/header.py.tmpl _PSUF EXACTLY (order: _perm, _n, _b, _hz, reduce,
    features, calib, _tp). A mismatch makes infer read a NONEXISTENT cell -> 0 trades ->
    Calmar 0.0; in the permute gate that read as a vacuous "perfect collapse" AUTO-PASS for
    every suffix-keyed config (2026-06-10 leak-review HIGH finding: GLD's _n15_b3_ig control
    was a no-op). The suffix is now the SINGLE definition in lb.harness.psuf.cell_suffix —
    the SAME source is injected into the QC header's _PSUF, so the two call sites can never
    diverge again (the same bug class also hit 2026-06-08 via an _fx{set} variant)."""
    prefix = (f"{cfg['axis']}_{cfg['labeler'].replace('+','_x_')}_{cfg['sizing']}"
              f"_t{int(round(float(cfg['thresh'])*100))}")
    return prefix + cell_suffix(cfg)


def _run_one_config(target, cfg, tag="permval"):
    """Train+infer ONE config end-to-end (reusing the round's QC machinery) and return its
    result row. Used by the standing permuted-label GATE to validate a KEEP. Returns None
    if the train or infer leg fails to complete."""
    code, extra = render_train_config(cfg)
    tlabel = f"{tag}_{target}_train"
    tr = run_pool([(tlabel, code, extra)]).get(tlabel, {})
    if not str(tr.get("status", "")).startswith("Completed"):
        return None
    cell = _cell_key(cfg)
    ilabel = f"{tag}_{target}_infer"
    ir = run_pool([(ilabel, render_infer_cell(cfg["ticker"], cell))]).get(ilabel, {})
    return _extract_result(tag.upper(), tr, ir, cfg)


# ===========================================================================
# MAIN ROUND
# ===========================================================================
def run_round(argv):
    knowledge = _load_knowledge()
    per_etf_best = get_per_etf_best(knowledge)
    weakest, ranked = pick_weakest(per_etf_best)

    print("=== TOURNAMENT ROUND (v2): per-ETF, two-hypothesis, 2-node parallel ===")
    print("Per-ETF current best (real OOS Calmar):")
    for tk in ranked:
        b = per_etf_best.get(tk, {})
        flag = "  <== WEAKEST" if tk == weakest else ""
        print(f"  {tk:5s} Calmar={_f(b.get('real_calmar',0)):+.4f} "
              f"trades={int(b.get('trades',0) or 0):5d} G2={'Y' if b.get('g2_pass') else 'n'} "
              f"cell={b.get('cell')}{flag}")

    target, cfg_a, cfg_b = load_hypotheses(argv, weakest)
    if target != weakest:
        print(f"\n[NOTE] argv hypotheses target {target}, not the auto-picked weakest {weakest}. "
              f"Using {target} (explicit argv overrides the auto-pick).")
    prev_best = per_etf_best.get(target, {"real_calmar": 0.0, "cell": None, "trades": 0, "g2_pass": False})

    print(f"\nETF under test: {target}   prev best Calmar={_f(prev_best.get('real_calmar',0)):+.4f} "
          f"(cell={prev_best.get('cell')}, G2={'Y' if prev_best.get('g2_pass') else 'n'})")
    print(f"  Hypothesis A: {cfg_a}")
    print(f"  Hypothesis B: {cfg_b}")

    round_no = _round_count() + 1
    _write_status(running=True, etf=target, round=round_no,
                  since=datetime.now().strftime("%Y-%m-%d %H:%M"),
                  phase="training — fitting 2 models on the 2 QC nodes", legs="train 0/2 · infer 0/2",
                  hypotheses=[describe_cfg(cfg_a), describe_cfg(cfg_b)])

    # Render BOTH train scripts, then run them IN PARALLEL on the 2 nodes.
    train_jobs = []
    for nm, cfg in (("A", cfg_a), ("B", cfg_b)):
        code, extra = render_train_config(cfg)   # multi-file: extra = {"bar_builder.py": ...}
        if len(code) >= 64000:
            raise RuntimeError(f"rendered train {nm} too large: {len(code)} >= 64000")
        train_jobs.append((f"train_{target}_{nm}_{cfg['axis']}_{cfg['labeler']}", code, extra))
    print(f"\n[{_now()}] PHASE TRAIN: 2 hypotheses in parallel")
    train_res = run_pool(train_jobs)

    # Infer each completed cell (parallel). Skip inferring a crashed/timed-out train.
    infer_jobs = []
    cfg_by_name = {"A": cfg_a, "B": cfg_b}
    train_by_name = {}
    for nm, cfg in (("A", cfg_a), ("B", cfg_b)):
        tjob = f"train_{target}_{nm}_{cfg['axis']}_{cfg['labeler']}"
        bt = train_res.get(tjob, {})
        train_by_name[nm] = bt
        if str(bt.get("status", "")).startswith("Completed"):
            cell = _cell_key(cfg)   # ONE shared _PSUF mirror (2026-06-10: inline copies diverged twice — see _cell_key docstring)
            infer_jobs.append((f"infer_{target}_{nm}", render_infer_cell(cfg["ticker"], cell)))
        else:
            print(f"[{_now()}]   hypothesis {nm} train not completed ({bt.get('status','?')}) — skip infer")
    _write_status(phase="inferring — real OOS backtest of each trained cell",
                  legs=f"train done · infer 0/{len(infer_jobs)}")
    print(f"\n[{_now()}] PHASE INFER: {len(infer_jobs)} cells in parallel")
    infer_res = run_pool(infer_jobs) if infer_jobs else {}

    # Collate both hypotheses.
    rows = []
    for nm in ("A", "B"):
        cfg = cfg_by_name[nm]
        rows.append(_extract_result(nm, train_by_name.get(nm, {}),
                                    infer_res.get(f"infer_{target}_{nm}", {}), cfg))

    # Winner = the DEPLOYABLE hypothesis with the higher REAL Calmar. If neither is
    # deployable, the winner is the higher-Calmar of the two (reported, never kept).
    deployable = [r for r in rows if _is_deployable(r)]
    pool = deployable if deployable else rows
    winner = max(pool, key=lambda r: r["real_calmar"]) if pool else None

    # Keep iff the winner is DEPLOYABLE (G2 + DA reported) AND beats the ETF's
    # current best Calmar.
    prev_cal = _f(prev_best.get("real_calmar", 0.0))
    # program.md keep-rule (honesty > horsepower): beats prev best AND Calmar>0 AND
    # val_auc>0.52 (window-artifact guard) AND survives deflation (best-of-N-trials noise).
    _vauc = _f(winner.get("val_auc", 0.0)) if winner else 0.0
    _defl_ok, _defl_bench, _defl_n = _survives_deflation(target, winner) if winner else (True, 0.0, 0)
    # Trials-adjusted significance (Bonferroni-deflated PSR) — a CROWN must be trials-significant,
    # not just beat the best-of-N noise. Prevents crowning weak permute-passing edges on new tickers
    # with few trials / negative buy-hold (e.g. ITB Sharpe 0.22, KRE PSR 0.972 — real label signal but
    # NOT trials-significant => provisional, not deployable). GLD/UUP/SOXX clear this; KRE/ITB do not.
    _nt = (_count_trials(target) + 2) if winner else 0
    _psr, _sig = (_psr_significance(winner.get("real_sharpe"), winner.get("real_skew"),
                                    winner.get("real_kurt"), winner.get("n_days"), _nt)
                  if winner else (None, False))
    if winner is not None:
        winner["_psr"], winner["_sig"], winner["_ntrials"] = _psr, _sig, _nt
    _beats = bool(winner is not None and _is_deployable(winner) and winner["real_calmar"] > prev_cal)
    kept = bool(_beats and winner["real_calmar"] > 0 and _vauc > 0.52 and _defl_ok and _sig)

    # ---- STANDING PERMUTED-LABEL GATE (honesty harness) ----
    # A real edge must COLLAPSE when the TRAIN labels are shuffled. Re-run the winner with
    # permute_labels=True (distinct _perm cell, leak-safe). The right metric is the EXCESS over
    # the no-skill BUY-HOLD baseline (raw Calmar conflates retained drift with retained edge for
    # up-drifters): a REAL edge's excess-over-buyhold collapses to ~0 under permutation. PASS iff
    # (a) the real edge over buy-hold is MEANINGFUL (>0.15, not noise — rejects SPY's +0.07) AND
    # (b) permuting removes >=60% of that excess. Falls back to the raw permuted/real ratio only
    # when no buy-hold baseline is available. Runs only on a tentative KEEP (rare).
    _perm_note = ""
    if kept:
        _wcfg = dict(cfg_by_name[winner["name"]]); _wcfg["permute_labels"] = True
        print(f"\n[{_now()}] PERMUTE-GATE: re-running {target} winner with SHUFFLED train labels (must collapse)...")
        _pv = _run_one_config(target, _wcfg)
        # FAIL-LOUD (2026-06-10 leak-review): an empty/failed control is INDISTINGUISHABLE
        # from a vacuous nonexistent-cell read (Calmar 0.0 = fake "perfect collapse").
        # A KEEP requires a DEMONSTRATED collapse — no valid control, no crown.
        _pv_valid = (_pv is not None
                     and str(_pv.get("infer_status", "")).startswith("completed")
                     and int(_f(_pv.get("trades", 0))) >= 3)
        if not _pv_valid:
            kept = False
            _perm_note = (" · PERMUTE-GATE INVALID: control leg empty or failed "
                          f"(trades={None if _pv is None else _pv.get('trades')}) — "
                          "cannot demonstrate label-shuffle collapse, KEEP refused")
        else:
            _pc = _f(_pv.get("real_calmar", 0.0)); _rc = _f(winner.get("real_calmar", 0.0))
            # buy-hold baseline: knowledge['buyhold'], else an always_long arm in THIS round.
            _bh = knowledge.get("buyhold", {}).get(target, {}).get("calmar")
            if _bh is None:
                _alrows = [r for r in rows if r.get("labeler") == "always_long"]
                _bh = _alrows[0]["real_calmar"] if _alrows else None
            if _bh is not None:
                _bh = _f(_bh)
                _real_x = _rc - _bh; _perm_x = _pc - _bh            # excess over no-skill buy-hold
                if _real_x <= 0.15:
                    kept = False
                    _perm_note = (f" · PERMUTE-GATE FAILED: real edge over buy-hold {_real_x:+.4f} <= 0.15 "
                                  f"(real {_rc:+.4f} vs buy-hold {_bh:+.4f}) — too small to be a real edge")
                elif _perm_x >= 0.4 * _real_x:
                    kept = False
                    _perm_note = (f" · PERMUTE-GATE FAILED: permuted edge {_perm_x:+.4f} >= 40% of real edge "
                                  f"{_real_x:+.4f} (buy-hold {_bh:+.4f}) — edge survives label-shuffle = ARTIFACT")
                else:
                    _perm_note = (f" · permute-gate PASS: edge over buy-hold {_real_x:+.4f} COLLAPSES to {_perm_x:+.4f} "
                                  f"under label-shuffle (buy-hold {_bh:+.4f}) — REAL label signal")
            elif _pc >= 0.6 * _rc:                                   # fallback: no buy-hold baseline
                kept = False
                _perm_note = (f" · PERMUTE-GATE FAILED: permuted Calmar {_pc:+.4f} >= 60% of real {_rc:+.4f} "
                              f"(no buy-hold baseline) — edge survives label-shuffle = ARTIFACT, not crowned")
            else:
                _perm_note = f" · permute-gate PASS: permuted {_pc:+.4f} << real {_rc:+.4f} (real label signal)"

    # ---- PRINT clear comparison ----
    print("\n" + "=" * 84)
    print(f"COMPARISON — ETF {target}  (prev best Calmar={prev_cal:+.4f}, cell={prev_best.get('cell')})")
    print("-" * 84)
    print(f"{'H':>2} {'axis':9s} {'labeler':13s} {'thr':>4s} {'sizing':11s} "
          f"{'Calmar':>9s} {'DA':>8s} {'trades':>7s} {'G2':>3s} {'status':>10s}")
    for r in rows:
        tag = "*" if (winner is not None and r["name"] == winner["name"]) else " "
        st = r["infer_status"] if r["train_status"] == "completed" else f"train:{r['train_status']}"
        print(f"{tag}{r['name']:>1} {r['axis']:9s} {r['labeler']:13s} {r['thresh']:4.2f} "
              f"{r['sizing']:11s} {r['real_calmar']:+9.4f} {r['real_da']:8.3f} "
              f"{r['trades']:7d} {'Y' if r['g2_pass'] else 'n':>3s} {str(st):>10s}")
    print("-" * 84)
    if winner is None:
        print("WINNER: none (both hypotheses failed to produce a result).")
    else:
        depl = "DEPLOYABLE" if _is_deployable(winner) else "NOT-deployable (G2 fail / DA missing / crash)"
        print(f"WINNER: {winner['name']} — Calmar {winner['real_calmar']:+.4f}, "
              f"DA {winner['real_da']:.3f}, trades {winner['trades']} [{depl}]")
        if kept:
            print(f"VERDICT: KEEP — beats {target} prev best ({winner['real_calmar']:+.4f} > {prev_cal:+.4f}), "
                  f"Calmar>0, val_auc {_vauc:.3f}>0.52, survives deflation (best-of-{_defl_n} noise {_defl_bench})"
                  f"{_perm_note}. per_etf_best[{target}] updated.")
            if _psr is not None:
                print(f"  TRIALS-ADJUSTED TRUST: Sharpe {winner.get('real_sharpe')}, PSR {_psr}, "
                      f"N_trials {_nt}, Bonferroni {'PASS — significant' if _sig else 'FAIL — selection-bias-suspect (edge not trials-significant)'}")
        elif _beats:
            # beat prev best but FAILED an honest gate (program.md) — do NOT crown an artifact
            reasons = []
            if winner["real_calmar"] <= 0:
                reasons.append("Calmar<=0")
            if _vauc <= 0.52:
                reasons.append(f"val_auc {_vauc:.3f}<=0.52 (window/path artifact, not an edge)")
            if not _defl_ok:
                reasons.append(f"FAILS deflation (best-of-{_defl_n}-trials noise {_defl_bench}; selection-bias artifact)")
            if "PERMUTE-GATE FAILED" in _perm_note:
                reasons.append(_perm_note.split("·")[-1].strip())
            if not _sig:
                reasons.append(f"NOT trials-significant (Bonferroni FAIL, PSR {_psr}, N={_nt} — selection-bias-suspect)")
            print(f"VERDICT: DISCARD — winner {winner['real_calmar']:+.4f} beats prev best {prev_cal:+.4f} "
                  f"but FAILS the honest keep-gate: {', '.join(reasons)}.")
        elif _is_deployable(winner):
            print(f"VERDICT: DISCARD — winner {winner['real_calmar']:+.4f} does NOT beat "
                  f"prev best {prev_cal:+.4f}.")
        else:
            print(f"VERDICT: DISCARD — winner not deployable (needs G2 trades>{G2_MIN_TRADES} + real DA).")
    print("=" * 84)

    # ---- Update per_etf_best on keep ----
    if kept:
        cell_key = f"{target}_{winner['axis']}_{winner['labeler']}_{winner['sizing']}_t{winner['thresh']:.2f}"
        new_round = round_no   # real per-round number (was KEEP-only last_round+1, which froze the count)
        per_etf_best[target] = {
            "cell": cell_key,
            "real_calmar": round(winner["real_calmar"], 4),
            "real_cagr": round(winner.get("real_cagr", 0.0), 4),
            "real_mdd": round(winner.get("real_mdd", 0.0), 4),
            "real_da": round(winner["real_da"], 4),
            "real_sharpe": round(_f(winner.get("real_sharpe")), 3),
            "trades": winner["trades"],
            "g2_pass": True,
            "round": new_round,
            "source": "tournament",
            "psr": winner.get("_psr"),
            "n_trials": winner.get("_ntrials"),
            "significant": winner.get("_sig"),
            "leak_fixed": True,
            # full config incl. optional levers (max_depth/n_components/rebal_band/reduce/permute_labels/
            # horizons) so the crown is REPRODUCIBLE — the prior version dropped them (lost reduce=infogain etc.).
            "config": {k: winner[k] for k in ("ticker", "axis", "labeler", "thresh", "sizing",
                       "max_depth", "n_components", "rebal_band", "reduce", "permute_labels", "horizons", "features")
                       if k in winner},
        }
        knowledge["per_etf_best"] = per_etf_best
        knowledge["last_round"] = new_round
    else:
        # Still persist the seeded per_etf_best (so subsequent rounds are stable).
        knowledge["per_etf_best"] = per_etf_best
    _save_knowledge(knowledge)

    # ---- Append round_results.csv (BOTH hypotheses logged) ----
    _append_round_results(rows, target, prev_best, winner, kept)
    print(f"\n[{_now()}] logged 2 hypotheses -> {ROUND_RESULTS_CSV}")
    print(f"[{_now()}] knowledge.json per_etf_best {'UPDATED' if kept else 'unchanged'} for {target}")

    # ---- AUTO-REFRESH the report (additive, fully guarded; never fails a round) ----
    # The rounds ledger renders LIVE from round_results.csv, so just (a) append a
    # round node to the causal graph and (b) rebuild index.html. The Flask app also
    # renders live per request; this keeps the on-disk file current for static hosting.
    _append_causal_round_node(target, winner, kept, prev_cal)
    _refresh_report()
    print(f"[{_now()}] report auto-refreshed from round_results.csv (git commit still done by human/opus).")

    # ---- Live status -> idle/done (dashboard poller reads this) ----
    if winner is not None:
        verdict = ("KEEP — new best %.4f" % winner["real_calmar"]) if kept else (
            "DISCARD — winner %.4f did not beat %.4f" % (winner["real_calmar"], prev_cal))
    else:
        verdict = "no result (both hypotheses failed)"
    _write_status(running=False, etf=target,
                  note=f"{target}: {verdict}", phase="done")
    return {"weakest": target, "rows": rows, "winner": winner, "kept": kept}


if __name__ == "__main__":
    run_round(sys.argv[1:])
