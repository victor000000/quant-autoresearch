/* console.js — the autoresearch dashboard client (externalized from render_index).
 *
 * Cacheable, off the HTML payload. Owns five behaviours:
 *   1. the wall clock                     (#clock)
 *   2. client-side sortable tables        (any <table> with thead th[data-k])
 *   3. the rounds KEEP/DISCARD filter     (.fchip[data-f] -> toggle .ritem)
 *   4. "show all rounds" lazy pagination  (#showall -> /rounds.json?page=)
 *   5. the 8s status poll                 (/data.json -> patch #nowrunning)
 *   6. the lazy causal graph              (#graphwrap -> /graph.json -> vis.Network)
 *
 * Nothing here formats a headline number; the server (resolvers) owns those.
 */
(function () {
  "use strict";

  // -- 1. clock --------------------------------------------------------------
  function tick() {
    var c = document.getElementById("clock");
    if (c) c.textContent = new Date().toLocaleTimeString();
  }
  tick();
  setInterval(tick, 1000);

  // -- 2. sortable tables (per-table state; works for #lb, #scrlb, #complab) --
  var SORT = {}; // table id -> {key, dir}
  function sortTable(table, key, type) {
    var tb = table.tBodies[0];
    if (!tb) return;
    var ths = [].slice.call(table.tHead.querySelectorAll("th"));
    var idx = -1;
    ths.forEach(function (th, i) {
      th.classList.remove("sorted-asc", "sorted-desc");
      if (th.dataset.k === key) idx = i;
    });
    if (idx < 0) return;
    var id = table.id || "_t";
    var st = SORT[id] || { key: key, dir: type === "s" ? 1 : -1 };
    if (st.key === key) { /* keep dir (caller already flipped) */ } else {
      st = { key: key, dir: type === "s" ? 1 : -1 };
    }
    SORT[id] = st;
    ths[idx].classList.add(st.dir < 0 ? "sorted-desc" : "sorted-asc");
    var rows = [].slice.call(tb.querySelectorAll("tr"));
    rows.sort(function (a, b) {
      var x = (a.children[idx].innerText || "").replace(/[%+,]/g, "").trim();
      var y = (b.children[idx].innerText || "").replace(/[%+,]/g, "").trim();
      if (type === "n") {
        x = parseFloat(x); y = parseFloat(y);
        if (isNaN(x)) x = -1e9;
        if (isNaN(y)) y = -1e9;
        return (x - y) * st.dir;
      }
      return x.localeCompare(y) * st.dir;
    });
    rows.forEach(function (r) { tb.appendChild(r); });
  }

  // -- 3/4. delegated clicks: sort headers, filter chips, show-all -----------
  document.addEventListener("click", function (e) {
    var th = e.target.closest("thead th[data-k]");
    if (th) {
      var table = th.closest("table");
      if (!table) return;
      var id = table.id || "_t";
      var st = SORT[id];
      if (st && st.key === th.dataset.k) st.dir *= -1;
      else SORT[id] = { key: th.dataset.k, dir: th.dataset.t === "s" ? 1 : -1 };
      sortTable(table, th.dataset.k, th.dataset.t || "n");
      return;
    }
    var fc = e.target.closest(".fchip");
    if (fc) {
      var group = fc.parentNode;
      [].forEach.call(group.querySelectorAll(".fchip"), function (c) {
        c.classList.remove("on");
      });
      fc.classList.add("on");
      var f = fc.dataset.f;
      [].forEach.call(document.querySelectorAll(".ritem"), function (li) {
        li.style.display = (f === "all" || li.classList.contains(f)) ? "" : "none";
      });
      return;
    }
    if (e.target.id === "showall") loadMoreRounds(e.target);
  });

  // re-apply the default Calmar sort on the leaderboard at load
  (function () {
    var lb = document.getElementById("lb");
    if (lb) sortTable(lb, "calmar", "n");
  })();

  // -- 4. lazy rounds pagination (/rounds.json?page=) ------------------------
  var roundsBusy = false;
  function loadMoreRounds(btn) {
    if (roundsBusy) return;
    roundsBusy = true;
    var src = btn.dataset.src || "rounds.json";
    var page = parseInt(btn.dataset.page || "2", 10);
    btn.textContent = "loading…";
    fetch(src + "?page=" + page, { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(0); })
      .then(function (d) {
        var ol = document.querySelector("ol.rounds");
        if (ol && d.items) ol.insertAdjacentHTML("beforeend", d.items);
        roundsBusy = false;
        if (d.has_more) {
          btn.dataset.page = String(page + 1);
          btn.textContent = "show more rounds";
        } else {
          btn.remove();
        }
        // keep the active KEEP/DISCARD filter applied to the new rows
        var active = document.querySelector(".filters .fchip.on");
        if (active && active.dataset.f !== "all") {
          var f = active.dataset.f;
          [].forEach.call(document.querySelectorAll(".ritem"), function (li) {
            li.style.display = li.classList.contains(f) ? "" : "none";
          });
        }
      })
      .catch(function () { roundsBusy = false; btn.textContent = "show all rounds"; });
  }

  // -- 5. 8s status poll (/data.json -> patch #nowrunning + optional scoreboard)
  async function poll() {
    var el = document.getElementById("nowrunning");
    if (!el) return;
    try {
      var r = await fetch("data.json?_=" + Date.now(), { cache: "no-store" });
      if (!r.ok) throw 0;
      var d = await r.json();
      var s = d.status || {};
      if (s.running) {
        var legs = (s.hypotheses || []).map(function (h) {
          return "<li>▸ " + h + "</li>";
        }).join("");
        el.className = "running";
        el.innerHTML = '<h2><span class="livedot"></span>Now running' +
          (s.round ? " — round " + s.round : "") + (s.etf ? " · " + s.etf : "") + "</h2>" +
          '<p class="small">phase: <b>' + (s.phase || "…") + "</b>" +
          (s.since ? " · started " + s.since : "") + (s.legs ? " · " + s.legs : "") + "</p>" +
          '<ul class="hyps">' + legs + "</ul>";
      } else {
        el.className = "";
        el.innerHTML = '<h2><span class="idledot"></span>Idle</h2>' +
          '<p class="small">last: <b>' + (s.note || ("round " + (s.round || "?"))) + "</b></p>";
      }
      // scoreboard is server-rendered in the leaderboard now; patch only if present
      var sb = d.scoreboard || {};
      var sbel = document.getElementById("scoreboard");
      if (sbel) {
        sbel.innerHTML =
          '<div class="kpi"><div class="k">beat buy &amp; hold</div><div class="v acc">' + (sb.edges || 0) + "</div></div>" +
          '<div class="kpi"><div class="k">best Calmar</div><div class="v pos">' + (+(sb.best_calmar || 0)).toFixed(2) + "</div></div>" +
          '<div class="kpi"><div class="k">pass luck-check</div><div class="v">' + (sb.n_sig || 0) + "/" + (sb.n_assessed || 0) + "</div></div>" +
          '<div class="kpi"><div class="k">rounds</div><div class="v">' + (sb.rounds || 0) + "</div></div>";
      }
    } catch (e) {
      if (el && !el.querySelector("h2")) el.className = "";
    }
  }
  poll();
  setInterval(poll, 8000);

  // -- 6. lazy causal graph (#graphwrap -> /graph.json -> vis.Network) --------
  var VIS_CDN = "https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js";
  function loadVis() {
    return new Promise(function (resolve, reject) {
      if (window.vis && window.vis.Network) return resolve();
      var s = document.createElement("script");
      s.src = VIS_CDN;
      s.onload = function () { resolve(); };
      s.onerror = function () { reject(0); };
      document.head.appendChild(s);
    });
  }
  function buildGraph(wrap, data) {
    var groups = {
      finding: { shape: "box", color: { background: "#13231f", border: "#38e0c8" }, font: { color: "#cdeee7", size: 13 }, borderWidth: 2 },
      milestone: { shape: "box", color: { background: "#0f241a", border: "#3fd07a" }, font: { color: "#bfe9cd", size: 13, bold: true }, borderWidth: 2 },
      decision: { shape: "box", color: { background: "#1c1832", border: "#9a86ff" }, font: { color: "#d8d0ff", size: 13 }, borderWidth: 2 },
      round: { shape: "dot", color: { background: "#0d1320", border: "#33415a" }, font: { size: 11, color: "#9aa6b8" } }
    };
    var opts = {
      nodes: { shape: "box", margin: 8, widthConstraint: { maximum: 190 }, shadow: false },
      groups: groups,
      edges: {
        arrows: { to: { scaleFactor: 0.6 } }, color: { color: "#33415a", highlight: "#38e0c8" },
        font: { size: 11, color: "#9aa6b8", strokeWidth: 4, strokeColor: "#070a10", align: "middle" },
        smooth: { type: "cubicBezier", roundness: 0.4 }
      },
      physics: { stabilization: { iterations: 300 }, barnesHut: { gravitationalConstant: -14000, springLength: 150, springConstant: 0.02, avoidOverlap: 0.5 } },
      interaction: { hover: true, tooltipDelay: 120, navigationButtons: true, keyboard: false, zoomView: false, dragView: true },
      layout: { improvedLayout: true }
    };
    var nodes = new window.vis.DataSet(data.nodes || []);
    var edges = new window.vis.DataSet(data.edges || []);
    var phases = data.phases || [];
    wrap.style.height = "560px";
    wrap.innerHTML = "";
    var net = new window.vis.Network(wrap, { nodes: nodes, edges: edges }, opts);
    function collapse() {
      phases.forEach(function (p) {
        net.cluster({
          joinCondition: function (o) { return o.group === "round" && o.phase === p; },
          processProperties: function (c, kids) { c.label = p + " · " + kids.length + " experiments ▸"; return c; },
          clusterNodeProperties: { shape: "box", borderWidth: 2, shapeProperties: { borderDashes: [4, 3] }, color: { background: "#161d2b", border: "#94a0ad" }, font: { size: 12, color: "#cdd6e3" } }
        });
      });
    }
    net.on("doubleClick", function (p) {
      if (p.nodes.length && net.isCluster(p.nodes[0])) net.openCluster(p.nodes[0]);
    });
    net.once("stabilizationIterationsDone", function () { net.setOptions({ physics: false }); collapse(); });
  }
  var gw = document.getElementById("graphwrap");
  if (gw && "IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (ents) {
      ents.forEach(function (e) {
        if (!e.isIntersecting || !gw.dataset.pending) return;
        gw.dataset.pending = "";
        var src = gw.dataset.graphSrc || "/graph.json";
        gw.innerHTML = '<div class="small">loading interactive lineage…</div>';
        loadVis()
          .then(function () { return fetch(src, { cache: "no-store" }); })
          .then(function (r) { return r.ok ? r.json() : Promise.reject(0); })
          .then(function (d) { buildGraph(gw, d); })
          .catch(function () {
            gw.innerHTML = '<div class="small">graph unavailable — open the full page link below.</div>';
          });
      });
    }, { rootMargin: "200px" });
    io.observe(gw);
  }

  // smooth-scroll old deep links (#overview -> #leaderboard, etc.)
  function jump() {
    var h = (location.hash || "").replace("#overview", "#leaderboard").replace("#map", "#leaderboard").replace("#story", "#arc").replace("#insights", "#arc");
    var t = h && document.querySelector(h);
    if (t) t.scrollIntoView({ behavior: "smooth" });
  }
  window.addEventListener("hashchange", jump);
  if (location.hash) setTimeout(jump, 200);
})();
