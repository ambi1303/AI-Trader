/* Profit-target feasibility slider. Debounced calls to
 * /api/profit-feasibility; results rendered with textContent (CSP-safe). */
(function () {
  "use strict";

  var root = document.getElementById("feasibility");
  if (!root) return;
  var symbol = root.getAttribute("data-symbol");
  if (!symbol) return;

  var targetEl = document.getElementById("feas-target");
  var daysEl = document.getElementById("feas-days");
  var targetVal = document.getElementById("feas-target-val");
  var daysVal = document.getElementById("feas-days-val");
  var rTarget = document.getElementById("feas-r-target");
  var rDays = document.getElementById("feas-r-days");
  var verdict = document.getElementById("feas-verdict");
  var prob = document.getElementById("feas-prob");
  var price = document.getElementById("feas-price");
  var notes = document.getElementById("feas-notes");
  if (!targetEl || !daysEl) return;

  // Seed the target slider from the computed zone target, if provided.
  var def = parseInt(root.getAttribute("data-default-target"), 10);
  if (!isNaN(def) && def >= 3 && def <= 40) {
    targetEl.value = String(def);
  }

  var TONE = {
    good: "text-emerald-400",
    ok: "text-amber-300",
    warn: "text-amber-400",
    bad: "text-rose-400"
  };

  var timer = null;

  function clearNotes() {
    while (notes.firstChild) notes.removeChild(notes.firstChild);
  }

  function syncLabels() {
    targetVal.textContent = targetEl.value;
    daysVal.textContent = daysEl.value;
    rTarget.textContent = "+" + targetEl.value + "%";
    rDays.textContent = daysEl.value;
  }

  function render(d) {
    verdict.className = "mt-1 text-2xl font-bold " + (TONE[d && d.tone] || "text-slate-300");
    if (!d || !d.available) {
      verdict.textContent = "\u2014";
      prob.textContent = "Not enough data for this stock.";
      price.textContent = "";
      clearNotes();
      return;
    }
    verdict.textContent = d.verdict;
    prob.textContent = "~" + d.prob_touch_pct + "% chance \u00b7 typical swing \u00b1"
      + d.typical_move_pct + "%";
    if (d.target_price !== undefined && d.target_price !== null) {
      price.textContent = "Target price \u2248 \u20b9"
        + Number(d.target_price).toLocaleString("en-IN", { maximumFractionDigits: 2 });
    } else {
      price.textContent = "";
    }
    clearNotes();
    (d.notes || []).forEach(function (n) {
      var li = document.createElement("li");
      li.textContent = n;
      notes.appendChild(li);
    });
  }

  function compute() {
    syncLabels();
    var params = new URLSearchParams({
      symbol: symbol,
      target_pct: targetEl.value || "10",
      days: daysEl.value || "30"
    });
    fetch("/api/profit-feasibility?" + params.toString(), {
      headers: { "Accept": "application/json" },
      credentials: "same-origin"
    })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(render)
      .catch(function () {});
  }

  [targetEl, daysEl].forEach(function (s) {
    s.addEventListener("input", function () {
      syncLabels();
      if (timer) clearTimeout(timer);
      timer = setTimeout(compute, 200);
    });
  });

  syncLabels();
  compute();  // initial render with defaults
})();
