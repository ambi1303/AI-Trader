/* Position-size calculator. Debounced calls to /api/position-size; results
 * rendered with textContent (CSP-safe, no innerHTML with data). */
(function () {
  "use strict";

  var ids = ["rs-capital", "rs-risk", "rs-entry", "rs-stop", "rs-target", "rs-lot"];
  var el = {};
  ids.forEach(function (id) { el[id] = document.getElementById(id); });
  if (!el["rs-capital"]) return;

  var out = {
    shares: document.getElementById("rs-shares"),
    value: document.getElementById("rs-value"),
    pct: document.getElementById("rs-pct"),
    riskAmt: document.getElementById("rs-risk-amt"),
    riskPct: document.getElementById("rs-risk-pct"),
    rr: document.getElementById("rs-rr"),
    notes: document.getElementById("rs-notes")
  };

  var timer = null;

  function rupees(n) {
    if (n === null || n === undefined) return "\u2014";
    return "\u20b9" + Number(n).toLocaleString("en-IN", { maximumFractionDigits: 0 });
  }

  function compute() {
    var params = new URLSearchParams({
      capital: el["rs-capital"].value || "0",
      risk_pct: el["rs-risk"].value || "0",
      entry: el["rs-entry"].value || "0",
      stop: el["rs-stop"].value || "0",
      lot_size: el["rs-lot"].value || "1"
    });
    if (el["rs-target"].value) params.set("target", el["rs-target"].value);

    fetch("/api/position-size?" + params.toString(), {
      headers: { "Accept": "application/json" },
      credentials: "same-origin"
    })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(render)
      .catch(function () {});
  }

  function render(d) {
    if (!d) return;
    if (d.valid) {
      out.shares.textContent = Number(d.shares).toLocaleString("en-IN");
      out.value.textContent = rupees(d.position_value);
      out.pct.textContent = d.pct_of_capital + "% of capital";
      out.riskAmt.textContent = rupees(d.risk_amount);
      out.riskPct.textContent = d.risk_pct_actual + "% of capital";
      out.rr.textContent = (d.risk_reward !== null && d.risk_reward !== undefined)
        ? "1 : " + d.risk_reward : "\u2014";
    } else {
      out.shares.textContent = "\u2014";
      out.value.textContent = "\u2014";
      out.pct.textContent = "";
      out.riskAmt.textContent = "\u2014";
      out.riskPct.textContent = "";
      out.rr.textContent = "\u2014";
    }
    out.notes.textContent = (d.notes && d.notes.length) ? d.notes.join(" ") : "";
  }

  ids.forEach(function (id) {
    el[id].addEventListener("input", function () {
      if (timer) clearTimeout(timer);
      timer = setTimeout(compute, 200);
    });
  });

  compute();  // initial render with defaults
})();
