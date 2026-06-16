/* Near-real-time price updater.
 *
 * Finds every element carrying `data-live-symbol="SYMBOL"` and refreshes its
 * child price/change/badge nodes from `/api/ltp?symbols=...` while the market
 * is open. When the market is shut (or a quote is unavailable) it leaves the
 * server-rendered EOD close in place and shows a "Closed" badge.
 *
 * Markup contract (all children optional):
 *   <element data-live-symbol="RELIANCE" data-entry="2800" data-qty="10">
 *     <span class="live-ltp">…</span>     price, formatted ₹
 *     <span class="live-change">…</span>  +/- change %
 *     <span class="live-pnl">…</span>     (entry+qty present) unrealised P&L
 *     <span class="live-badge">…</span>   LIVE / Closed pill
 *   </element>
 *
 * No inline script (CSP-safe); served from /static.
 */
(function () {
  "use strict";

  var nodes = Array.prototype.slice.call(
    document.querySelectorAll("[data-live-symbol]")
  );
  var statusNodes = Array.prototype.slice.call(
    document.querySelectorAll("[data-market-status]")
  );
  if (!nodes.length && !statusNodes.length) return;

  var symbols = [];
  nodes.forEach(function (n) {
    var s = (n.getAttribute("data-live-symbol") || "").toUpperCase();
    if (s && symbols.indexOf(s) === -1) symbols.push(s);
  });

  var POLL_OPEN_MS = 10000; // market open: ~10s (matches server cache TTL)
  var POLL_SHUT_MS = 60000; // market shut: slow probe so it flips to LIVE on open
  var inr = new Intl.NumberFormat("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  var timer = null;

  function fmtMoney(v) {
    if (v === null || v === undefined || isNaN(v)) return "—";
    return "\u20B9" + inr.format(v);
  }

  function setText(node, sel, text, cls) {
    var el = node.querySelector(sel);
    if (!el) return;
    el.textContent = text;
    if (cls !== undefined) el.className = cls;
  }

  function render(node, qt, marketOpen) {
    var live = marketOpen && qt && qt.ltp !== null && qt.ltp !== undefined;
    var shown = live ? qt.ltp : qt ? qt.prev_close : null;

    setText(node, ".live-ltp", fmtMoney(shown));

    // Change %
    var chEl = node.querySelector(".live-change");
    if (chEl && qt && qt.change_pct !== null && qt.change_pct !== undefined) {
      var up = qt.change_pct >= 0;
      chEl.textContent =
        (up ? "+" : "") + qt.change_pct.toFixed(2) + "%";
      chEl.className =
        "live-change text-xs font-normal " +
        (up ? "text-emerald-400" : "text-rose-400");
    } else if (chEl) {
      chEl.textContent = "";
    }

    // Unrealised P&L for open positions (needs entry + qty on the node).
    var entry = parseFloat(node.getAttribute("data-entry"));
    var qty = parseFloat(node.getAttribute("data-qty"));
    var pnlEl = node.querySelector(".live-pnl");
    var pnlPctEl = node.querySelector(".live-pnlpct");
    if ((pnlEl || pnlPctEl) && shown !== null && !isNaN(entry) && entry !== 0) {
      var pnlPct = (shown / entry - 1.0) * 100.0;
      var posPct = pnlPct >= 0;
      if (pnlEl && !isNaN(qty)) {
        var pnl = (shown - entry) * qty;
        pnlEl.textContent = (pnl >= 0 ? "+" : "-") + "\u20B9" + inr.format(Math.abs(pnl));
        pnlEl.className =
          "live-pnl " + (pnl >= 0 ? "text-emerald-400" : "text-rose-400");
      }
      if (pnlPctEl) {
        pnlPctEl.textContent = (posPct ? "+" : "") + pnlPct.toFixed(2) + "%";
        pnlPctEl.className =
          "live-pnlpct " + (posPct ? "text-emerald-400" : "text-rose-400");
      }
    }

    // Badge
    var badge = node.querySelector(".live-badge");
    if (badge) {
      if (live) {
        badge.textContent = "● LIVE";
        badge.title = "Live price, delayed ~10s";
        badge.className =
          "live-badge inline-flex items-center gap-1 text-[11px] px-1.5 py-0.5 rounded border bg-emerald-500/15 text-emerald-300 border-emerald-700";
      } else {
        badge.textContent = "Closed";
        badge.title = qt && qt.as_of ? "Last close " + qt.as_of : "Market closed";
        badge.className =
          "live-badge inline-flex items-center gap-1 text-[11px] px-1.5 py-0.5 rounded border bg-slate-700/40 text-slate-300 border-slate-600";
      }
    }
  }

  function renderStatus(open) {
    statusNodes.forEach(function (el) {
      if (open) {
        el.textContent = "● Market open · LIVE";
        el.title = "Live prices, delayed ~10s";
        el.className =
          el.getAttribute("data-base-class") +
          " bg-emerald-500/15 text-emerald-300 border-emerald-700";
      } else {
        el.textContent = "Market closed";
        el.title = "Showing last end-of-day close";
        el.className =
          el.getAttribute("data-base-class") +
          " bg-slate-700/40 text-slate-300 border-slate-600";
      }
    });
  }

  function schedule(marketOpen) {
    if (timer) clearTimeout(timer);
    timer = setTimeout(poll, marketOpen ? POLL_OPEN_MS : POLL_SHUT_MS);
  }

  function poll() {
    var url = "/api/ltp?symbols=" + encodeURIComponent(symbols.join(","));
    fetch(url, {
      credentials: "same-origin",
      headers: { Accept: "application/json" },
    })
      .then(function (res) {
        if (!res.ok) throw new Error("http " + res.status);
        return res.json();
      })
      .then(function (data) {
        var quotes = data.quotes || {};
        var open = !!data.market_open;
        nodes.forEach(function (node) {
          var sym = (node.getAttribute("data-live-symbol") || "").toUpperCase();
          render(node, quotes[sym], open);
        });
        renderStatus(open);
        schedule(open);
      })
      .catch(function () {
        // Network/auth hiccup: leave EOD values, retry on the slow cadence.
        schedule(false);
      });
  }

  // Pause polling when the tab is hidden to save the API budget; resume on focus.
  document.addEventListener("visibilitychange", function () {
    if (document.hidden) {
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
    } else if (!timer) {
      poll();
    }
  });

  poll();
})();
