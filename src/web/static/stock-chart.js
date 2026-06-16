/* Per-stock detail chart bootstrap.
 *
 * Kept as an external file (not inline) so the page CSP can stay strict
 * (`script-src 'self'`, no 'unsafe-inline'). Reads the symbol from the
 * #price-chart container's data attribute, fetches /api/ohlc/<symbol>, and
 * renders a candlestick chart with EMA overlays + volume, plus RSI and MACD
 * sub-panels, using the vendored TradingView lightweight-charts library.
 */
(function () {
  "use strict";

  var LWC = window.LightweightCharts;
  var priceEl = document.getElementById("price-chart");
  var rsiEl = document.getElementById("rsi-chart");
  var macdEl = document.getElementById("macd-chart");
  var errEl = document.getElementById("chart-error");

  function showError() {
    if (errEl) errEl.classList.remove("hidden");
  }

  if (!LWC || !priceEl) {
    showError();
    return;
  }

  var symbol = priceEl.getAttribute("data-symbol");
  if (!symbol) {
    showError();
    return;
  }

  var darkLayout = {
    layout: { background: { color: "transparent" }, textColor: "#94a3b8" },
    grid: {
      vertLines: { color: "rgba(148,163,184,0.08)" },
      horzLines: { color: "rgba(148,163,184,0.08)" },
    },
    rightPriceScale: { borderColor: "rgba(148,163,184,0.2)" },
    timeScale: { borderColor: "rgba(148,163,184,0.2)" },
    crosshair: { mode: 0 },
  };

  function mkChart(el, height) {
    return LWC.createChart(el, Object.assign({
      width: el.clientWidth,
      height: height,
      handleScale: true,
      handleScroll: true,
    }, darkLayout));
  }

  fetch("/api/ohlc/" + encodeURIComponent(symbol), {
    headers: { "Accept": "application/json" },
  })
    .then(function (r) {
      if (!r.ok) throw new Error("ohlc " + r.status);
      return r.json();
    })
    .then(function (data) {
      if (!data || !data.candles || data.candles.length === 0) {
        showError();
        return;
      }

      // ---- Price chart: candles + EMAs + volume ----
      var priceChart = mkChart(priceEl, priceEl.clientHeight || 360);
      var candle = priceChart.addCandlestickSeries({
        upColor: "#22c55e", downColor: "#ef4444",
        borderUpColor: "#22c55e", borderDownColor: "#ef4444",
        wickUpColor: "#22c55e", wickDownColor: "#ef4444",
      });
      candle.setData(data.candles);

      var vol = priceChart.addHistogramSeries({
        priceFormat: { type: "volume" },
        priceScaleId: "",
      });
      vol.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
      if (data.volume) vol.setData(data.volume);

      function addLine(chart, points, color, width) {
        if (!points || points.length === 0) return;
        var s = chart.addLineSeries({
          color: color, lineWidth: width || 2, priceLineVisible: false,
          lastValueVisible: false, crosshairMarkerVisible: false,
        });
        s.setData(points);
      }
      addLine(priceChart, data.ema20, "#3b82f6");
      addLine(priceChart, data.ema50, "#f59e0b");
      addLine(priceChart, data.ema200, "#a855f7");

      // ---- Target / stop / last reference lines ----
      var levels = data.levels || {};
      function priceLine(price, color, title) {
        if (price === undefined || price === null) return;
        candle.createPriceLine({
          price: price, color: color, lineWidth: 1, lineStyle: 2,
          axisLabelVisible: true, title: title,
        });
      }
      priceLine(levels.target, "#22c55e", "Target");
      priceLine(levels.stop, "#ef4444", "Stop");

      // ---- RSI sub-panel ----
      if (rsiEl && data.rsi && data.rsi.length) {
        var rsiChart = mkChart(rsiEl, rsiEl.clientHeight || 110);
        var rsi = rsiChart.addLineSeries({ color: "#38bdf8", lineWidth: 1.5, lastValueVisible: true });
        rsi.setData(data.rsi);
        rsi.createPriceLine({ price: 70, color: "rgba(239,68,68,0.5)", lineWidth: 1, lineStyle: 2, axisLabelVisible: false });
        rsi.createPriceLine({ price: 30, color: "rgba(34,197,94,0.5)", lineWidth: 1, lineStyle: 2, axisLabelVisible: false });
        syncTime(priceChart, rsiChart);
      }

      // ---- MACD sub-panel ----
      if (macdEl && data.macd && data.macd.length) {
        var macdChart = mkChart(macdEl, macdEl.clientHeight || 120);
        if (data.macd_hist && data.macd_hist.length) {
          var hist = macdChart.addHistogramSeries({});
          hist.setData(data.macd_hist);
        }
        var macdLine = macdChart.addLineSeries({ color: "#60a5fa", lineWidth: 1.5, lastValueVisible: false });
        macdLine.setData(data.macd);
        if (data.macd_signal && data.macd_signal.length) {
          var sig = macdChart.addLineSeries({ color: "#f59e0b", lineWidth: 1.5, lastValueVisible: false });
          sig.setData(data.macd_signal);
        }
        syncTime(priceChart, macdChart);
      }

      priceChart.timeScale().fitContent();

      // ---- Responsive resize ----
      window.addEventListener("resize", function () {
        priceChart.applyOptions({ width: priceEl.clientWidth });
      });
    })
    .catch(function () {
      showError();
    });

  // Keep sub-panel time scales aligned with the main chart.
  function syncTime(main, other) {
    try {
      main.timeScale().subscribeVisibleLogicalRangeChange(function (range) {
        if (range) other.timeScale().setVisibleLogicalRange(range);
      });
      other.timeScale().subscribeVisibleLogicalRangeChange(function (range) {
        if (range) main.timeScale().setVisibleLogicalRange(range);
      });
    } catch (e) { /* non-fatal */ }
  }
})();
