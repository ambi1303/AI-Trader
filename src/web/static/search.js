/* Stock search autocomplete.
 *
 * Debounced lookups against /api/search, a keyboard-navigable dropdown, and
 * navigation to /stock/<SYMBOL>. CSP-safe: external file, no inline handlers,
 * and results are rendered with textContent (never innerHTML) so a symbol can
 * never inject markup.
 */
(function () {
  "use strict";

  var input = document.getElementById("stock-search");
  var box = document.getElementById("stock-search-results");
  if (!input || !box) return;

  var items = [];      // current result symbols
  var active = -1;     // highlighted index
  var timer = null;

  function clean(s) {
    return (s || "").toUpperCase().replace(/[^A-Z0-9&-]/g, "");
  }

  function go(symbol) {
    var sym = clean(symbol);
    if (sym) window.location.href = "/stock/" + encodeURIComponent(sym);
  }

  function hide() {
    box.classList.add("hidden");
    box.replaceChildren();
    items = [];
    active = -1;
  }

  function highlight() {
    var children = box.children;
    for (var i = 0; i < children.length; i++) {
      if (i === active) {
        children[i].classList.add("bg-slate-800");
      } else {
        children[i].classList.remove("bg-slate-800");
      }
    }
  }

  function render(symbols) {
    box.replaceChildren();
    items = symbols || [];
    active = -1;
    if (!items.length) {
      hide();
      return;
    }
    items.forEach(function (sym, idx) {
      var row = document.createElement("button");
      row.type = "button";
      row.className =
        "w-full text-left px-3 py-2 text-sm text-slate-100 hover:bg-slate-800 " +
        "flex items-center justify-between";
      var name = document.createElement("span");
      name.className = "font-medium tracking-tight";
      name.textContent = sym;                 // safe: textContent
      var hint = document.createElement("span");
      hint.className = "text-xs text-slate-500";
      hint.textContent = "Analyze \u2192";
      row.appendChild(name);
      row.appendChild(hint);
      row.addEventListener("mousedown", function (e) {
        e.preventDefault();                    // keep focus; fire before blur
        go(sym);
      });
      row.addEventListener("mouseenter", function () {
        active = idx;
        highlight();
      });
      box.appendChild(row);
    });
    box.classList.remove("hidden");
  }

  function query(term) {
    fetch("/api/search?term=" + encodeURIComponent(term), {
      headers: { "Accept": "application/json" },
      credentials: "same-origin"
    })
      .then(function (r) { return r.ok ? r.json() : { results: [] }; })
      .then(function (data) { render((data && data.results) || []); })
      .catch(function () { hide(); });
  }

  input.addEventListener("input", function () {
    var term = clean(input.value);
    if (timer) clearTimeout(timer);
    if (term.length < 1) {
      hide();
      return;
    }
    timer = setTimeout(function () { query(term); }, 180);
  });

  input.addEventListener("keydown", function (e) {
    if (e.key === "ArrowDown" && items.length) {
      e.preventDefault();
      active = (active + 1) % items.length;
      highlight();
    } else if (e.key === "ArrowUp" && items.length) {
      e.preventDefault();
      active = (active - 1 + items.length) % items.length;
      highlight();
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (active >= 0 && items[active]) {
        go(items[active]);
      } else if (input.value.trim()) {
        go(input.value);                       // type a ticker + Enter
      }
    } else if (e.key === "Escape") {
      hide();
    }
  });

  document.addEventListener("click", function (e) {
    if (!box.contains(e.target) && e.target !== input) hide();
  });
})();
