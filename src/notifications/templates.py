"""Jinja2 templates for the daily notification.

Three render targets:

1. **HTML** -- the email body. Inline CSS only (so Gmail/Outlook render it
   identically). Auto-escaped to prevent injected HTML/JS from any
   malicious feature value (e.g. a manipulated symbol string).

2. **plaintext** -- the multipart/alternative MIME part for clients that
   refuse HTML (it's also a graceful fallback when the user previews on
   a phone with images off).

3. **WhatsApp** -- a tight, ~700-char text variant. WhatsApp does NOT
   render HTML at all; what we get is monospace + line breaks.

Security notes
--------------
- ``select_autoescape(['html'])`` is enabled. The HTML render passes user
  text through Jinja's HTML escaping, so newline / quote / angle-bracket
  characters in symbols or notes can never inject markup.
- Plaintext / WhatsApp versions strip CR/LF from any string field before
  inclusion, defending against terminal/log injection if the message is
  ever piped to a logger or shell.
"""

from __future__ import annotations

import re
from typing import Any

from jinja2 import Environment, select_autoescape

from src.notifications.report_builder import DailyReport

# ---------------------------------------------------------------------------
# Sanitisation helpers
# ---------------------------------------------------------------------------

_BAD_TEXT_CHARS = re.compile(r"[\r\n\x00]+")


def _scrub(s: Any) -> str:
    """Strip CR/LF/NUL from any string field before plaintext rendering."""
    if s is None:
        return ""
    return _BAD_TEXT_CHARS.sub(" ", str(s))


# ---------------------------------------------------------------------------
# Jinja env
# ---------------------------------------------------------------------------

_env = Environment(
    autoescape=select_autoescape(["html", "xml"]),
    trim_blocks=True,
    lstrip_blocks=True,
)
_env.filters["scrub"] = _scrub
_env.filters["pct"] = lambda v, d=2: ("n/a" if v is None else f"{float(v):.{d}f}%")
_env.filters["num"] = lambda v, d=2: ("n/a" if v is None else f"{float(v):,.{d}f}")


def _fmt(value, spec: str) -> str:
    """str.format-style filter so we can use thousands separators / signs.

    ``{{ x|fmt('+,.0f') }}`` is equivalent to ``format(x, '+,.0f')``.
    Jinja's built-in ``format`` filter uses Python's ``%`` operator which
    does NOT accept ``+,.0f`` style specs, so we ship our own.
    """
    if value is None:
        return "n/a"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


_env.filters["fmt"] = _fmt


# ---------------------------------------------------------------------------
# Template strings
# ---------------------------------------------------------------------------

# IMPORTANT: keep the HTML self-contained (inline CSS) for cross-mail-client
# rendering. We don't link external stylesheets because most clients block
# them. Tables are used for layout because Outlook still butchers flexbox.

_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ subject }}</title>
</head>
<body style="margin:0;padding:0;font-family:Arial,Helvetica,sans-serif;background:#f5f5f7;color:#222;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f5f5f7;">
    <tr><td align="center" style="padding:24px 8px;">
      <table role="presentation" width="640" cellpadding="0" cellspacing="0"
             style="background:#fff;border:1px solid #e2e2e6;border-radius:10px;overflow:hidden;">
        <tr><td style="background:#0b3d91;color:#fff;padding:18px 22px;">
          <div style="font-size:13px;letter-spacing:.06em;opacity:.85;">AI TRADER &mdash; DAILY REPORT</div>
          <div style="font-size:22px;font-weight:700;margin-top:4px;">{{ report.report_date }}</div>
          <div style="font-size:14px;opacity:.9;margin-top:6px;">{{ report.headline }}</div>
        </td></tr>

        <tr><td style="padding:18px 22px;">
          <h3 style="margin:0 0 8px;font-size:15px;color:#0b3d91;">Today's signals</h3>
          {% if report.signals %}
          <table cellpadding="6" cellspacing="0" width="100%"
                 style="border-collapse:collapse;font-size:13px;border:1px solid #e2e2e6;">
            <tr style="background:#eef2fb;text-align:left;">
              <th>Symbol</th><th>Calibrated&nbsp;prob</th><th>Decision</th>
            </tr>
            {% for s in report.signals %}
            <tr style="border-top:1px solid #e2e2e6;">
              <td style="font-family:Consolas,monospace;">{{ s.symbol|e }}</td>
              <td>{{ '%.4f'|format(s.calibrated_prob or 0) }}</td>
              <td style="color:#1f7a1f;font-weight:600;">BUY</td>
            </tr>
            {% endfor %}
          </table>
          {% else %}
          <p style="margin:0;color:#666;">No symbols crossed the threshold today
            ({{ '%.3f'|format(report.threshold_used or 0) }}). Stay flat.</p>
          {% endif %}
        </td></tr>

        {% if report.predictions %}
        <tr><td style="padding:0 22px 18px;">
          <h3 style="margin:0 0 8px;font-size:15px;color:#0b3d91;">
            Top {{ report.top_n }} predictions
          </h3>
          <table cellpadding="6" cellspacing="0" width="100%"
                 style="border-collapse:collapse;font-size:13px;border:1px solid #e2e2e6;">
            <tr style="background:#eef2fb;text-align:left;">
              <th>#</th><th>Symbol</th><th>Calibrated</th><th>Raw</th><th>Signal</th>
            </tr>
            {% for p in report.predictions %}
            <tr style="border-top:1px solid #e2e2e6;">
              <td>{{ loop.index }}</td>
              <td style="font-family:Consolas,monospace;">{{ p.symbol|e }}</td>
              <td>{{ '%.4f'|format(p.calibrated_prob or 0) }}</td>
              <td>{{ '%.4f'|format(p.raw_prob or 0) }}</td>
              <td>{{ 'YES' if p.is_signal else '' }}</td>
            </tr>
            {% endfor %}
          </table>
        </td></tr>
        {% endif %}

        <tr><td style="padding:0 22px 18px;">
          <h3 style="margin:0 0 8px;font-size:15px;color:#0b3d91;">Model snapshot</h3>
          {% if report.latest_model %}
          <table cellpadding="4" cellspacing="0" style="font-size:13px;">
            <tr><td style="color:#555;">Run ID</td>
                <td style="font-family:Consolas,monospace;">{{ report.latest_model.run_id|e }}</td></tr>
            <tr><td style="color:#555;">Trained window</td>
                <td>{{ report.latest_model.trained_from }} &rarr; {{ report.latest_model.trained_to }}</td></tr>
            <tr><td style="color:#555;">Threshold</td>
                <td>{{ '%.3f'|format(report.threshold_used or 0) }}</td></tr>
            <tr><td style="color:#555;">Git SHA</td>
                <td style="font-family:Consolas,monospace;">{{ (report.latest_model.git_sha or 'unknown')|e }}</td></tr>
          </table>
          {% else %}
          <p style="margin:0;color:#666;">No trained model yet. Run <code>train_model.py</code>.</p>
          {% endif %}
        </td></tr>

        <tr><td style="padding:0 22px 18px;">
          <h3 style="margin:0 0 8px;font-size:15px;color:#0b3d91;">Latest backtest</h3>
          {% if report.latest_backtest %}
          {% set m = report.latest_backtest.metrics %}
          <table cellpadding="4" cellspacing="0" style="font-size:13px;">
            <tr><td style="color:#555;">Name</td>
                <td>{{ report.latest_backtest.name|e }}</td></tr>
            <tr><td style="color:#555;">Window</td>
                <td>{{ report.latest_backtest.start_date }} &rarr; {{ report.latest_backtest.end_date }}</td></tr>
            <tr><td style="color:#555;">Sharpe / Sortino</td>
                <td>{{ '%.2f'|format(m.get('sharpe', 0) or 0) }} /
                    {{ '%.2f'|format(m.get('sortino', 0) or 0) }}</td></tr>
            <tr><td style="color:#555;">Max DD</td>
                <td>{{ '%.2f'|format(m.get('max_drawdown_pct', 0) or 0) }}%</td></tr>
            <tr><td style="color:#555;">Total return</td>
                <td>{{ '%.2f'|format(m.get('total_return_pct', 0) or 0) }}%</td></tr>
            <tr><td style="color:#555;">Trades / Hit-rate</td>
                <td>{{ m.get('n_trades', 0) }} /
                    {{ '%.1f'|format(m.get('hit_rate_pct', 0) or 0) }}%</td></tr>
          </table>
          {% else %}
          <p style="margin:0;color:#666;">No backtest runs found yet.</p>
          {% endif %}
        </td></tr>

        {% if report.recent_trades %}
        <tr><td style="padding:0 22px 18px;">
          <h3 style="margin:0 0 8px;font-size:15px;color:#0b3d91;">
            Recent trades (latest backtest)
          </h3>
          <table cellpadding="6" cellspacing="0" width="100%"
                 style="border-collapse:collapse;font-size:12px;border:1px solid #e2e2e6;">
            <tr style="background:#eef2fb;text-align:left;">
              <th>Symbol</th><th>Entry</th><th>Exit</th><th>Hold</th>
              <th>P&amp;L (Rs)</th><th>P&amp;L %</th><th>Reason</th>
            </tr>
            {% for t in report.recent_trades %}
            <tr style="border-top:1px solid #e2e2e6;">
              <td style="font-family:Consolas,monospace;">{{ t.symbol|e }}</td>
              <td>{{ t.entry_date }}</td>
              <td>{{ t.exit_date }}</td>
              <td>{{ t.holding_days }}d</td>
              <td style="color:{{ '#1f7a1f' if t.net_pnl >= 0 else '#a01818' }};">
                {{ '%.0f'|format(t.net_pnl) }}
              </td>
              <td style="color:{{ '#1f7a1f' if t.pnl_pct >= 0 else '#a01818' }};">
                {{ '%.2f'|format(t.pnl_pct) }}%
              </td>
              <td>{{ t.exit_reason|e }}</td>
            </tr>
            {% endfor %}
          </table>
        </td></tr>
        {% endif %}

        <tr><td style="padding:0 22px 18px;">
          <h3 style="margin:0 0 8px;font-size:15px;color:#0b3d91;">Paper portfolio</h3>
          <p style="margin:0 0 6px;font-size:13px;">
            Open: <strong>{{ report.paper.open_count }}</strong> &middot;
            Unrealised:
            <strong style="color:{{ '#1f7a1f' if report.paper.unrealised_pnl >= 0 else '#a01818' }};">
              Rs {{ report.paper.unrealised_pnl|fmt(',.0f') }}
            </strong> &middot;
            Realised ({{ report.paper.window_days }}d):
            <strong style="color:{{ '#1f7a1f' if report.paper.realised_pnl_window >= 0 else '#a01818' }};">
              Rs {{ report.paper.realised_pnl_window|fmt(',.0f') }}
            </strong> &middot;
            Win-rate: {{ '%.1f'|format(report.paper.win_rate_pct_window) }}%
          </p>
          {% if report.paper.open_positions %}
          <table cellpadding="6" cellspacing="0" width="100%"
                 style="border-collapse:collapse;font-size:12px;border:1px solid #e2e2e6;">
            <tr style="background:#eef2fb;text-align:left;">
              <th>Symbol</th><th>Qty</th><th>Entry</th><th>Last</th>
              <th>SL</th><th>TP</th><th>Hold</th><th>Unreal P&amp;L</th>
            </tr>
            {% for p in report.paper.open_positions %}
            <tr style="border-top:1px solid #e2e2e6;">
              <td style="font-family:Consolas,monospace;">{{ p.symbol|e }}</td>
              <td>{{ p.qty }}</td>
              <td>{{ '%.2f'|format(p.entry_price) }}</td>
              <td>{{ '%.2f'|format(p.last_close or 0) }}</td>
              <td>{{ '%.2f'|format(p.stop_loss or 0) }}</td>
              <td>{{ '%.2f'|format(p.take_profit or 0) }}</td>
              <td>{{ p.holding_days }}d</td>
              <td style="color:{{ '#1f7a1f' if p.unrealised_pnl >= 0 else '#a01818' }};">
                {{ p.unrealised_pnl|fmt(',.0f') }}
                ({{ '%.2f'|format(p.unrealised_pnl_pct) }}%)
              </td>
            </tr>
            {% endfor %}
          </table>
          {% else %}
          <p style="margin:0;color:#666;">No open paper positions.</p>
          {% endif %}
        </td></tr>

        {% if report.paper.closed_recent %}
        <tr><td style="padding:0 22px 18px;">
          <h3 style="margin:0 0 8px;font-size:15px;color:#0b3d91;">
            Recently closed paper trades (last {{ report.paper.window_days }} days)
          </h3>
          <table cellpadding="6" cellspacing="0" width="100%"
                 style="border-collapse:collapse;font-size:12px;border:1px solid #e2e2e6;">
            <tr style="background:#eef2fb;text-align:left;">
              <th>Symbol</th><th>Entry</th><th>Exit</th><th>Hold</th>
              <th>P&amp;L (Rs)</th><th>P&amp;L %</th><th>Reason</th>
            </tr>
            {% for t in report.paper.closed_recent %}
            <tr style="border-top:1px solid #e2e2e6;">
              <td style="font-family:Consolas,monospace;">{{ t.symbol|e }}</td>
              <td>{{ t.entry_date }}</td>
              <td>{{ t.exit_date }}</td>
              <td>{{ t.holding_days }}d</td>
              <td style="color:{{ '#1f7a1f' if t.net_pnl >= 0 else '#a01818' }};">
                {{ t.net_pnl|fmt(',.0f') }}
              </td>
              <td style="color:{{ '#1f7a1f' if t.pnl_pct >= 0 else '#a01818' }};">
                {{ '%.2f'|format(t.pnl_pct) }}%
              </td>
              <td>{{ t.exit_reason|e }}</td>
            </tr>
            {% endfor %}
          </table>
        </td></tr>
        {% endif %}

        <tr><td style="padding:0 22px 22px;">
          <h3 style="margin:0 0 6px;font-size:15px;color:#0b3d91;">Health check</h3>
          {% if report.freshness.is_stale %}
          <p style="margin:0 0 4px;color:#a01818;">
            <strong>Data is stale.</strong>
            Latest price bar: {{ report.freshness.latest_price_date or 'unknown' }}
            ({{ report.freshness.days_since_last_price or '?' }} days behind).
          </p>
          {% endif %}
          {% if report.validation.total == 0 %}
          <p style="margin:0;color:#1f7a1f;">
            No validation issues in the last {{ report.validation.window_days }} days.
          </p>
          {% else %}
          <p style="margin:0;color:#a01818;">
            {{ report.validation.total }} event(s) in the last {{ report.validation.window_days }} days.
            Severity breakdown:
            {% for sev, n in report.validation.by_severity.items() %}
              <strong>{{ sev|e }}={{ n }}</strong>{% if not loop.last %}, {% endif %}
            {% endfor %}.
          </p>
          {% endif %}
          <p style="margin:8px 0 0;font-size:12px;color:#888;">
            Universe size: {{ report.universe_size }} &middot;
            Latest features: {{ report.freshness.latest_feature_date or 'n/a' }} &middot;
            Generated at {{ report.generated_at }} (UTC) &middot;
            Auto-generated &mdash; <em>not investment advice</em>.
          </p>
        </td></tr>

      </table>
    </td></tr>
  </table>
</body>
</html>
"""

_TEXT_TEMPLATE = r"""AI TRADER -- DAILY REPORT
=========================
Date          : {{ report.report_date }}
Headline      : {{ report.headline|scrub }}
Universe size : {{ report.universe_size }}
Generated at  : {{ report.generated_at }} (UTC)

[1] Today's signals
{% if report.signals %}
{% for s in report.signals %}  - {{ s.symbol|scrub }}  prob={{ '%.4f'|format(s.calibrated_prob or 0) }}  decision=BUY
{% endfor %}
{% else %}  (no signals; threshold = {{ '%.3f'|format(report.threshold_used or 0) }})
{% endif %}

[2] Top {{ report.top_n }} predictions
{% if report.predictions %}
{% for p in report.predictions %}  {{ '%2d'|format(loop.index) }}. {{ '%-12s'|format(p.symbol|scrub) }}  cal={{ '%.4f'|format(p.calibrated_prob or 0) }}  raw={{ '%.4f'|format(p.raw_prob or 0) }}{% if p.is_signal %}  SIGNAL{% endif %}
{% endfor %}
{% else %}  (no predictions for this date)
{% endif %}

[3] Model snapshot
{% if report.latest_model %}  run_id          : {{ report.latest_model.run_id|scrub }}
  trained_window  : {{ report.latest_model.trained_from }} -> {{ report.latest_model.trained_to }}
  threshold       : {{ '%.3f'|format(report.threshold_used or 0) }}
  git_sha         : {{ (report.latest_model.git_sha or 'unknown')|scrub }}
{% else %}  (no trained model yet)
{% endif %}

[4] Latest backtest
{% if report.latest_backtest %}{% set m = report.latest_backtest.metrics %}  name      : {{ report.latest_backtest.name|scrub }}
  window    : {{ report.latest_backtest.start_date }} -> {{ report.latest_backtest.end_date }}
  Sharpe    : {{ '%.2f'|format(m.get('sharpe', 0) or 0) }}
  MaxDD     : {{ '%.2f'|format(m.get('max_drawdown_pct', 0) or 0) }}%
  TotalRet  : {{ '%.2f'|format(m.get('total_return_pct', 0) or 0) }}%
  Trades    : {{ m.get('n_trades', 0) }}  HitRate={{ '%.1f'|format(m.get('hit_rate_pct', 0) or 0) }}%
{% else %}  (no backtest runs)
{% endif %}

[5] Paper portfolio
  open={{ report.paper.open_count }}  unrealised={{ report.paper.unrealised_pnl|fmt('+,.0f') }}  realised_{{ report.paper.window_days }}d={{ report.paper.realised_pnl_window|fmt('+,.0f') }}  win_rate={{ '%.1f'|format(report.paper.win_rate_pct_window) }}%
{% if report.paper.open_positions %}{% for p in report.paper.open_positions %}    - {{ '%-12s'|format(p.symbol|scrub) }} qty={{ '%4d'|format(p.qty) }} entry={{ '%8.2f'|format(p.entry_price) }} last={{ '%8.2f'|format(p.last_close or 0) }} SL={{ '%8.2f'|format(p.stop_loss or 0) }} TP={{ '%8.2f'|format(p.take_profit or 0) }} hold={{ p.holding_days }}d unreal={{ p.unrealised_pnl|fmt('+,.0f') }} ({{ p.unrealised_pnl_pct|fmt('+.2f') }}%)
{% endfor %}{% else %}    (no open paper positions)
{% endif %}
[6] Recently closed paper trades (last {{ report.paper.window_days }} days)
{% if report.paper.closed_recent %}{% for t in report.paper.closed_recent %}  {{ '%-12s'|format(t.symbol|scrub) }} {{ t.entry_date }} -> {{ t.exit_date }} hold={{ t.holding_days }}d pnl={{ t.net_pnl|fmt('+,.0f') }} ({{ t.pnl_pct|fmt('+.2f') }}%) reason={{ t.exit_reason|scrub }}
{% endfor %}{% else %}  (none in window)
{% endif %}

[7] Health check
{% if report.freshness.is_stale %}  Data is STALE. Latest price bar: {{ report.freshness.latest_price_date or 'unknown' }} ({{ report.freshness.days_since_last_price or '?' }} days behind)
{% endif %}{% if report.validation.total == 0 %}  No validation issues in the last {{ report.validation.window_days }} days.
{% else %}  {{ report.validation.total }} event(s) in last {{ report.validation.window_days }} days
{% for sev, n in report.validation.by_severity.items() %}    {{ sev|scrub }}: {{ n }}
{% endfor %}
{% endif %}

-- Auto-generated. Not investment advice. --
"""

_WHATSAPP_TEMPLATE = r"""*AI TRADER -- {{ report.report_date }}*
{{ report.headline|scrub }}

{% if report.signals %}*Signals ({{ report.signals|length }}):*
{% for s in report.signals %}- {{ s.symbol|scrub }}  p={{ '%.3f'|format(s.calibrated_prob or 0) }}
{% endfor %}{% else %}_No BUY signals today (thr={{ '%.3f'|format(report.threshold_used or 0) }})._
{% endif %}
{% if report.paper.open_count > 0 %}*Open ({{ report.paper.open_count }}):*
{% for p in report.paper.open_positions[:5] %}- {{ p.symbol|scrub }} q={{ p.qty }} {{ p.unrealised_pnl_pct|fmt('+.1f') }}%
{% endfor %}{% endif %}{% if report.latest_backtest %}{% set m = report.latest_backtest.metrics %}_Last BT:_ Sharpe={{ '%.2f'|format(m.get('sharpe', 0) or 0) }} | DD={{ '%.1f'|format(m.get('max_drawdown_pct', 0) or 0) }}%
{% endif %}_Realised{{ report.paper.window_days }}d:_ {{ report.paper.realised_pnl_window|fmt('+,.0f') }} | _WinRate:_ {{ '%.0f'|format(report.paper.win_rate_pct_window) }}%
{% if report.freshness.is_stale %}*WARN:* data {{ report.freshness.days_since_last_price }}d stale.
{% endif %}"""


# Pre-compile so render() is cheap when called repeatedly (hourly digests).
_HTML = _env.from_string(_HTML_TEMPLATE)
_TEXT = _env.from_string(_TEXT_TEMPLATE)
_WHATSAPP = _env.from_string(_WHATSAPP_TEMPLATE)


# ---------------------------------------------------------------------------
# Public render API
# ---------------------------------------------------------------------------


def render_html(report: DailyReport, *, subject: str) -> str:
    return _HTML.render(report=report, subject=subject)


def render_text(report: DailyReport) -> str:
    return _TEXT.render(report=report)


def render_whatsapp(report: DailyReport, *, max_chars: int = 700) -> str:
    """Render the WhatsApp variant and hard-trim to ``max_chars``.

    WhatsApp will silently drop overflow on most providers, so we do the
    trim explicitly with an ellipsis and log when it happens.
    """
    rendered = _WHATSAPP.render(report=report).strip()
    if len(rendered) <= max_chars:
        return rendered
    return rendered[: max_chars - 1].rstrip() + "\u2026"
