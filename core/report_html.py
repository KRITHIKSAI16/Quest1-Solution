"""builds a single self-contained report.html. no server, no external
requests at view time. CSS is embedded and the frame gets inlined as base64
instead of a sibling PNG, so the file works on its own if you move it.
"""
from __future__ import annotations

import base64
import html
import os

from core.report import Result, format_timestamp

_OUTCOME_LABELS = {
    "CONFIRMED_BY_AUDIO": ("Confirmed · Audio", "ok"),
    "CONFIRMED_BY_VISUAL": ("Confirmed · Visual", "ok"),
    "CORROBORATED": ("Corroborated · Both channels", "ok"),
    "AMBIGUOUS": ("Ambiguous · Channels disagree", "warn"),
    "NOT_FOUND": ("Not found", "fail"),
}

_CSS = """
:root {
  --navy: #0b1c3d;
  --navy-2: #132b57;
  --orange: #ee6c2c;
  --bg: #ffffff;
  --panel: #f5f7fb;
  --border: #e3e7ee;
  --muted: #6b7684;
  --text: #10182b;
  --ok: #1a8a54;
  --ok-bg: #eaf7f0;
  --warn: #b5720a;
  --warn-bg: #fdf3e2;
  --fail: #b23a3a;
  --fail-bg: #fbecec;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.topbar { height: 6px; background: var(--navy); }
.navbar {
  display: flex; align-items: center; gap: 10px;
  padding: 14px 32px; border-bottom: 1px solid var(--border);
}
.logo {
  width: 22px; height: 22px; border-radius: 5px;
  background: linear-gradient(135deg, var(--navy) 55%, var(--orange) 55%);
}
.brand { font-weight: 800; color: var(--navy); letter-spacing: -0.01em; }
.breadcrumb {
  font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
  color: var(--muted); font-size: 13px; margin-left: 2px;
}
.navbar .spacer { flex: 1; }
.badge {
  font-size: 12px; font-weight: 600; padding: 5px 12px; border-radius: 999px;
  white-space: nowrap;
}
.badge.ok { color: var(--ok); background: var(--ok-bg); }
.badge.warn { color: var(--warn); background: var(--warn-bg); }
.badge.fail { color: var(--fail); background: var(--fail-bg); }
main { max-width: 980px; margin: 0 auto; padding: 48px 32px 64px; }
h1 {
  font-size: 40px; font-weight: 800; letter-spacing: -0.02em;
  color: var(--navy); margin: 0 0 6px;
}
h1 em { color: var(--orange); font-style: italic; }
.subtitle { color: var(--muted); font-size: 15px; margin: 0 0 32px; }
.target-card {
  border: 1px solid var(--border); background: var(--panel);
  border-radius: 10px; padding: 16px 20px; margin-bottom: 32px;
}
.target-label {
  font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 6px;
}
.target-text {
  font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
  font-size: 15px; color: var(--text);
}
.target-text::before { content: "\\201C"; color: var(--orange); }
.target-text::after { content: "\\201D"; color: var(--orange); }
/* same styling as target-text, just green instead of orange */
.extracted-text::before { color: var(--ok); }
.extracted-text::after { color: var(--ok); }
.evidence-label {
  font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--muted); margin: 28px 0 8px;
}
.grid { display: grid; grid-template-columns: 1.15fr 1fr; gap: 24px; }
@media (max-width: 720px) { .grid { grid-template-columns: 1fr; } }
.card {
  border: 1px solid var(--border); border-radius: 12px; overflow: hidden;
  background: var(--bg);
}
.frame-img { display: block; width: 100%; height: auto; background: var(--panel); }
.frame-caption {
  padding: 10px 16px; font-size: 12px; color: var(--muted);
  border-top: 1px solid var(--border);
  font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
}
.meta { padding: 22px; display: flex; flex-direction: column; gap: 18px; }
.stat-label {
  font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 4px;
}
.stat-value { font-size: 15px; font-weight: 600; color: var(--text); }
.stat-value.big {
  font-size: 26px; font-weight: 800; color: var(--navy);
  font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
}
.confidence-bar {
  height: 8px; border-radius: 999px; background: var(--panel); overflow: hidden; margin-top: 6px;
}
.confidence-fill { height: 100%; background: var(--orange); border-radius: 999px; }
.note {
  font-size: 13px; color: var(--warn); background: var(--warn-bg);
  border-radius: 8px; padding: 10px 14px; margin-top: 4px;
}
.empty-card { padding: 40px 32px; text-align: center; }
.empty-card .badge { display: inline-block; margin-bottom: 14px; }
.empty-card p { color: var(--muted); font-size: 15px; max-width: 480px; margin: 0 auto; }
table { width: 100%; border-collapse: collapse; margin-top: 24px; }
th, td {
  text-align: left; padding: 10px 16px; font-size: 13px; border-bottom: 1px solid var(--border);
}
th { color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em; }
td.mono { font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; }
footer {
  max-width: 980px; margin: 0 auto; padding: 20px 32px 40px;
  color: var(--muted); font-size: 12px; border-top: 1px solid var(--border);
  font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
}
"""


def _badge(outcome: str) -> str:
    label, kind = _OUTCOME_LABELS.get(outcome, (outcome, "warn"))
    return f'<span class="badge {kind}">{html.escape(label)}</span>'


def _image_data_uri(image_path: str) -> str | None:
    if not image_path or not os.path.isfile(image_path):
        return None
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _candidate_table(candidates: list[dict]) -> str:
    """table markup, reused whenever more than one channel found something."""
    rows = "".join(f"""
        <tr>
          <td class="mono">{html.escape(c['channel'])}</td>
          <td class="mono">{html.escape(format_timestamp(c['timestamp_s']))}</td>
          <td class="mono">{c['confidence']:.1f}</td>
          <td>{html.escape(c['matched_text'])}</td>
        </tr>""" for c in candidates)
    return f"""
    <div class="evidence-label">Per-channel evidence</div>
    <table>
      <thead><tr><th>Channel</th><th>Timestamp</th><th>Confidence</th><th>Extracted text</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def _found_section(result: Result) -> str:
    """everything except NOT_FOUND. usually there's a primary candidate (stat
    panel + image), but an unresolved AMBIGUOUS has no single timestamp, so
    it just gets the comparison table instead.
    """
    has_primary = result.timestamp is not None
    data_uri = _image_data_uri(result.image_path or "") if has_primary else None
    img_html = (
        f'<img class="frame-img" src="{data_uri}" alt="Extracted frame">'
        if data_uri else
        '<div class="frame-img" style="aspect-ratio:16/9;display:flex;align-items:center;'
        'justify-content:center;color:var(--muted);font-size:13px;">'
        f'{"Frame image unavailable" if has_primary else "Channels disagree — no single frame selected"}</div>'
    )
    note_html = f'<div class="note">{html.escape(result.ambiguity_status)}</div>' if result.ambiguity_status else ""

    if has_primary:
        confidence = result.confidence or 0.0
        channel_label = {"audio": "Audio", "visual": "Visual"}.get(result.channel or "", result.channel or "—")
        extracted_html = f"""
          <div>
            <div class="stat-label">Extracted text</div>
            <div class="target-text extracted-text stat-value">{html.escape(result.text or "")}</div>
          </div>""" if result.text else ""
        stats_html = f"""
          <div>
            <div class="stat-label">Timestamp</div>
            <div class="stat-value big">{html.escape(result.timestamp or "—")}</div>
          </div>
          <div>
            <div class="stat-label">Frame number</div>
            <div class="stat-value">{html.escape(str(result.frame_number))}</div>
          </div>
          <div>
            <div class="stat-label">Channel</div>
            <div class="stat-value">{html.escape(channel_label)}</div>
          </div>
          <div>
            <div class="stat-label">Confidence</div>
            <div class="stat-value">{confidence:.1f}</div>
            <div class="confidence-bar"><div class="confidence-fill" style="width:{max(0, min(100, confidence)):.1f}%;"></div></div>
          </div>"""
    else:
        extracted_html = ""
        stats_html = ""

    frame_caption = f"frame_{html.escape(str(result.frame_number))}.png" if has_primary else "no frame saved"
    table_html = _candidate_table(result.all_candidates) if len(result.all_candidates) > 1 else ""

    return f"""
    <div class="grid">
      <div class="card">
        {img_html}
        <div class="frame-caption">{frame_caption}</div>
      </div>
      <div class="card">
        <div class="meta">
          <div>{_badge(result.outcome)}</div>
          {extracted_html}
          {stats_html}
          {note_html}
        </div>
      </div>
    </div>
    {table_html}
    """


def _not_found_section(result: Result) -> str:
    rows = ""
    for nm in result.near_misses:
        rows += f"""
        <tr>
          <td class="mono">{html.escape(nm['channel'])}</td>
          <td class="mono">{html.escape(format_timestamp(nm['timestamp_s']))}</td>
          <td class="mono">{nm['confidence']:.1f}</td>
          <td>{html.escape(nm['text'])}</td>
        </tr>"""

    table_html = f"""
    <table>
      <thead><tr><th>Channel</th><th>Timestamp</th><th>Score</th><th>Nearest text</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>""" if result.near_misses else ""

    return f"""
    <div class="card empty-card">
      {_badge(result.outcome)}
      <p>No confident match for the target text was found in either channel.</p>
      {table_html}
    </div>
    """


def generate_html_report(result: Result, target_text: str, source: str, out_dir: str) -> str:
    """writes report.html next to result.json, returns the path. NOT_FOUND
    isn't treated as an error here, it's just another outcome to render.
    """
    # keying off outcome rather than result.timestamp, since an unresolved
    # AMBIGUOUS also has no timestamp but isn't a failed search
    body = _not_found_section(result) if result.outcome == "NOT_FOUND" else _found_section(result)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Quest1 Dialogue Extraction Report</title>
<style>{_CSS}</style>
</head>
<body>
  <div class="topbar"></div>
  <div class="navbar">
    <div class="logo"></div>
    <span class="brand">Quest1</span>
    <span class="breadcrumb">~/quest1/report</span>
    <div class="spacer"></div>
    {_badge(result.outcome)}
  </div>
  <main>
    <h1>Dialogue <em>Extraction</em> Report</h1>
    <p class="subtitle">Frame &amp; dialogue extractor — dual-channel OCR + ASR pipeline</p>
    <div class="target-card">
      <div class="target-label">Target text searched</div>
      <div class="target-text">{html.escape(target_text)}</div>
    </div>
    {body}
  </main>
  <footer>source: {html.escape(source)}</footer>
</body>
</html>
"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return path
