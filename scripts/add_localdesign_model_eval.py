#!/usr/bin/env python3
"""Merge LocalDesign prompt-to-UI evaluation records into the backend report.

This is deliberately a separate pass from llama-bench: API generation latency,
output throughput, and checklist coverage are a different measurement method.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from urllib.parse import quote

START = "<!-- LOCALDESIGN_GENERATION_BENCHMARKS_START -->"
END = "<!-- LOCALDESIGN_GENERATION_BENCHMARKS_END -->"


def load_runs(evidence_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(evidence_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # Standard LocalDesign generation-evaluation shape. Vision-only,
        # diffusion, workflow, and infrastructure fixtures are not generation
        # cases and remain in their own reports/sections.
        if not all(k in data for k in ("modelLabel", "caseId", "usage", "score")):
            continue
        if data.get("apiStatus") != 201 or not isinstance(data.get("score"), dict):
            continue
        rel = path.relative_to(evidence_dir)
        backend = backend_for(rel)
        rows.append({"data": data, "path": rel.as_posix(), "backend": backend})
    return sorted(rows, key=lambda r: (r["data"].get("createdAt", ""), r["data"].get("modelLabel", ""), r["data"].get("caseId", "")))


def backend_for(rel: Path) -> str:
    s = rel.as_posix().lower()
    if "openvino" in s:
        return "OpenVINO GPU"
    if "sycl-opencl" in s or "sycl" in s or "bonsai" in s or "live-2026-09-30" in s or s[:10].startswith("2026-09-28") or "sharp-minicpm5" in s:
        return "SYCL0"
    if "vulkan" in s:
        return "Vulkan"
    return "PrismML backend (see source)"


def model_identity(value: object) -> str:
    # Do not expose host-specific absolute model paths in the public report.
    text = str(value or "unknown")
    return Path(text).name if "/" in text else text


def esc(value: object) -> str:
    return html.escape(str(value if value is not None else "—"), quote=True)


def render_section(rows: list[dict], evidence_dir: Path) -> str:
    labels = {r["data"].get("modelLabel", "unknown") for r in rows}
    identities = {model_identity(r["data"].get("model")) for r in rows}
    throughputs = [float(r["data"]["outputTokensPerSecond"]) for r in rows if r["data"].get("outputTokensPerSecond") is not None]
    maximum = max(throughputs, default=1.0)
    chart_rows = sorted(rows, key=lambda r: (float(r["data"].get("outputTokensPerSecond") or 0), r["data"].get("modelLabel", "")), reverse=True)
    chart_h = 54 + 27 * len(chart_rows)
    svg = [f'<svg class="chart" style="height:auto" viewBox="0 0 1120 {chart_h}" role="img" aria-label="LocalDesign API output tokens per second for all recorded generation cases">',
           '<text class="chart-title" x="18" y="27">LocalDesign prompt-to-design output throughput</text>',
           '<text class="chart-unit" x="1100" y="27" text-anchor="end">completion tokens/s · per run</text>']
    colors = {"Vulkan": "#6ea8fe", "SYCL0": "#70d6a5", "OpenVINO GPU": "#f6bd60", "PrismML backend (see source)": "#bc9aff"}
    for i, row in enumerate(chart_rows):
        d = row["data"]
        rate = float(d.get("outputTokensPerSecond") or 0)
        y = 45 + i * 27
        label = f"{d.get('modelLabel', 'unknown')} · {'Garden' if d.get('caseId') == 'garden-dashboard' else 'Fieldnotes' if d.get('caseId') == 'fieldnotes-detailed' else d.get('caseId', '')}"
        short = label if len(label) <= 43 else label[:40] + "…"
        svg.append(f'<title>{esc(label)} · {esc(row["backend"])} · {rate:g} tokens/s</title>')
        svg.append(f'<text class="axis-label" x="14" y="{y+13}" text-anchor="start">{esc(short)}</text>')
        svg.append(f'<rect x="440" y="{y}" width="{max(1, 560*rate/maximum):.2f}" height="17" rx="4" fill="{colors.get(row["backend"], "#bc9aff")}"/>')
        svg.append(f'<text class="bar-value" x="{min(1007, 447+560*rate/maximum):.2f}" y="{y+13}">{rate:g}</text>')
    svg.append('</svg>')

    body = []
    for row in rows:
        d = row["data"]
        score = d.get("score", {})
        usage = d.get("usage", {})
        ms = d.get("modelDurationMs") or d.get("totalElapsedMs")
        duration = f"{float(ms)/1000:.1f}" if ms is not None else "—"
        rate = d.get("outputTokensPerSecond")
        coverage = f"{score.get('coveredCount', '—')}/{score.get('requirementCount', '—')}"
        status = "Complete HTML" if score.get("completeHtml") else "Incomplete HTML"
        source = "../../../../LocalDesign/docs/evidence/model-evaluation/" + quote(row["path"], safe="/._-")
        created_raw = str(d.get("createdAt", ""))
        try:
            created = datetime.fromisoformat(created_raw.replace("Z", "+00:00")).astimezone(ZoneInfo("Australia/Perth")).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            created = created_raw[:16].replace("T", " ") or "—"
        body.append("<tr>" + "".join([
            f"<td>{esc(created)}</td>", f"<td>{esc(d.get('modelLabel'))}<br><small>{esc(model_identity(d.get('model')))}</small></td>",
            f"<td>{esc(d.get('title') or d.get('caseId'))}</td>", f"<td>{esc(row['backend'])}</td>",
            f"<td>{esc(usage.get('completionTokens'))}</td>", f"<td>{esc(f'{float(rate):.1f}' if rate is not None else '—')}</td>",
            f"<td>{esc(duration)}</td>", f"<td>{esc(coverage)}</td>", f"<td>{esc(status)}<br><a href=\"{source}\">source JSON</a></td>",
        ]) + "</tr>")
    table = '<div style="overflow:auto"><table><thead><tr><th>Run (AWST)</th><th>Model / quant</th><th>Prompt case</th><th>Backend</th><th>Output tokens</th><th>Output tok/s</th><th>Model time (s)</th><th>Checklist</th><th>Validation / evidence</th></tr></thead><tbody>' + "".join(body) + "</tbody></table></div>"
    backend_counts = Counter(r["backend"] for r in rows)
    legend = " · ".join(f'<span style="color:{colors.get(k, "#bc9aff")}">■ {esc(k)}: {v}</span>' for k, v in sorted(backend_counts.items()))
    return (f'{START}<section id="localdesign-model-generation-benchmarks-20261001" class="panel">'
            f'<h2>LocalDesign model generation benchmarks</h2>'
            f'<p>Supplementary application-level evaluation: <b>{len(labels)} model configurations</b>, <b>{len(identities)} recorded model identities</b>, and <b>{len(rows)} accepted API generation cases</b> across the saved trials. This expands the report beyond the six llama-bench model files; it does not alter or mix measurements into the llama-bench backend charts.</p>'
            f'<p class="meta">Each bar is one prompt run, sorted by observed completion tokens/s. Colors identify the requested backend. Runs span multiple dates, model builds, context/output limits, and test rounds, so treat this as an evidence index and exploratory comparison rather than a controlled cross-backend ranking. Checklist coverage is exact phrase matching; “Complete HTML” only indicates document completeness, not visual quality. One accepted case is recorded with incomplete HTML and remains visible.</p>'
            f'<p class="meta">{legend}</p><div style="overflow:auto">{"".join(svg)}</div>{table}'
            f'<p class="meta">Diffusion model trials, vision-only checks, failed/rejected attempts, and workflow acceptance fixtures are intentionally not counted as text-generation API cases; they remain in their dedicated report panels/evidence. All case rows link to the original local JSON record.</p>'
            f'</section>{END}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/benchmarks/backend-report.html")
    parser.add_argument("--evidence-dir", type=Path, default=Path(__file__).resolve().parents[3] / "LocalDesign/docs/evidence/model-evaluation")
    args = parser.parse_args()
    report = args.report.resolve()
    evidence = args.evidence_dir.resolve()
    if not evidence.is_dir():
        parser.error(f"Evidence directory not found: {evidence}")
    page = report.read_text(encoding="utf-8")
    rows = load_runs(evidence)
    if not rows:
        parser.error(f"No accepted generation records found in {evidence}")
    section = render_section(rows, evidence)
    if START in page and END in page:
        page = re.sub(re.escape(START) + r".*?" + re.escape(END), section, page, count=1, flags=re.S)
    else:
        anchor = '<section id="localdesign-image-benchmarks-20260930"'
        if anchor not in page:
            parser.error("Could not locate the supplemental report insertion point")
        page = page.replace(anchor, section + "\n" + anchor, 1)
    page = page.replace("<span>models discovered</span>", "<span>llama-bench models discovered</span>", 1)
    page = page.replace("repeat(auto-fit,minmax(170px,1fr))", "repeat(auto-fit,minmax(150px,1fr))", 1)
    cards_match = re.search(r'<section class="cards">.*?</section>', page, flags=re.S)
    if cards_match:
        cards = re.sub(r'<div class="card localdesign-card">.*?</div>', "", cards_match.group(0), flags=re.S)
        extra = (f'<div class="card localdesign-card"><strong>{len({r["data"].get("modelLabel") for r in rows})}</strong><span>LocalDesign model configs</span></div>'
                 f'<div class="card localdesign-card"><strong>{len({model_identity(r["data"].get("model")) for r in rows})}</strong><span>LocalDesign model identities</span></div>'
                 f'<div class="card localdesign-card"><strong>{len(rows)}</strong><span>LocalDesign API generation cases</span></div>')
        cards = cards[:-10] + extra + "</section>"
        page = page[:cards_match.start()] + cards + page[cards_match.end():]
    report.write_text(page, encoding="utf-8")
    print(f"Updated {report}: {len(rows)} API generation cases, {len({r['data'].get('modelLabel') for r in rows})} configs, {len({model_identity(r['data'].get('model')) for r in rows})} identities")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
