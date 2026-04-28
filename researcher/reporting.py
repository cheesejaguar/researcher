"""Evidence matrix and cited report rendering for completed runs."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import duckdb


def evidence_rows(db_path: Path, run_id: str | None = None) -> list[dict[str, Any]]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        field_rows = con.execute(
            """
            SELECT e.id, e.entity_type, e.name,
                   f.field_name, f.value_json, f.confidence,
                   f.provenance_ids_json, f.last_seen_run
            FROM fields f
            JOIN entities e ON e.id = f.entity_id
            WHERE f.value_json != 'null'
            ORDER BY e.entity_type, e.name, f.field_name
            """
        ).fetchall()
        prov = {
            row[0]: json.loads(row[1])
            for row in con.execute("SELECT provenance_id, data_json FROM provenance").fetchall()
        }
        conflicts = {
            (row[0], row[1]): row[2]
            for row in con.execute(
                "SELECT entity_id, field, status FROM conflicts"
            ).fetchall()
        }
        rows: list[dict[str, Any]] = []
        for entity_id, entity_type, name, field, value_json, confidence, prov_json, last_seen_run in field_rows:
            provenance_ids = json.loads(prov_json or "[]")
            provenance_id = provenance_ids[0] if provenance_ids else ""
            pdata = prov.get(provenance_id, {})
            try:
                value = json.loads(value_json)
            except Exception:
                value = value_json
            rows.append(
                {
                    "run_id": run_id,
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "entity_name": name,
                    "field": field,
                    "value": value,
                    "confidence": float(confidence),
                    "provenance_id": provenance_id,
                    "source_url": pdata.get("url", ""),
                    "quote": pdata.get("snippet", ""),
                    "conflict_status": conflicts.get((entity_id, field), ""),
                    "last_seen_run": last_seen_run,
                }
            )
        return rows
    finally:
        con.close()


def source_summary(db_path: Path) -> list[dict[str, Any]]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            """
            SELECT s.source_id, s.title, s.url, s.source_type, s.path,
                   COUNT(c.chunk_id) AS chunks
            FROM sources s
            LEFT JOIN source_chunks c ON c.source_id = s.source_id
            GROUP BY s.source_id, s.title, s.url, s.source_type, s.path
            ORDER BY s.title, s.url
            """
        ).fetchall()
    except duckdb.Error:
        rows = []
    finally:
        con.close()
    return [
        {
            "source_id": r[0],
            "title": r[1],
            "url": r[2],
            "source_type": r[3],
            "path": r[4],
            "chunks": int(r[5]),
        }
        for r in rows
    ]


def verification_votes(db_path: Path, run_id: str | None = None) -> list[dict[str, Any]]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        if run_id:
            rows = con.execute(
                """
                SELECT vote_id, run_id, entity_id, field_name, model,
                       vote_json, confidence, rationale, created_at
                FROM verification_votes
                WHERE run_id = ?
                ORDER BY created_at, model
                """,
                [run_id],
            ).fetchall()
        else:
            rows = con.execute(
                """
                SELECT vote_id, run_id, entity_id, field_name, model,
                       vote_json, confidence, rationale, created_at
                FROM verification_votes
                ORDER BY created_at, model
                """
            ).fetchall()
    except duckdb.Error:
        rows = []
    finally:
        con.close()
    return [
        {
            "vote_id": r[0],
            "run_id": r[1],
            "entity_id": r[2],
            "field": r[3],
            "model": r[4],
            "vote": json.loads(r[5]),
            "confidence": float(r[6]),
            "rationale": r[7],
            "created_at": r[8],
        }
        for r in rows
    ]


def render_evidence_csv(rows: list[dict[str, Any]]) -> str:
    import csv
    import io

    columns = [
        "entity_type",
        "entity_name",
        "field",
        "value",
        "confidence",
        "source_url",
        "quote",
        "provenance_id",
        "conflict_status",
        "last_seen_run",
    ]
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        rendered = dict(row)
        if isinstance(rendered.get("value"), (dict, list)):
            rendered["value"] = json.dumps(rendered["value"], default=str)
        writer.writerow({col: rendered.get(col, "") for col in columns})
    return out.getvalue()


def render_evidence_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Entity | Field | Value | Conf. | Source |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        value = row["value"]
        if isinstance(value, (dict, list)):
            value = json.dumps(value, default=str)
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row["entity_name"]),
                    _md(row["field"]),
                    _md(str(value)),
                    f"{row['confidence']:.2f}",
                    _md(row.get("source_url", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_report(
    run_id: str,
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    votes: list[dict[str, Any]],
    template: str = "analyst",
    fmt: str = "md",
) -> str | dict[str, Any]:
    citation_map: dict[str, int] = {}
    for row in rows:
        url = str(row.get("source_url") or "")
        if url and url not in citation_map:
            citation_map[url] = len(citation_map) + 1

    payload = {
        "run_id": run_id,
        "template": template,
        "metrics": metrics,
        "entity_count": metrics.get("entities_total", 0),
        "field_count": len(rows),
        "source_count": len(citation_map),
        "verification_votes": votes,
        "evidence": rows,
        "sources": sources,
    }
    if fmt == "json":
        return payload

    md = _report_markdown(run_id, metrics, rows, sources, votes, citation_map, template)
    if fmt == "md":
        return md
    if fmt == "html":
        return "<!doctype html><html><body>" + _markdown_to_simple_html(md) + "</body></html>\n"
    raise ValueError(f"unsupported report format: {fmt}")


def _report_markdown(
    run_id: str,
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    votes: list[dict[str, Any]],
    citation_map: dict[str, int],
    template: str,
) -> str:
    title = {
        "brief": "Research Brief",
        "systematic": "Systematic Evidence Report",
        "analyst": "Analyst Research Report",
    }.get(template, "Research Report")
    lines = [
        f"# {title}: {run_id}",
        "",
        "## Summary",
        "",
        f"- Entities: {metrics.get('entities_total', 0)}",
        f"- Field fill: {float(metrics.get('strict_fields_filled_pct') or 0.0) * 100:.1f}%",
        f"- Open conflicts: {metrics.get('conflicts_open', 0)}",
        f"- Cost: ${float(metrics.get('cost_usd') or 0.0):.4f}",
        f"- Stop reason: {metrics.get('reason') or 'unknown'}",
        "",
        "## Findings",
        "",
    ]
    by_entity: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_entity.setdefault(str(row["entity_name"]), []).append(row)
    for entity_name, entity_rows in by_entity.items():
        lines.append(f"### {entity_name}")
        for row in entity_rows:
            value = row["value"]
            if isinstance(value, (dict, list)):
                value = json.dumps(value, default=str)
            cite = ""
            url = str(row.get("source_url") or "")
            if url:
                cite = f" [{citation_map[url]}]"
            lines.append(f"- **{row['field']}**: {value}{cite}")
        lines.append("")
        if template == "brief" and len(lines) > 80:
            break
    if votes:
        lines.extend(["## Verification Council", ""])
        for vote in votes:
            lines.append(
                f"- {vote['model']} on {vote['field']}: {vote['vote']} "
                f"(confidence {vote['confidence']:.2f})"
            )
        lines.append("")
    lines.extend(["## Sources", ""])
    urls_by_id = {v: k for k, v in citation_map.items()}
    for idx in sorted(urls_by_id):
        lines.append(f"{idx}. {urls_by_id[idx]}")
    if sources:
        lines.extend(["", "## Source Pack", ""])
        for source in sources:
            lines.append(
                f"- {source['title']} ({source['source_type']}), chunks={source['chunks']}: {source['url']}"
            )
    return "\n".join(lines).rstrip() + "\n"


def _markdown_to_simple_html(md: str) -> str:
    out: list[str] = []
    for line in md.splitlines():
        escaped = html.escape(line)
        if line.startswith("# "):
            out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            out.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("- "):
            out.append(f"<li>{html.escape(line[2:])}</li>")
        elif line:
            out.append(f"<p>{escaped}</p>")
    return "\n".join(out)


def _md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")[:300]
