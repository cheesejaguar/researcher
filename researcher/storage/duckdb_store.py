"""DuckDB-backed KnowledgeStore — the real persistent store for Wave 1-B.

Single-file, columnar, fast analytical queries, no daemon. This is the first
concrete `KnowledgeStore` implementation the orchestrator wires up for real
runs (the in-memory `StubKnowledgeStore` remains for unit tests of higher
layers).

Schema overview (all `CREATE TABLE IF NOT EXISTS`, idempotent):

- ``entities``      one row per (type, name) — UUID4 hex primary key.
- ``fields``        one row per (entity_id, field_name); JSON-encoded values.
- ``embeddings``    one row per entity_id with a JSON-encoded float vector.
- ``provenance``    raw provenance blobs keyed by provenance_id.
- ``conflicts``     unresolved/resolved conflicts surfaced by the resolver.
- ``run_summary``   one row per run summary write (multiple per run_id ok).
- ``schema_meta``   JSON schemas registered via :meth:`init_schema`.

Notes:

- DuckDB is single-writer; we serialize all mutations behind an
  ``asyncio.Lock`` so concurrent agents in the orchestrator don't race.
- ``find_similar`` uses python-side cosine similarity over JSON-decoded
  vectors. The DuckDB ``vss`` extension is a deliberate later optimization
  — see Wave 1-B follow-ups.
- ``set_vector`` is a public hook (not part of the ABC) used by both tests
  and the future `EntityResolver` to persist embeddings as they're computed.
"""

from __future__ import annotations

import asyncio
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import duckdb
from pydantic import BaseModel

from researcher.storage.store import (
    Conflict,
    Entity,
    FieldCell,
    KnowledgeStore,
    StoreMetrics,
)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity for two equal-length float vectors. 0.0 on degeneracy."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _now() -> datetime:
    return datetime.now(UTC)


class DuckDBKnowledgeStore(KnowledgeStore):
    """KnowledgeStore implementation backed by a single DuckDB file.

    Construct with a path; call :meth:`open` (or use as an async context
    manager) to materialize the schema. All ABC methods are async; sync
    DuckDB calls run in-line under the write lock since DuckDB is fast and
    single-writer anyway.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._lock = asyncio.Lock()
        self._opened = False

    # ---- lifecycle -------------------------------------------------------

    async def open(self) -> None:
        if self._opened:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self._db_path))
        self._init_tables()
        self._opened = True

    async def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._opened = False

    def _init_tables(self) -> None:
        assert self._conn is not None
        c = self._conn
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS fields (
                entity_id TEXT NOT NULL,
                field_name TEXT NOT NULL,
                value_json TEXT NOT NULL,
                confidence DOUBLE NOT NULL,
                provenance_ids_json TEXT NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                PRIMARY KEY (entity_id, field_name)
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                entity_id TEXT PRIMARY KEY,
                vector_json TEXT NOT NULL
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS provenance (
                provenance_id TEXT PRIMARY KEY,
                data_json TEXT NOT NULL
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS conflicts (
                conflict_id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                field TEXT NOT NULL,
                status TEXT NOT NULL,
                candidates_json TEXT NOT NULL,
                winning_value_json TEXT,
                reason TEXT
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS run_summary (
                run_id TEXT NOT NULL,
                started_at TIMESTAMP NOT NULL,
                summary_json TEXT NOT NULL
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                entity_class_name TEXT PRIMARY KEY,
                json_schema TEXT NOT NULL
            )
            """
        )

    def _require_conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            raise RuntimeError("DuckDBKnowledgeStore is not open()")
        return self._conn

    # ---- schema registration --------------------------------------------

    async def init_schema(self, entity_class: type[BaseModel]) -> None:
        async with self._lock:
            conn = self._require_conn()
            schema_json = json.dumps(entity_class.model_json_schema())
            # DuckDB doesn't support INSERT OR REPLACE; emulate via DELETE+INSERT.
            conn.execute(
                "DELETE FROM schema_meta WHERE entity_class_name = ?",
                [entity_class.__name__],
            )
            conn.execute(
                "INSERT INTO schema_meta VALUES (?, ?)",
                [entity_class.__name__, schema_json],
            )

    # ---- entities --------------------------------------------------------

    async def upsert_entity(
        self, entity_type: str, name: str, fields: dict[str, FieldCell]
    ) -> str:
        async with self._lock:
            conn = self._require_conn()
            now = _now()

            row = conn.execute(
                "SELECT id FROM entities WHERE entity_type = ? AND name = ?",
                [entity_type, name],
            ).fetchone()

            if row is None:
                eid = uuid4().hex
                conn.execute(
                    "INSERT INTO entities VALUES (?, ?, ?, ?, ?)",
                    [eid, entity_type, name, now, now],
                )
                # Always persist the canonical 'name' cell so get_entity
                # round-trips a complete record (matches StubKnowledgeStore).
                self._write_field(
                    eid,
                    "name",
                    FieldCell(
                        value=name,
                        confidence=1.0,
                        provenance_ids=[],
                        updated_at=now,
                    ),
                )
            else:
                eid = row[0]
                conn.execute(
                    "UPDATE entities SET updated_at = ? WHERE id = ?",
                    [now, eid],
                )

            for field_name, cell in fields.items():
                self._write_field(eid, field_name, cell)

            return eid

    def _write_field(self, entity_id: str, field_name: str, cell: FieldCell) -> None:
        """Insert-or-replace a single (entity_id, field_name) row."""
        conn = self._require_conn()
        conn.execute(
            "DELETE FROM fields WHERE entity_id = ? AND field_name = ?",
            [entity_id, field_name],
        )
        conn.execute(
            "INSERT INTO fields VALUES (?, ?, ?, ?, ?, ?)",
            [
                entity_id,
                field_name,
                json.dumps(cell.value),
                float(cell.confidence),
                json.dumps(list(cell.provenance_ids)),
                cell.updated_at,
            ],
        )

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        async with self._lock:
            conn = self._require_conn()
            ent_row = conn.execute(
                "SELECT id, entity_type, name FROM entities WHERE id = ?",
                [entity_id],
            ).fetchone()
            if ent_row is None:
                return None

            field_rows = conn.execute(
                """
                SELECT field_name, value_json, confidence, provenance_ids_json, updated_at
                FROM fields
                WHERE entity_id = ?
                """,
                [entity_id],
            ).fetchall()

            fields: dict[str, FieldCell] = {}
            for fname, value_json, conf, prov_json, updated_at in field_rows:
                fields[fname] = FieldCell(
                    value=json.loads(value_json),
                    confidence=float(conf),
                    provenance_ids=json.loads(prov_json),
                    updated_at=updated_at,
                )

            return Entity(id=ent_row[0], type=ent_row[1], fields=fields)

    # ---- generic query ---------------------------------------------------

    async def query(self, sql: str, params: tuple = ()) -> list[dict]:
        async with self._lock:
            conn = self._require_conn()
            cur = conn.execute(sql, list(params))
            rows = cur.fetchall()
            cols = [d[0] for d in (cur.description or [])]
            return [dict(zip(cols, row, strict=False)) for row in rows]

    # ---- embeddings + similarity ----------------------------------------

    def set_vector(self, entity_id: str, vec: list[float]) -> None:
        """Persist an embedding for an entity.

        Public hook (not part of the ABC) used by tests and by the future
        :class:`EntityResolver` to write embeddings as it computes them.
        Sync because the resolver typically calls it from inside its own
        async pipeline already holding the event loop's attention; if you
        need a coroutine, ``asyncio.to_thread(store.set_vector, ...)`` works.
        """
        conn = self._require_conn()
        conn.execute("DELETE FROM embeddings WHERE entity_id = ?", [entity_id])
        conn.execute(
            "INSERT INTO embeddings VALUES (?, ?)",
            [entity_id, json.dumps(list(vec))],
        )

    async def find_similar(
        self, entity_type: str, embedding: list[float], k: int = 5
    ) -> list[tuple[str, float]]:
        async with self._lock:
            conn = self._require_conn()
            rows = conn.execute(
                """
                SELECT e.entity_id, e.vector_json
                FROM embeddings e
                JOIN entities ent ON ent.id = e.entity_id
                WHERE ent.entity_type = ?
                """,
                [entity_type],
            ).fetchall()

            scored: list[tuple[str, float]] = []
            for eid, vec_json in rows:
                vec = json.loads(vec_json)
                scored.append((eid, _cosine(embedding, vec)))
            scored.sort(key=lambda p: p[1], reverse=True)
            return scored[:k]

    # ---- provenance ------------------------------------------------------

    async def record_provenance(self, provenance_id: str, data: dict) -> None:
        async with self._lock:
            conn = self._require_conn()
            conn.execute(
                "DELETE FROM provenance WHERE provenance_id = ?", [provenance_id]
            )
            conn.execute(
                "INSERT INTO provenance VALUES (?, ?)",
                [provenance_id, json.dumps(data)],
            )

    # ---- conflicts -------------------------------------------------------

    async def get_conflicts(self, status: str = "open") -> list[Conflict]:
        async with self._lock:
            conn = self._require_conn()
            rows = conn.execute(
                """
                SELECT conflict_id, entity_id, field, status,
                       candidates_json, winning_value_json, reason
                FROM conflicts
                WHERE status = ?
                """,
                [status],
            ).fetchall()

            conflicts: list[Conflict] = []
            for cid, ent_id, field, st, cand_json, win_json, reason in rows:
                candidates = [
                    FieldCell.model_validate(c) for c in json.loads(cand_json)
                ]
                winning_value: Any = (
                    json.loads(win_json) if win_json is not None else None
                )
                conflicts.append(
                    Conflict(
                        conflict_id=cid,
                        entity_id=ent_id,
                        field=field,
                        candidates=candidates,
                        status=st,
                        winning_value=winning_value,
                        reason=reason,
                    )
                )
            return conflicts

    async def record_conflict(self, conflict: Conflict) -> None:
        async with self._lock:
            conn = self._require_conn()
            cand_json = json.dumps(
                [json.loads(c.model_dump_json()) for c in conflict.candidates]
            )
            win_json = (
                json.dumps(conflict.winning_value)
                if conflict.winning_value is not None
                else None
            )
            conn.execute(
                "DELETE FROM conflicts WHERE conflict_id = ?", [conflict.conflict_id]
            )
            conn.execute(
                "INSERT INTO conflicts VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    conflict.conflict_id,
                    conflict.entity_id,
                    conflict.field,
                    conflict.status,
                    cand_json,
                    win_json,
                    conflict.reason,
                ],
            )

    # ---- metrics + run summaries ----------------------------------------

    async def snapshot_metrics(self) -> StoreMetrics:
        async with self._lock:
            conn = self._require_conn()

            entities_total = conn.execute(
                "SELECT COUNT(*) FROM entities"
            ).fetchone()[0]

            by_type_rows = conn.execute(
                "SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type"
            ).fetchall()
            by_type = {etype: int(cnt) for etype, cnt in by_type_rows}

            # Field-fill: cells whose JSON-decoded value is non-null divided
            # by total cells across all entities. Mirrors StubKnowledgeStore.
            total_cells = conn.execute(
                "SELECT COUNT(*) FROM fields"
            ).fetchone()[0]
            if total_cells:
                # value_json == 'null' means the python value was None.
                filled = conn.execute(
                    "SELECT COUNT(*) FROM fields WHERE value_json != 'null'"
                ).fetchone()[0]
                fields_filled_pct = float(filled) / float(total_cells)
            else:
                fields_filled_pct = 0.0

            conflicts_open = conn.execute(
                "SELECT COUNT(*) FROM conflicts WHERE status = 'open'"
            ).fetchone()[0]

            return StoreMetrics(
                entities_total=int(entities_total),
                by_type=by_type,
                fields_filled_pct=float(fields_filled_pct),
                conflicts_open=int(conflicts_open),
                cost_usd_total=0.0,
            )

    async def write_run_summary(self, run_id: str, summary: dict) -> None:
        async with self._lock:
            conn = self._require_conn()
            conn.execute(
                "INSERT INTO run_summary VALUES (?, ?, ?)",
                [run_id, _now(), json.dumps(summary)],
            )
