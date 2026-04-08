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
    CoverageSnapshot,
    Entity,
    FieldCell,
    KnowledgeStore,
    StoreMetrics,
    _classify_source,
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
        self._vss_enabled: bool = False
        self._duckpgq_enabled: bool = False
        # Optional CostTracker hook. When set (via set_cost_tracker), the
        # store's snapshot_metrics reads the real USD total from it instead
        # of reporting 0.0 unconditionally.
        self._cost_tracker: Any = None

    def set_cost_tracker(self, tracker: Any) -> None:
        """Install an optional CostTracker so snapshot_metrics can report
        real USD spend. The tracker must expose ``total_usd() -> float``.
        """
        self._cost_tracker = tracker

    # ---- lifecycle -------------------------------------------------------

    async def open(self) -> None:
        if self._opened:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self._db_path))
        # Install + load the VSS extension for HNSW-based k-NN search.
        # The HNSW index in vss requires an experimental flag for persistent
        # databases (in-memory works without it). Both INSTALL and LOAD are
        # idempotent so reopening an existing file is cheap.
        try:
            self._conn.execute("INSTALL vss")
            self._conn.execute("LOAD vss")
            self._conn.execute("SET hnsw_enable_experimental_persistence = true")
            self._vss_enabled = True
        except duckdb.Error:
            self._vss_enabled = False
        # Attempt to load the DuckPGQ community extension for SQL/PGQ graph
        # queries. Not available on every platform; fall back gracefully.
        try:
            self._conn.execute("INSTALL duckpgq FROM community")
            self._conn.execute("LOAD duckpgq")
            self._duckpgq_enabled = True
        except duckdb.Error:
            self._duckpgq_enabled = False
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
                first_seen_run TEXT,
                last_seen_run TEXT,
                superseded_by_run TEXT,
                PRIMARY KEY (entity_id, field_name)
            )
            """
        )
        # Migrate existing fields tables to add the temporal columns when
        # opening a DuckDB file created before v1.1 cross-run merge mode.
        try:
            c.execute("ALTER TABLE fields ADD COLUMN first_seen_run TEXT")
        except duckdb.Error:
            pass
        try:
            c.execute("ALTER TABLE fields ADD COLUMN last_seen_run TEXT")
        except duckdb.Error:
            pass
        try:
            c.execute("ALTER TABLE fields ADD COLUMN superseded_by_run TEXT")
        except duckdb.Error:
            pass
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                entity_id TEXT PRIMARY KEY,
                vector_json TEXT NOT NULL,
                vector_array FLOAT[384]
            )
            """
        )
        if self._vss_enabled:
            try:
                c.execute(
                    "CREATE INDEX IF NOT EXISTS embeddings_hnsw "
                    "ON embeddings USING HNSW (vector_array) "
                    "WITH (metric = 'cosine')"
                )
            except duckdb.Error:
                # HNSW index creation can fail on existing tables with
                # duplicate keys; safe to swallow and fall back to linear scan.
                pass
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
        # v1.2 schema-migration audit trail. Every init_schema call that
        # detects a schema change records a row here so operators can see
        # exactly when fields were added/removed. DuckDB has no
        # AUTO_INCREMENT; we use an explicit sequence for the id column.
        c.execute(
            """
            CREATE SEQUENCE IF NOT EXISTS schema_migrations_id_seq START 1
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id INTEGER PRIMARY KEY,
                entity_class_name TEXT NOT NULL,
                schema_hash TEXT NOT NULL,
                schema_json TEXT NOT NULL,
                migration_type TEXT NOT NULL,
                diff_added_json TEXT,
                diff_removed_json TEXT,
                applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS run_checkpoints (
                run_id TEXT PRIMARY KEY,
                cycle INTEGER NOT NULL,
                data_json TEXT NOT NULL,
                saved_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Typed graph edges between entities. The (source_id, target_id,
        # relation_label) composite key means multiple distinct labels between
        # the same two entities coexist, but duplicate triples are deduped on
        # record_relation() with a higher-confidence-wins policy.
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS entity_relations (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_label TEXT NOT NULL,
                confidence DOUBLE NOT NULL,
                run_id TEXT NOT NULL,
                valid_from TIMESTAMP,
                valid_to TIMESTAMP,
                PRIMARY KEY (source_id, target_id, relation_label)
            )
            """
        )
        # Declare (or refresh) the DuckPGQ property graph view so ``MATCH``
        # clauses can traverse (entities)-[entity_relations]->(entities).
        # Only attempted when the extension loaded; swallowed on failure so
        # reopening an existing file stays idempotent.
        if self._duckpgq_enabled:
            try:
                c.execute(
                    """
                    CREATE OR REPLACE PROPERTY GRAPH researcher_kg
                    VERTEX TABLES (entities)
                    EDGE TABLES (
                        entity_relations
                        SOURCE KEY (source_id) REFERENCES entities (id)
                        DESTINATION KEY (target_id) REFERENCES entities (id)
                    )
                    """
                )
            except duckdb.Error:
                pass

    def _require_conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            raise RuntimeError("DuckDBKnowledgeStore is not open()")
        return self._conn

    # ---- schema registration --------------------------------------------

    async def init_schema(
        self, entity_class: type[BaseModel], force: bool = False
    ) -> None:
        """Register an entity class's schema with migration audit trail.

        v1.2 behavior:

        - First call (no prior ``schema_migrations`` rows) records an
          ``initial`` migration.
        - Subsequent call with identical schema hash: idempotent no-op.
        - Subsequent call with only added fields: records an
          ``additive`` migration.
        - Subsequent call that removes (or renames) fields: raises
          :class:`SchemaMigrationError` unless ``force=True``, in which
          case it records a ``breaking_forced`` migration.

        The legacy ``schema_meta`` upsert still runs so existing queries
        against that table continue to return the most-recent schema.
        """
        from researcher.storage.migrations import (
            SchemaMigrationError,
            compute_schema_hash,
        )

        async with self._lock:
            conn = self._require_conn()
            new_hash = compute_schema_hash(entity_class)
            new_schema_dict = entity_class.model_json_schema()
            new_schema_json = json.dumps(new_schema_dict)
            new_class_name = entity_class.__name__

            # Look up the most recent migration row across all entity
            # classes. Wave 1 runs with a single entity type per store so
            # "latest row" is the right comparison baseline.
            latest_row = conn.execute(
                "SELECT entity_class_name, schema_hash, schema_json "
                "FROM schema_migrations ORDER BY applied_at DESC, id DESC LIMIT 1"
            ).fetchone()

            if latest_row is None:
                # First-time init — record an 'initial' migration.
                conn.execute(
                    "INSERT INTO schema_migrations "
                    "(id, entity_class_name, schema_hash, schema_json, "
                    " migration_type) "
                    "VALUES (nextval('schema_migrations_id_seq'), ?, ?, ?, "
                    "'initial')",
                    [new_class_name, new_hash, new_schema_json],
                )
                self._record_schema_meta(conn, new_class_name, new_schema_json)
                return

            _prev_class_name, prev_hash, prev_schema_json = latest_row

            if prev_hash == new_hash:
                # Idempotent: same schema, no migration recorded.
                return

            # Compute a field-name diff directly from the stored JSON
            # schema so we don't need to reconstruct a Pydantic class
            # for the prior version.
            prev_schema = json.loads(prev_schema_json)
            prev_fields = set(prev_schema.get("properties", {}).keys())
            new_fields = set(new_schema_dict.get("properties", {}).keys())
            added = new_fields - prev_fields
            removed = prev_fields - new_fields

            if removed and not force:
                raise SchemaMigrationError(
                    f"Breaking schema change for {new_class_name}: "
                    f"removed fields {sorted(removed)}. "
                    f"Pass force=True to override and record as "
                    f"'breaking_forced'."
                )

            migration_type = "breaking_forced" if removed else "additive"
            conn.execute(
                "INSERT INTO schema_migrations "
                "(id, entity_class_name, schema_hash, schema_json, "
                " migration_type, diff_added_json, diff_removed_json) "
                "VALUES (nextval('schema_migrations_id_seq'), ?, ?, ?, ?, "
                "?, ?)",
                [
                    new_class_name,
                    new_hash,
                    new_schema_json,
                    migration_type,
                    json.dumps(sorted(added)),
                    json.dumps(sorted(removed)),
                ],
            )
            self._record_schema_meta(conn, new_class_name, new_schema_json)

    def _record_schema_meta(
        self,
        conn: duckdb.DuckDBPyConnection,
        entity_class_name: str,
        schema_json: str,
    ) -> None:
        """Upsert the latest JSON schema into the legacy schema_meta table."""
        conn.execute(
            "DELETE FROM schema_meta WHERE entity_class_name = ?",
            [entity_class_name],
        )
        conn.execute(
            "INSERT INTO schema_meta VALUES (?, ?)",
            [entity_class_name, schema_json],
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
        """Insert-or-replace a single (entity_id, field_name) row.

        Always writes the legacy six-column tuple; the temporal columns
        (first_seen_run / last_seen_run / superseded_by_run) are populated
        only by :meth:`merge_field` and remain NULL on the overwrite path.
        """
        conn = self._require_conn()
        conn.execute(
            "DELETE FROM fields WHERE entity_id = ? AND field_name = ?",
            [entity_id, field_name],
        )
        conn.execute(
            """
            INSERT INTO fields (
                entity_id, field_name, value_json, confidence,
                provenance_ids_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
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

    # ---- cross-run merge -------------------------------------------------

    async def merge_field(
        self,
        entity_id: str,
        field_name: str,
        value: Any,
        confidence: float,
        run_id: str,
        provenance_ids: list[str],
    ) -> dict:
        """Merge a single field value into an entity with cross-run semantics.

        Returns ``{"status": ...}`` where ``status`` is one of:
          - ``"inserted"``                — no prior value, written fresh
          - ``"confirmed"``               — same value as existing, last_seen_run bumped
          - ``"superseded"``              — new value beat existing on confidence
          - ``"conflict_kept_existing"``  — new value lost; existing wins
        """
        async with self._lock:
            conn = self._require_conn()
            existing = conn.execute(
                "SELECT value_json, confidence, first_seen_run "
                "FROM fields WHERE entity_id = ? AND field_name = ?",
                [entity_id, field_name],
            ).fetchone()
            now = _now()
            value_json = json.dumps(value)
            prov_json = json.dumps(list(provenance_ids))
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO fields (
                        entity_id, field_name, value_json, confidence,
                        provenance_ids_json, updated_at,
                        first_seen_run, last_seen_run, superseded_by_run
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    [
                        entity_id,
                        field_name,
                        value_json,
                        float(confidence),
                        prov_json,
                        now,
                        run_id,
                        run_id,
                    ],
                )
                conn.execute(
                    "UPDATE entities SET updated_at = ? WHERE id = ?",
                    [now, entity_id],
                )
                return {"status": "inserted"}

            existing_value_json, existing_conf, _first_seen = existing
            existing_value = json.loads(existing_value_json)
            if existing_value == value:
                conn.execute(
                    "UPDATE fields SET last_seen_run = ?, updated_at = ? "
                    "WHERE entity_id = ? AND field_name = ?",
                    [run_id, now, entity_id, field_name],
                )
                return {"status": "confirmed"}

            if float(confidence) > float(existing_conf):
                conn.execute(
                    """
                    UPDATE fields
                       SET value_json = ?,
                           confidence = ?,
                           provenance_ids_json = ?,
                           updated_at = ?,
                           last_seen_run = ?,
                           superseded_by_run = ?
                     WHERE entity_id = ? AND field_name = ?
                    """,
                    [
                        value_json,
                        float(confidence),
                        prov_json,
                        now,
                        run_id,
                        run_id,
                        entity_id,
                        field_name,
                    ],
                )
                conn.execute(
                    "UPDATE entities SET updated_at = ? WHERE id = ?",
                    [now, entity_id],
                )
                return {"status": "superseded"}

            return {"status": "conflict_kept_existing"}

    # ---- graph edges -----------------------------------------------------

    async def record_relation(
        self,
        source_id: str,
        target_id: str,
        relation_label: str,
        confidence: float,
        run_id: str,
    ) -> None:
        """Insert or upsert an entity_relations row.

        On conflict (same source/target/label), keep the higher-confidence
        value. The run_id + valid_from are overwritten along with the
        confidence so downstream consumers can see which run promoted the
        edge last.
        """
        async with self._lock:
            conn = self._require_conn()
            existing = conn.execute(
                "SELECT confidence FROM entity_relations "
                "WHERE source_id = ? AND target_id = ? AND relation_label = ?",
                [source_id, target_id, relation_label],
            ).fetchone()
            now = _now()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO entity_relations (
                        source_id, target_id, relation_label, confidence,
                        run_id, valid_from, valid_to
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL)
                    """,
                    [
                        source_id,
                        target_id,
                        relation_label,
                        float(confidence),
                        run_id,
                        now,
                    ],
                )
            elif float(confidence) > float(existing[0]):
                conn.execute(
                    """
                    UPDATE entity_relations
                       SET confidence = ?, run_id = ?, valid_from = ?
                     WHERE source_id = ? AND target_id = ? AND relation_label = ?
                    """,
                    [
                        float(confidence),
                        run_id,
                        now,
                        source_id,
                        target_id,
                        relation_label,
                    ],
                )

    async def find_neighbors(
        self, entity_id: str, relation_label: str | None = None
    ) -> list[dict]:
        """Return outbound neighbors of an entity.

        Optionally filtered by ``relation_label``. Returns a list of dicts
        with ``target_id``, ``relation_label``, and ``confidence`` keys.
        """
        async with self._lock:
            conn = self._require_conn()
            if relation_label:
                rows = conn.execute(
                    "SELECT target_id, relation_label, confidence "
                    "FROM entity_relations "
                    "WHERE source_id = ? AND relation_label = ?",
                    [entity_id, relation_label],
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT target_id, relation_label, confidence "
                    "FROM entity_relations WHERE source_id = ?",
                    [entity_id],
                ).fetchall()
            return [
                {
                    "target_id": r[0],
                    "relation_label": r[1],
                    "confidence": float(r[2]),
                }
                for r in rows
            ]

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

        ``vec`` is padded with zeros (or truncated) to exactly 384 dims so
        the parallel ``vector_array FLOAT[384]`` column stays consistent
        with the HNSW index. The hash-based test embedder and the real
        sentence-transformers MiniLM both produce 384-dim outputs already;
        the pad path exists only for shorter test stubs.
        """
        conn = self._require_conn()
        padded = list(vec)
        if len(padded) < 384:
            padded = padded + [0.0] * (384 - len(padded))
        elif len(padded) > 384:
            padded = padded[:384]
        json_blob = json.dumps(padded)
        # DELETE + INSERT for upsert (DuckDB has no INSERT OR REPLACE).
        conn.execute("DELETE FROM embeddings WHERE entity_id = ?", [entity_id])
        conn.execute(
            "INSERT INTO embeddings (entity_id, vector_json, vector_array) "
            "VALUES (?, ?, ?)",
            [entity_id, json_blob, padded],
        )

    async def find_similar(
        self, entity_type: str, embedding: list[float], k: int = 5
    ) -> list[tuple[str, float]]:
        async with self._lock:
            conn = self._require_conn()
            # Pad/truncate the query vector to 384 so it's compatible with the
            # FLOAT[384] index column. Real embedders (LocalEmbedder MiniLM,
            # OpenAI text-embedding-3-small at 384-dim) already match.
            query_vec = list(embedding)
            if len(query_vec) < 384:
                query_vec = query_vec + [0.0] * (384 - len(query_vec))
            elif len(query_vec) > 384:
                query_vec = query_vec[:384]

            if self._vss_enabled:
                # VSS array_cosine_distance returns 1 - cosine_similarity, so
                # ASCENDING order is closest-first. We convert back to a
                # similarity score (higher = better) before returning so the
                # public contract matches the python-side fallback.
                rows = conn.execute(
                    """
                    SELECT ent.id AS entity_id,
                           array_cosine_distance(
                               emb.vector_array,
                               ?::FLOAT[384]
                           ) AS distance
                    FROM entities ent
                    JOIN embeddings emb ON emb.entity_id = ent.id
                    WHERE ent.entity_type = ?
                      AND emb.vector_array IS NOT NULL
                    ORDER BY distance ASC
                    LIMIT ?
                    """,
                    [query_vec, entity_type, k],
                ).fetchall()
                return [(r[0], 1.0 - float(r[1])) for r in rows]

            # Fallback: python-side cosine over the JSON-encoded vectors.
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
                scored.append((eid, _cosine(query_vec, vec)))
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

    async def snapshot_coverage(
        self, confidence_threshold: float = 0.5
    ) -> CoverageSnapshot:
        """Coverage signals only the store can answer (v1.2 #8).

        Returns the per-field count of cells whose confidence is below
        ``confidence_threshold`` (skipping null-valued cells, which are
        unknown rather than low-confidence) and the per-source-type
        count of provenance rows. URLs are bucketed via
        :func:`_classify_source`. Provenance rows are deduped by
        ``(entity_id, field, url)`` when those keys exist in the
        recorded JSON, falling back to ``(provenance_id, url)``
        otherwise so production rows (which only carry a ``url``) are
        not over-counted.
        """
        async with self._lock:
            conn = self._require_conn()
            field_rows = conn.execute(
                "SELECT field_name, COUNT(*) FROM fields "
                "WHERE value_json != 'null' AND confidence < ? "
                "GROUP BY field_name",
                [float(confidence_threshold)],
            ).fetchall()
            fields_below: dict[str, int] = {}
            for fname, cnt in field_rows:
                fields_below[fname] = fields_below.get(fname, 0) + int(cnt)

            source_breakdown: dict[str, int] = {}
            try:
                prov_rows = conn.execute(
                    "SELECT provenance_id, data_json FROM provenance"
                ).fetchall()
            except duckdb.Error:
                prov_rows = []

            seen: set[tuple[str, str, str]] = set()
            for prov_id, data_json in prov_rows:
                try:
                    data = json.loads(data_json) if data_json else {}
                except (TypeError, ValueError):
                    data = {}
                url = str(data.get("url", "") or "")
                ent_id = data.get("entity_id")
                field = data.get("field")
                if ent_id is not None and field is not None:
                    key = (str(ent_id), str(field), url)
                else:
                    key = (str(prov_id), "", url)
                if key in seen:
                    continue
                seen.add(key)
                bucket = _classify_source(url)
                source_breakdown[bucket] = source_breakdown.get(bucket, 0) + 1

            return CoverageSnapshot(
                fields_below_confidence=fields_below,
                source_type_breakdown=source_breakdown,
            )

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

            cost_usd_total = 0.0
            if self._cost_tracker is not None:
                try:
                    cost_usd_total = float(self._cost_tracker.total_usd())
                except Exception:
                    cost_usd_total = 0.0

            return StoreMetrics(
                entities_total=int(entities_total),
                by_type=by_type,
                fields_filled_pct=float(fields_filled_pct),
                conflicts_open=int(conflicts_open),
                cost_usd_total=cost_usd_total,
            )

    async def write_run_summary(self, run_id: str, summary: dict) -> None:
        async with self._lock:
            conn = self._require_conn()
            conn.execute(
                "INSERT INTO run_summary VALUES (?, ?, ?)",
                [run_id, _now(), json.dumps(summary)],
            )

    # ---- run checkpoints (crash-safe resume) ----------------------------

    async def save_checkpoint(
        self, run_id: str, cycle: int, data: dict
    ) -> None:
        """Persist (or overwrite) a per-run checkpoint blob.

        The checkpoint stores the orchestrator's small control state (the
        scheduler queue + entity history, the budget counters, the current
        cycle index) so a crashed run can resume from where it stopped
        instead of starting over from cycle zero.
        """
        async with self._lock:
            conn = self._require_conn()
            blob = json.dumps(data)
            # Upsert: DELETE then INSERT (DuckDB has no INSERT OR REPLACE).
            conn.execute(
                "DELETE FROM run_checkpoints WHERE run_id = ?", [run_id]
            )
            conn.execute(
                "INSERT INTO run_checkpoints (run_id, cycle, data_json) "
                "VALUES (?, ?, ?)",
                [run_id, int(cycle), blob],
            )

    async def load_checkpoint(self, run_id: str) -> Optional[dict]:
        """Return the most recent checkpoint for a run, or None if absent."""
        async with self._lock:
            conn = self._require_conn()
            row = conn.execute(
                "SELECT cycle, data_json FROM run_checkpoints WHERE run_id = ?",
                [run_id],
            ).fetchone()
            if row is None:
                return None
            return {"cycle": int(row[0]), "data": json.loads(row[1])}
