"""Schema diff + migration helpers (v1.2).

Compares two Pydantic entity classes and reports added/removed fields.
A migration is "additive" if all changes are field additions (or a no-op);
"breaking" if any field was removed or renamed (we treat renames as
remove + add).

This module is the backbone of :meth:`DuckDBKnowledgeStore.init_schema`'s
v1.2 schema-evolution discipline: additive changes are quiet and
auto-applied, removals are loud and require an explicit ``force=True``
override.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from pydantic import BaseModel


class SchemaMigrationError(RuntimeError):
    """Raised when init_schema detects a breaking change without force=True."""


@dataclass
class SchemaDiff:
    """Field-level diff between two Pydantic entity classes.

    ``added`` / ``removed`` / ``common`` are sets of field names. A diff
    with no ``removed`` fields is considered ``is_additive`` (the no-op
    case also counts as additive since it can be safely re-applied).
    """

    added: set[str]
    removed: set[str]
    common: set[str]

    @property
    def is_additive(self) -> bool:
        """True when no fields were removed.

        A no-op diff (added and removed both empty) is also additive so
        the re-apply path stays quiet.
        """
        return len(self.removed) == 0

    @property
    def is_breaking(self) -> bool:
        """True when any field was removed (or renamed, treated as remove)."""
        return len(self.removed) > 0

    @property
    def is_no_op(self) -> bool:
        """True when neither added nor removed has any fields."""
        return len(self.added) == 0 and len(self.removed) == 0


def compute_schema_hash(entity_class: type[BaseModel]) -> str:
    """Stable sha256 of the Pydantic JSON schema (sorted keys).

    Deterministic across runs: two calls with the same class produce the
    same digest; a class with a different field set produces a different
    digest.
    """
    schema = entity_class.model_json_schema()
    blob = json.dumps(schema, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _field_names(entity_class: type[BaseModel]) -> set[str]:
    return set(entity_class.model_fields.keys())


def diff_schemas(
    old: type[BaseModel], new: type[BaseModel]
) -> SchemaDiff:
    """Return a :class:`SchemaDiff` comparing two Pydantic classes.

    Field names are taken from ``model_fields`` — we intentionally
    compare field *identities*, not their types, so a type change (e.g.
    ``int`` → ``str``) on an existing field is currently treated as a
    no-op at the diff level. Future work can extend this to type-aware
    diffs.
    """
    old_fields = _field_names(old)
    new_fields = _field_names(new)
    return SchemaDiff(
        added=new_fields - old_fields,
        removed=old_fields - new_fields,
        common=old_fields & new_fields,
    )
