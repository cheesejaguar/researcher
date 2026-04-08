"""Static configuration loaders for the researcher package.

Currently exposes :class:`McpServerPin` and :func:`load_mcp_pins` for
version-pinning MCP servers that researcher integrates with. Pins are the
primary defense against MCP tool-description poisoning (Invariant Labs,
April 2025): by freezing the allowlisted tool names per server/version,
we can drop anything the server advertises that we don't expect without
ever letting the description text touch a log, exception, or LLM context.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_DEFAULT_PINS_PATH = Path(__file__).parent / "mcp_pins.yaml"


@dataclass(frozen=True)
class McpServerPin:
    """Immutable pin record for a single MCP server."""

    name: str
    version: str
    allowed_tools: tuple[str, ...]


def load_mcp_pins(path: Path | str | None = None) -> dict[str, McpServerPin]:
    """Load MCP server pins from a YAML file.

    Returns an empty dict on missing file, YAML parse errors, or a root
    document that isn't a mapping. The shipped default lives next to this
    module as ``mcp_pins.yaml`` and defines the single ``exa`` server.
    """
    p = Path(path) if path is not None else _DEFAULT_PINS_PATH
    try:
        raw = p.read_text()
    except (FileNotFoundError, OSError):
        return {}
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        return {}
    if not isinstance(data, dict):
        return {}
    servers = data.get("servers")
    if not isinstance(servers, dict):
        return {}
    pins: dict[str, McpServerPin] = {}
    for key, entry in servers.items():
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        version = entry.get("version")
        allowed = entry.get("allowed_tools", [])
        if not isinstance(name, str) or not isinstance(version, str):
            continue
        if not isinstance(allowed, list):
            continue
        tools = tuple(str(t) for t in allowed)
        pins[str(key)] = McpServerPin(name=name, version=version, allowed_tools=tools)
    return pins


__all__ = ["McpServerPin", "load_mcp_pins"]
