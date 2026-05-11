"""Small persistent cache for prompt-level sidecar agent outputs.

The cache is intentionally exact-keyed, not semantic. It speeds up repeated
development/demo runs without letting stale "similar prompt" guesses weaken the
generation contract.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any


CACHE_SCHEMA_VERSION = 1
CACHE_DIR = Path(os.getenv("HARNESS_PROMPT_CACHE_DIR", ".harness_cache"))


def cache_enabled() -> bool:
    """Return whether prompt-sidecar caching is enabled."""

    return os.getenv("HARNESS_DISABLE_PROMPT_CACHE", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def load_cached_json(namespace: str, key: dict[str, Any]) -> dict[str, Any] | None:
    """Load a cached JSON payload for an exact namespace/key pair."""

    if not cache_enabled():
        return None
    path = _cache_path(namespace, key)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if record.get("schema_version") != CACHE_SCHEMA_VERSION:
        return None
    payload = record.get("payload")
    return payload if isinstance(payload, dict) else None


def save_cached_json(namespace: str, key: dict[str, Any], payload: dict[str, Any]) -> None:
    """Persist a JSON payload for an exact namespace/key pair."""

    if not cache_enabled():
        return
    path = _cache_path(namespace, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "namespace": namespace,
        "created_at": time.time(),
        "key_hash": path.stem,
        "payload": payload,
    }
    text = json.dumps(record, indent=2, sort_keys=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        prefix=f".{path.stem}.",
        suffix=".tmp",
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _cache_path(namespace: str, key: dict[str, Any]) -> Path:
    safe_namespace = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in namespace)
    digest = hashlib.sha256(_canonical_json(key).encode("utf-8")).hexdigest()
    return CACHE_DIR / safe_namespace / f"{digest}.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
