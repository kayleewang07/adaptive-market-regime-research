"""Experiment run metadata and artifact tracking."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path


def create_run_directory(root: Path) -> tuple[str, Path]:
    """Create and return a UTC timestamp-based artifact directory."""
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = root / run_id
    path.mkdir(parents=True, exist_ok=False)
    return run_id, path


def write_manifest(path: Path, specification: object) -> None:
    """Persist a JSON-serializable experiment specification."""
    payload = (
        asdict(specification) if hasattr(specification, "__dataclass_fields__") else specification
    )
    (path / "manifest.json").write_text(json.dumps(payload, indent=2, default=str))
