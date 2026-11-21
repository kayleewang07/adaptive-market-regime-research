"""Experiment run metadata and artifact tracking."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .specification import ExperimentSpec


def create_run_directory(root: Path, label: str = "") -> tuple[str, Path]:
    """Create a unique, time-sortable artifact directory and return its id and path."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = "-".join(part for part in (stamp, label, uuid.uuid4().hex[:6]) if part)
    path = root / run_id
    path.mkdir(parents=True, exist_ok=False)
    return run_id, path


def write_manifest(path: Path, specification: ExperimentSpec, **extra: object) -> None:
    """Persist the experiment specification plus run metadata as JSON."""
    payload = {"specification": asdict(specification), **extra}
    (path / "manifest.json").write_text(json.dumps(payload, indent=2, default=str))
