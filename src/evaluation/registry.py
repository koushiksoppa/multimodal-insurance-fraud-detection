"""Experiment registry.

Every training run writes one record here. The point is reproducibility: a
number in the results table should be traceable to the exact code, data version
and configuration that produced it, months later, without relying on memory.

Deliberately file-based. A full experiment-tracking platform would add a
service to run, a schema to migrate and a login to manage, none of which this
project needs -- there is one machine, one author, and runs measured in
hundreds rather than millions. JSON per run plus a flat index is enough to
answer the questions that matter: what was run, on what data, with what
settings, and what came out. The records are mirrored into SQL by
:mod:`app.database` for analytical queries.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import get_paths
from src.utils.io import read_json, write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)

INDEX_FILENAME = "index.csv"


def _git_commit() -> str | None:
    """Short hash of the current commit, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=get_paths().root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _git_dirty() -> bool | None:
    """Whether the working tree has uncommitted changes.

    Recorded because a result produced from a dirty tree cannot be reproduced
    from the commit hash alone, and that is worth knowing later.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=get_paths().root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


@dataclass
class ExperimentRecord:
    """One training-and-evaluation run."""

    experiment_id: str
    model: str
    """Model family, for example 'logistic_regression' or 'fusion_gated'."""

    modalities: list[str]
    """Which modalities the model consumed: any of tabular, text, image."""

    dataset_version: str
    track: str
    split_strategy: str
    seed: int

    hyperparameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    """Keyed by evaluation set: {'val': {...}, 'test': {...}}."""

    threshold: dict[str, Any] = field(default_factory=dict)
    feature_version: str = "v1"
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    git_commit: str | None = field(default_factory=_git_commit)
    git_dirty: bool | None = field(default_factory=_git_dirty)
    python_version: str = field(default_factory=platform.python_version)
    duration_seconds: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def modality_key(self) -> str:
        """Canonical name for the modality combination, used in ablation tables."""
        order = ["tabular", "text", "image"]
        present = [m for m in order if m in self.modalities]
        return "+".join(present) if present else "none"


class ExperimentRegistry:
    """Read/write access to the run records under ``experiments/runs``."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or get_paths().artifacts
        self.root.mkdir(parents=True, exist_ok=True)

    # -- writing ---------------------------------------------------------

    def save(self, record: ExperimentRecord) -> Path:
        run_dir = self.root / record.experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = write_json(run_dir / "record.json", record.as_dict())
        self._append_to_index(record)
        logger.info("recorded experiment %s -> %s", record.experiment_id, path)
        return path

    def save_artifact(self, experiment_id: str, name: str, payload: Any) -> Path:
        """Store a side artefact (predictions, curves, attributions) with a run."""
        run_dir = self.root / experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, pd.DataFrame):
            path = run_dir / name
            if path.suffix == ".parquet":
                payload.to_parquet(path, index=False)
            else:
                payload.to_csv(path, index=False)
            return path
        return write_json(run_dir / name, payload)

    def _append_to_index(self, record: ExperimentRecord) -> None:
        """Maintain a flat index so runs can be compared without opening each file."""
        row = {
            "experiment_id": record.experiment_id,
            "timestamp": record.timestamp,
            "model": record.model,
            "modalities": record.modality_key,
            "track": record.track,
            "dataset_version": record.dataset_version,
            "split_strategy": record.split_strategy,
            "seed": record.seed,
            "git_commit": record.git_commit,
            "git_dirty": record.git_dirty,
            "duration_seconds": record.duration_seconds,
        }
        for split in ("val", "test"):
            for metric in ("pr_auc", "roc_auc", "precision", "recall", "f1", "brier"):
                row[f"{split}_{metric}"] = record.metrics.get(split, {}).get(metric)

        index_path = self.root / INDEX_FILENAME
        frame = pd.DataFrame([row])
        if index_path.exists():
            existing = pd.read_csv(index_path)
            # A re-run under the same id replaces its row rather than duplicating it.
            existing = existing[existing["experiment_id"] != record.experiment_id]
            frame = pd.concat([existing, frame], ignore_index=True)
        frame.to_csv(index_path, index=False)

    # -- reading ---------------------------------------------------------

    def load(self, experiment_id: str) -> ExperimentRecord:
        payload = read_json(self.root / experiment_id / "record.json")
        return ExperimentRecord(**payload)

    def index(self) -> pd.DataFrame:
        index_path = self.root / INDEX_FILENAME
        if not index_path.exists():
            return pd.DataFrame()
        return pd.read_csv(index_path)

    def all_records(self) -> list[ExperimentRecord]:
        records = []
        for run_dir in sorted(self.root.iterdir()):
            record_path = run_dir / "record.json"
            if run_dir.is_dir() and record_path.exists():
                records.append(ExperimentRecord(**read_json(record_path)))
        return records

    def find(
        self,
        *,
        track: str | None = None,
        model: str | None = None,
        modalities: str | None = None,
    ) -> list[ExperimentRecord]:
        out = self.all_records()
        if track:
            out = [r for r in out if r.track == track]
        if model:
            out = [r for r in out if r.model == model]
        if modalities:
            out = [r for r in out if r.modality_key == modalities]
        return out


def make_experiment_id(
    *, model: str, modalities: list[str], track: str, seed: int, split_strategy: str
) -> str:
    """Deterministic, readable run identifier.

    Deterministic rather than random so that re-running the same configuration
    overwrites its own record instead of accumulating near-duplicates.
    """
    order = ["tabular", "text", "image"]
    short = {"tabular": "tab", "text": "txt", "image": "img"}
    parts = [short[m] for m in order if m in modalities] or ["none"]
    return f"{track}__{model}__{'-'.join(parts)}__{split_strategy}__s{seed}"
