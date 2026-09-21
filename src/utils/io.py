"""Small filesystem helpers used across the pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_json(path: Path | str, payload: Any, *, indent: int = 2) -> Path:
    """Write JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=indent, default=_json_default, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_table(df: pd.DataFrame, path: Path | str) -> Path:
    """Write a dataframe, choosing the writer from the file suffix.

    Parquet is preferred for intermediate artefacts because it preserves dtypes
    -- important here, where silently turning a categorical into an object
    column would change downstream encoding behaviour.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"unsupported table suffix: {path.suffix!r}")
    return path


def read_table(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported table suffix: {path.suffix!r}")


def file_sha256(path: Path | str, *, chunk_size: int = 1 << 20) -> str:
    """Checksum a file in chunks, so large image archives do not need to fit in RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(obj: Any) -> Any:
    """Coerce values numpy/pandas produce that the json module cannot encode."""
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serialisable")
