"""Central configuration.

Every path in the project is derived from :data:`PATHS` rather than hard-coded,
so the repository can be cloned to any location on any operating system. Values
are read from the environment (optionally via a ``.env`` file); see
``.env.example`` for the full list.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: src/utils/config.py -> src/utils -> src -> <root>
ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings, populated from the environment."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        env_prefix="MIFD_",
        extra="ignore",
    )

    # --- storage ---
    data_dir: Path = Path("data")
    artifact_dir: Path = Path("experiments/runs")
    figure_dir: Path = Path("evaluation/figures")

    # --- database ---
    database_url: str = "sqlite:///database/claims.db"

    # --- compute ---
    device: str = "auto"
    num_threads: int | None = None

    # --- backbones ---
    text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    vision_encoder: str = "resnet18"

    # --- reproducibility ---
    seed: int = 20260921

    # --- api ---
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: str = ""

    @field_validator("data_dir", "artifact_dir", "figure_dir")
    @classmethod
    def _absolutise(cls, value: Path) -> Path:
        """Resolve relative paths against the repository root."""
        return value if value.is_absolute() else (ROOT_DIR / value).resolve()

    @field_validator("device")
    @classmethod
    def _check_device(cls, value: str) -> str:
        allowed = {"auto", "cpu", "cuda"}
        if value not in allowed:
            raise ValueError(f"device must be one of {sorted(allowed)}, got {value!r}")
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


class Paths:
    """Resolved project paths.

    Grouped as an object rather than loose module constants so that tests can
    build an alternative instance pointing at a temporary directory.
    """

    def __init__(self, settings: Settings) -> None:
        self.root = ROOT_DIR
        self.data = settings.data_dir
        self.raw = self.data / "raw"
        self.processed = self.data / "processed"
        self.synthetic = self.data / "synthetic"
        self.cache = self.data / "cache"

        self.artifacts = settings.artifact_dir
        self.figures = settings.figure_dir
        self.experiments = ROOT_DIR / "experiments"
        self.evaluation = ROOT_DIR / "evaluation"
        self.reports = self.evaluation / "reports"
        self.database = ROOT_DIR / "database"

        # Dataset-specific locations, referenced by the acquisition scripts.
        self.raw_structured = self.raw / "structured"
        self.raw_images = self.raw / "images"

    def ensure(self) -> Paths:
        """Create every managed directory. Safe to call repeatedly."""
        for attr in (
            self.raw,
            self.processed,
            self.synthetic,
            self.cache,
            self.artifacts,
            self.figures,
            self.reports,
            self.database,
            self.raw_structured,
            self.raw_images,
        ):
            attr.mkdir(parents=True, exist_ok=True)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_paths() -> Paths:
    return Paths(get_settings())


def resolve_device(requested: str | None = None) -> str:
    """Return the torch device string to use.

    ``auto`` falls back to CPU whenever CUDA is unavailable, which is the normal
    case for this project's development machine. Importing torch is deferred so
    that modules which only need paths stay cheap to import.
    """
    choice = requested or get_settings().device
    if choice != "auto":
        return choice
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency at runtime
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def configure_threads() -> None:
    """Apply the configured CPU thread count, if any.

    On a 4-core laptop, letting every library spawn its own thread pool causes
    heavy oversubscription; pinning the count measurably improves throughput.
    """
    threads = get_settings().num_threads
    if not threads:
        return
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:  # pragma: no cover
        pass


SETTINGS = get_settings()
PATHS = get_paths()
