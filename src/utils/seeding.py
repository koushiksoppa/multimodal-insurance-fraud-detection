"""Reproducibility helpers.

Every experiment in this project is expected to be re-runnable and produce the
same numbers. That requires seeding each source of randomness that is actually
in play -- Python's ``random``, NumPy, and PyTorch -- and, for the dataset
construction code, using explicitly-seeded generators rather than global state.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

from src.utils.config import get_settings


def set_seed(seed: int | None = None, *, deterministic_torch: bool = True) -> int:
    """Seed every global random number generator and return the seed used."""
    resolved = get_settings().seed if seed is None else seed

    os.environ["PYTHONHASHSEED"] = str(resolved)
    random.seed(resolved)
    np.random.seed(resolved)

    try:
        import torch
    except ImportError:  # pragma: no cover
        return resolved

    torch.manual_seed(resolved)
    if torch.cuda.is_available():  # pragma: no cover - no CUDA on the dev machine
        torch.cuda.manual_seed_all(resolved)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return resolved


def rng(seed: int | None = None) -> np.random.Generator:
    """Return an independent NumPy generator.

    Preferred over the legacy global ``np.random`` functions inside dataset
    construction: a local generator makes the randomness of each step explicit
    and keeps unrelated code from perturbing the stream.
    """
    return np.random.default_rng(get_settings().seed if seed is None else seed)


def derive_seed(base: int, label: str) -> int:
    """Derive a stable child seed from a base seed and a label.

    Lets each stage of the pipeline (splitting, image linkage, text generation,
    model init) draw from an independent stream while remaining a deterministic
    function of the single top-level seed.
    """
    # Python's builtin hash() is salted per process for strings, so it would
    # produce a different seed on every run. SHA-256 is stable across runs.
    payload = f"{base}:{label}".encode()
    digest = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    return digest % (2**31 - 1)


@contextmanager
def temporary_seed(seed: int) -> Iterator[None]:
    """Run a block under a specific seed, then restore the previous state."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        import torch

        torch_state = torch.get_rng_state()
    except ImportError:  # pragma: no cover
        torch_state = None

    set_seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if torch_state is not None:
            import torch

            torch.set_rng_state(torch_state)
