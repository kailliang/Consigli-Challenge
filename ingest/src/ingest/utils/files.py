"""Filesystem helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_sha256(path: Path, *, chunk_size: int = 65536) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
