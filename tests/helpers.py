"""Helpers shared by tests (plain module; fixtures live in conftest.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from chunkct.sources.base import ProjectionSource

UINT16_MAX = np.iinfo(np.uint16).max


class ArraySource(ProjectionSource):
    """In-memory projection source used throughout the tests."""

    def __init__(self, projections: np.ndarray, ag, ig):
        self.projections = projections
        self.ag = ag
        self.ig = ig

    def get_geometries(self):
        return self.ag, self.ig

    def read_chunk(self, start, end):
        return self.projections[start:end]

    def process_transmission_chunk(self, chunk):
        chunk /= UINT16_MAX
        return chunk


def to_uint16_transmission(transmission: np.ndarray) -> np.ndarray:
    return np.round(np.clip(transmission, 0, 1) * UINT16_MAX).astype(np.uint16)


def write_tiff_stack(directory: Path, projections: np.ndarray, stem="proj_", pad=0) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(projections):
        tifffile.imwrite(directory / f"{stem}{i:0{pad}d}.tif", p)
