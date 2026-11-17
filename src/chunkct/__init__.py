"""Chunked CT reconstruction on top of CIL."""

from importlib.metadata import PackageNotFoundError, version

try:
    import cil  # noqa: F401
except ImportError as e:  # pragma: no cover - depends on environment
    raise ImportError(
        "chunkct requires the Core Imaging Library (CIL), which is not available on PyPI. "
        "Install it from conda, e.g. `conda install -c conda-forge -c ccpi cil tigre`, "
        "or use the pixi environment shipped with chunkct (`pixi install`)."
    ) from e

from chunkct.recon import ChunkedCT
from chunkct.sources import (
    NikonZeissTiffProjectionSource,
    PickleTiffProjectionSource,
    ProjectionSource,
    TiffProjectionSource,
)

try:
    __version__ = version("chunkct")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "ChunkedCT",
    "NikonZeissTiffProjectionSource",
    "PickleTiffProjectionSource",
    "ProjectionSource",
    "TiffProjectionSource",
    "__version__",
]
