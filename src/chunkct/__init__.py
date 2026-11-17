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

from chunkct._compat import limit_cil_threads_to_allowed_cpus

limit_cil_threads_to_allowed_cpus()

from chunkct.recon import ChunkedCT  # noqa: E402
from chunkct.sources import (  # noqa: E402
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
