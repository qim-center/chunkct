"""Shared fixtures, and the CPU/GPU test split.

Two kinds of tests for now:
- unmarked: fast, CPU only. `pixi run test` (or `pixi run -e cpu test` on a login node).
- @pytest.mark.gpu: needs a CUDA GPU + TIGRE (real FDK). `pixi run test-gpu`.
  Selecting them on a machine without a usable GPU stops the run with one clear error.
"""

from __future__ import annotations

import functools

import pytest
from helpers import to_uint16_transmission


@functools.cache
def gpu_available() -> bool:
    try:
        from tigre.utilities.gpu import getGpuNames

        return len(getGpuNames()) > 0
    except Exception:
        return False


def pytest_collection_finish(session):
    """Stop with one clear error if GPU tests were selected but no GPU can be used.

    Runs after `-m` has deselected tests, so `pixi run test` (`-m 'not gpu'`) is unaffected.
    """
    if any(item.get_closest_marker("gpu") for item in session.items) and not gpu_available():
        raise pytest.UsageError(
            "GPU tests were selected, but no CUDA GPU / TIGRE is available. "
            "Run them on a GPU node, or select only the CPU tests with `pixi run test`."
        )


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
@functools.cache
def _sim_cone_full():
    """CIL's bundled simulated cone-beam data: (300, 128, 128) transmission, no download."""
    from cil.utilities import dataexample

    data = dataexample.SIMULATED_CONE_BEAM_DATA.get()
    return data.geometry, to_uint16_transmission(data.as_array())


@pytest.fixture
def sim_cone():
    """(ag, ig, uint16 projections) at full size: 300 x 128 x 128 -> 128^3 volume."""
    ag, proj = _sim_cone_full()
    ag = ag.copy()
    return ag, ag.get_ImageGeometry(), proj.copy()
