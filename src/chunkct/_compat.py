"""Workarounds for upstream issues. Each one says why it exists and when it can go."""

import os


def limit_cil_threads_to_allowed_cpus() -> None:
    """Cap CIL's thread count at the CPUs this process may actually use.

    CIL (26.0) sets ``cil.utilities.multiprocessing.NUM_THREADS = cpu_count() / 2``,
    counting *every* core on the machine, and processors such as
    TransmissionAbsorptionConverter call ``numba.set_num_threads(NUM_THREADS)``.
    Numba only allows as many threads as the CPUs the process is allowed to run on,
    so under a batch scheduler (e.g. an LSF job given 4 of 24 cores) that call raises
    ``ValueError: The number of threads must be between 1 and 4``.

    Remove once fixed upstream in CIL.
    """
    from cil.utilities import multiprocessing as cil_mp

    try:
        allowed = len(os.sched_getaffinity(0))
    except AttributeError:  # not available on macOS
        allowed = os.cpu_count() or 1
    try:
        import numba

        allowed = min(allowed, numba.config.NUMBA_NUM_THREADS)
    except ImportError:
        pass
    cil_mp.NUM_THREADS = max(1, min(cil_mp.NUM_THREADS, allowed))
