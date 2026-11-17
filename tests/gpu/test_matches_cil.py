"""End-to-end: chunked reconstruction must match a plain, unchunked CIL FDK.

FDK is linear in the projections and every voxel is computed independently, so
splitting the angles into chunks (and summing) and splitting the volume into blocks
should reproduce the full reconstruction up to float32 rounding. These tests run
the whole ChunkedCT pipeline (projection Zarr -> per-chunk FDK -> recon Zarr) on
CIL's simulated cone-beam data (300 angles, 128 x 128 detector -> 128^3 volume).
"""

import pickle

import numpy as np
import pytest
import zarr
from cil.framework import AcquisitionData
from cil.processors import Padder, TransmissionAbsorptionConverter
from cil.recon import FDK
from helpers import UINT16_MAX, ArraySource, write_tiff_stack

from chunkct import ChunkedCT, PickleTiffProjectionSource

pytestmark = pytest.mark.gpu


def cil_reference(ag, ig, projections_uint16, pad_factor=None, filter="ram-lak"):
    """The 'ground truth': ordinary CIL FDK on all projections at once."""
    data = AcquisitionData(
        projections_uint16.astype(np.float32) / UINT16_MAX, geometry=ag, deep_copy=True
    )
    data = TransmissionAbsorptionConverter()(data)
    if pad_factor is not None:
        dim_horizontal = data.get_data_axes_order().index("horizontal")
        pad_width = round(pad_factor * data.shape[dim_horizontal])
        data = Padder.edge(pad_width={"horizontal": pad_width})(data)
    data.reorder(order="tigre")
    return FDK(data, image_geometry=ig, filter=filter).run(verbose=0).as_array()


def assert_close(chunked, reference, rel_tol=1e-3):
    """Max abs difference relative to the reference's dynamic range.

    Chunking is not bit-identical to an unchunked FDK: TIGRE interpolates on the GPU
    with limited (fixed-point) precision and a sub-volume shifts the sample
    coordinates slightly, giving differences around 1e-4 of the range. A real bug
    (e.g. wrong chunk weighting or placement) gives differences of order 1.
    """
    assert chunked.shape == reference.shape
    dynamic_range = float(reference.max() - reference.min())
    diff = np.abs(chunked - reference)
    rel_max = float(diff.max()) / dynamic_range
    rel_mean = float(diff.mean()) / dynamic_range
    print(f"|chunked - reference| / range: max = {rel_max:.2e}, mean = {rel_mean:.2e}")
    assert rel_max < rel_tol


def run_chunked(source, out_dir, angle_chunks, recon_chunk_shape, vol_chunk_shape, **kwargs):
    ct = ChunkedCT(
        projection_source=source,
        output_dir=out_dir,
        angle_chunks=angle_chunks,
        recon_chunk_shape=recon_chunk_shape,
        vol_chunk_shape=vol_chunk_shape,
        verbose=0,
        **kwargs,
    )
    ct.write_projection_chunks()
    ct.reconstruct()
    return zarr.open_array(ct.recon_zarr_path, mode="r")[:]


CHUNKINGS = {
    # name: (angle_chunks, recon_chunk_shape, vol_chunk_shape)
    "unchunked": (1, (128, 128, 128), (128, 128, 128)),
    "even": (3, (64, 64, 64), (32, 32, 32)),
    "uneven": (7, (50, 70, 128), (10, 10, 16)),  # remainders on every axis and in angles
}


@pytest.mark.parametrize("chunking", CHUNKINGS.values(), ids=CHUNKINGS.keys())
@pytest.mark.parametrize("pad_factor", [None, 0.25], ids=["nopad", "pad"])
def test_chunked_matches_cil(tmp_path, sim_cone, chunking, pad_factor):
    ag, ig, proj = sim_cone
    angle_chunks, recon_chunk_shape, vol_chunk_shape = chunking

    chunked = run_chunked(
        ArraySource(proj, ag, ig),
        tmp_path,
        angle_chunks,
        recon_chunk_shape,
        vol_chunk_shape,
        pad_factor=pad_factor,
    )
    assert_close(chunked, cil_reference(ag, ig, proj, pad_factor=pad_factor))


@pytest.mark.parametrize("filter", ["shepp-logan", "hann"])
def test_chunked_matches_cil_other_filters(tmp_path, sim_cone, filter):
    ag, ig, proj = sim_cone
    chunked = run_chunked(ArraySource(proj, ag, ig), tmp_path, *CHUNKINGS["even"], filter=filter)
    assert_close(chunked, cil_reference(ag, ig, proj, filter=filter))


def test_tiff_source_matches_cil(tmp_path, sim_cone):
    """Same check, but reading TIFFs from disk via PickleTiffProjectionSource."""
    ag, ig, proj = sim_cone
    write_tiff_stack(tmp_path / "tiffs", proj)
    for name, geometry in (("ag", ag), ("ig", ig)):
        with open(tmp_path / f"{name}.pkl", "wb") as f:
            pickle.dump(geometry, f)

    source = PickleTiffProjectionSource(
        tiff_dir=tmp_path / "tiffs", geometry_dir=tmp_path, processes=2
    )
    chunked = run_chunked(source, tmp_path / "out", *CHUNKINGS["uneven"])
    assert_close(chunked, cil_reference(ag, ig, proj))
