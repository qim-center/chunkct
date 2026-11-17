"""Splitting the projections into angle chunks (CPU only, no reconstruction)."""

import numpy as np
import pytest
import zarr
from cil.framework import AcquisitionGeometry
from helpers import ArraySource

from chunkct import ChunkedCT


def make_ct(tmp_path, n_angles, angle_chunks):
    ag = (
        AcquisitionGeometry.create_Cone3D(
            source_position=[0, -100, 0], detector_position=[0, 100, 0]
        )
        .set_panel([8, 8])
        .set_angles(np.linspace(0, 360, n_angles, endpoint=False))
    )
    # every projection gets its own value, so misplaced projections would be noticed
    projections = np.broadcast_to(
        np.arange(n_angles, dtype=np.uint16)[:, None, None], ag.shape
    ).copy()
    source = ArraySource(projections, ag, ag.get_ImageGeometry())
    ct = ChunkedCT(
        projection_source=source,
        output_dir=tmp_path,
        angle_chunks=angle_chunks,
        recon_chunk_shape=(8, 8, 8),
        vol_chunk_shape=(8, 8, 8),
        verbose=-1,
    )
    return ct, projections


@pytest.mark.parametrize(("n_angles", "angle_chunks"), [(30, 1), (30, 3), (300, 7), (30, 30)])
def test_all_projections_written(tmp_path, n_angles, angle_chunks):
    ct, projections = make_ct(tmp_path, n_angles, angle_chunks)
    assert ct.angle_chunks == angle_chunks
    ct.write_projection_chunks()
    np.testing.assert_array_equal(zarr.open_array(ct.proj_zarr_path, mode="r")[:], projections)


def test_fewer_chunks_than_requested(tmp_path):
    """30 angles in 7 chunks: size 5, so 6 chunks cover everything (used to crash)."""
    with pytest.warns(UserWarning, match="using 6 instead of 7"):
        ct, projections = make_ct(tmp_path, 30, 7)
    assert (ct.angle_chunk_size, ct.angle_chunks) == (5, 6)
    ct.write_projection_chunks()
    np.testing.assert_array_equal(zarr.open_array(ct.proj_zarr_path, mode="r")[:], projections)


@pytest.mark.parametrize("angle_chunks", [0, 31])
def test_invalid_angle_chunks(tmp_path, angle_chunks):
    with pytest.raises(ValueError, match="angle_chunks must be between 1 and 30"):
        make_ct(tmp_path, 30, angle_chunks)
