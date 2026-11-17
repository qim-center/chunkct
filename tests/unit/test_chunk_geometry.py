"""Geometry of the volume blocks (CPU only, no reconstruction).

Each block is reconstructed with its own ImageGeometry. The block's voxels must sit at exactly the
same world positions as the matching voxels of the full volume, wherever the full volume is
centred.
"""

import itertools

import numpy as np
import pytest
from cil.framework import AcquisitionGeometry
from helpers import ArraySource

from chunkct import ChunkedCT

AXES = ("z", "y", "x")  # array axis order of the volume


def voxel_centres(ig, axis):
    """World coordinates of the voxel centres of `ig` along one axis."""
    n = getattr(ig, f"voxel_num_{axis}")
    size = getattr(ig, f"voxel_size_{axis}")
    centre = getattr(ig, f"center_{axis}")
    return centre + (np.arange(n) - (n - 1) / 2) * size


@pytest.mark.parametrize("centre", [(0.0, 0.0, 0.0), (3.5, -2.0, 7.0)], ids=["centred", "offset"])
def test_blocks_match_full_volume(tmp_path, centre):
    ag = (
        AcquisitionGeometry.create_Cone3D(
            source_position=[0, -100, 0], detector_position=[0, 100, 0]
        )
        .set_panel([16, 16])
        .set_angles(np.linspace(0, 360, 4, endpoint=False))
    )
    ig = ag.get_ImageGeometry()
    # Volume sizes are not multiples of the block sizes (8, 10, 12) below, so every axis ends with
    # a smaller block: z 20 = 8+8+4, y 24 = 10+10+4, x 28 = 12+12+4.
    for axis, n, c in zip(AXES, (20, 24, 28), centre, strict=True):
        setattr(ig, f"voxel_num_{axis}", n)
        setattr(ig, f"center_{axis}", c)

    ct = ChunkedCT(
        projection_source=ArraySource(np.zeros(ag.shape, np.uint16), ag, ig),
        output_dir=tmp_path,
        angle_chunks=1,
        recon_chunk_shape=(8, 10, 12),
        vol_chunk_shape=(4, 5, 6),
        verbose=-1,
    )

    for block in itertools.product(*map(range, ct.recon_grid_shape)):
        block_ig = ct.make_ig_chunk(block)
        slices = ct.get_vol_chunk_slices(block)
        for axis, sl in zip(AXES, slices, strict=True):
            np.testing.assert_allclose(
                voxel_centres(block_ig, axis),
                voxel_centres(ig, axis)[sl],
                err_msg=f"block {block}, axis {axis}",
            )
