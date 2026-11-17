"""Compare a chunked reconstruction of a real Nikon scan with plain CIL FDK.

Runs on a GPU node, e.g. as an LSF job. Steps:

1. Chunked: NikonZeissTiffProjectionSource -> ChunkedCT -> <output_dir>/recon.zarr
2. Reference: CIL's NikonDataReader reads ALL projections into memory, normalised by the
   WhiteLevel from the .xtekct (as chunkct does) -> plain, unchunked FDK
   -> <output_dir>/reference.zarr

Writes metrics.json and center-slice PNGs to <output_dir>/validation/.
The full reference needs all projections in RAM (~4 bytes x angles x pixels, times ~3).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import zarr
from cil.framework import AcquisitionData
from cil.io import NikonDataReader
from cil.processors import Padder, TransmissionAbsorptionConverter
from cil.recon import FDK
from cyclopts import App

from chunkct import ChunkedCT, NikonZeissTiffProjectionSource
from chunkct.sources.metadata import read_nikon_white_level

app = App()


def parse_shape(s: str) -> tuple[int, int, int]:
    shape = tuple(int(v) for v in s.split(","))
    if len(shape) != 3:
        raise ValueError(f"Expected three comma-separated integers, got {s!r}")
    return shape


def reference_fdk(raw: np.ndarray, ag, ig, normalization, pad_factor, filter):
    data = AcquisitionData(raw / np.float32(normalization), geometry=ag, deep_copy=False)
    data = TransmissionAbsorptionConverter()(data)
    if pad_factor is not None:
        dim_horizontal = data.get_data_axes_order().index("horizontal")
        pad_width = round(pad_factor * data.shape[dim_horizontal])
        data = Padder.edge(pad_width={"horizontal": pad_width})(data)
    data.reorder(order="tigre")
    return FDK(data, image_geometry=ig, filter=filter).run(verbose=0).as_array()


def compare(a: np.ndarray, ref: np.ndarray) -> dict:
    diff = np.abs(a - ref)
    dynamic_range = float(ref.max() - ref.min())
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "reference_range": dynamic_range,
        "max_abs_diff_rel_range": float(diff.max()) / dynamic_range,
        "mean_abs_diff_rel_range": float(diff.mean()) / dynamic_range,
    }


def save_center_slices(volumes: dict[str, np.ndarray], path: Path):
    import matplotlib.pyplot as plt

    names = list(volumes)
    fig, axes = plt.subplots(len(names), 3, figsize=(12, 4 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        v = volumes[name]
        center = [s // 2 for s in v.shape]
        slices = (v[center[0]], v[:, center[1]], v[:, :, center[2]])
        for col, (axis_name, sl) in enumerate(zip(("z", "y", "x"), slices, strict=True)):
            im = axes[row, col].imshow(sl, cmap="gray")
            axes[row, col].set_title(f"{name} — center {axis_name}")
            fig.colorbar(im, ax=axes[row, col], fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


@app.default
def main(
    config_path: Path,
    output_dir: Path,
    angle_chunks: int = 4,
    recon_chunk_shape: str = "500,500,500",
    vol_chunk_shape: str = "100,100,100",
    pad_factor: float | None = None,
    filter: str = "ram-lak",
):
    """Validate chunked reconstruction against plain CIL FDK on a real Nikon scan.

    Parameters
    ----------
    config_path
        Nikon .xtekct file (TIFF projections are read from its folder).
    output_dir
        New directory for all outputs (projections.zarr, recon.zarr, reference.zarr, report).
    angle_chunks, recon_chunk_shape, vol_chunk_shape, pad_factor, filter
        Passed to ChunkedCT.
    """
    output_dir = Path(output_dir)
    report_dir = output_dir / "validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    timings = {}

    # 1. chunked
    t = time.time()
    source = NikonZeissTiffProjectionSource(config_path=config_path)
    ct = ChunkedCT(
        projection_source=source,
        output_dir=output_dir,
        angle_chunks=angle_chunks,
        recon_chunk_shape=parse_shape(recon_chunk_shape),
        vol_chunk_shape=parse_shape(vol_chunk_shape),
        verbose=0,
        pad_factor=pad_factor,
        filter=filter,
    )
    ct.write_projection_chunks()
    ct.reconstruct(overwrite_on_mismatch=True)
    chunked = zarr.open_array(ct.recon_zarr_path, mode="r")[:]
    timings["chunked_s"] = time.time() - t
    ag, ig = source.get_geometries()

    # 2. reference: read all projections at once, as raw counts
    t = time.time()
    reader = NikonDataReader(file_name=str(config_path), normalise=False)
    raw = reader.read().as_array()
    white_level = read_nikon_white_level(config_path)
    timings["read_raw_s"] = time.time() - t

    volumes = {"chunked": chunked}
    metrics = {
        "config_path": str(config_path),
        "proj_shape": list(ag.shape),
        "volume_shape": list(ig.shape),
        "white_level": white_level,
        "chunking": {
            "angle_chunks": angle_chunks,
            "recon_chunk_shape": recon_chunk_shape,
            "vol_chunk_shape": vol_chunk_shape,
            "pad_factor": pad_factor,
            "filter": filter,
        },
    }
    t = time.time()
    ref = reference_fdk(raw, reader.get_geometry(), ig, white_level, pad_factor, filter)
    timings["reference_s"] = time.time() - t
    del raw
    out = zarr.open_array(
        str(output_dir / "reference.zarr"), mode="w", shape=ref.shape, dtype=ref.dtype
    )
    out[:] = ref
    metrics["chunked_vs_reference"] = compare(chunked, ref)
    print("chunked vs reference", json.dumps(metrics["chunked_vs_reference"], indent=2), flush=True)

    volumes["reference"] = ref
    volumes["|chunked - reference|"] = np.abs(chunked - ref)

    metrics["timings"] = timings
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_center_slices(volumes, report_dir / "center_slices.png")
    print(f"Report written to {report_dir}")


if __name__ == "__main__":
    app()
