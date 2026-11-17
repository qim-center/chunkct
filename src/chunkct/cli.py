from pathlib import Path

from cyclopts import App

from chunkct.recon import ChunkedCT
from chunkct.sources.metadata import NikonZeissTiffProjectionSource
from chunkct.sources.pickled import PickleTiffProjectionSource

app = App(name="chunkct")


def parse_simple_slice(s: str) -> slice:
    parts = s.split(":")
    if len(parts) > 2:
        raise ValueError("Follow the format start:end")
    start, stop = (int(p) if p else None for p in parts)
    return slice(start, stop)


@app.default
def main(
    config_path: str | None = None,
    tiff_dir: str | None = None,
    output_dir: str = "out",
    angle_chunks: int = 1,
    recon_chunk_shape: str | None = None,
    vol_chunk_shape: str | None = None,
    read_batch_size: int = 20,
    processes: int | None = None,
    verbose: int | None = None,
    geometry_dir: str | None = None,
    recon_range: str | None = None,
    write_projections: int = 1,
    pad_factor: float | None = None,
):
    """
    Description
    -----------
    Perform chunked CT reconstruction using the CIL framework.
    The chunking is done on projections along the angle dimension as `angle_chunks`
    and on the reconstruction volume over each axis as `vol_chunks`.

    Projections need to be written first to Zarr format:
        If reconstructing in parallel, this should be done before any reconstructions by calling with `write_projections=1`
        followed by running reconstructions afterwards with `write_projections=0` and appropriate `recon_range` for each reconstruct process.
        This is because each reconstruction process reads from the same Zarr chunks.

        If reconstructing sequentially, this is not needed.
        Although one should set `write_projections=0` if calling the function again to avoid repeating the writing.

    Parameters
    ----------
    config_path: str
        Path to CT scan metadata file. This only works for Nikon (.xtekct) and ZEISS (.txrm).
        If another format is needed, set `geometry_dir` in which the CIL AcquisitionGeometry needs to be saved in pickle fomat as 'ag.pkl'
        and the ImageGeometry as 'ig.pkl'.
    tiff_dir: str
        Path to directory of TIFF projections. Defaults to the parent folder of config_path. TIFF images are currently assumed to be uint16.
    output_dir: str
        Directory where everything is outputted.
    angle_chunks: int
        How many chunks should the projections be evenly chunked along the angle dimension.
    vol_chunks: str
        Volume chunks specified as CSV number of chunks in each dimension e.g. 2,2,2.
        This automatically distributes the chunk shapes to try to be the same.
    read_batch_size:
        Number of TIFFs per worker at a time during parallelized reading. Default 20.
        This only affects the writing of projections, that is, when `write_projections=1`.
    processes: int
        Number of processes for multiprocessing. None uses all available.
    verbose: int
        Level of intermediate output.
    geometry_dir: str
        See description under `config_path`.
    recon_range: str
        Volume chunks to reconstruct specified as start:end like Python indices (except no step).
        By default reconstructs nothing (so only relevant when writing projections with `write_projections=1`).
        Set to 'all' to reconstruct all chunks; this can be used to only write the projections without reconstructing.
    write_projections: int
        Whether (1) or not (0) to write_projections. Setting to 0 is useful when having multiple instances reconstruct the full volume.
    pad_factor: float
        Percentage to pad, applied to each side in the horizontal direction of the projections. Default None does not do padding. A value of 0.25 is usually good.
    """
    print("-" * 60)
    print("Performing chunked CT with the configurations:")
    for k, v in locals().items():
        print(f"  {k}: {v}")
    print("-" * 60 + "\n")

    recon_chunk_shape = tuple(int(d) for d in recon_chunk_shape.split(","))
    vol_chunk_shape = tuple(int(d) for d in vol_chunk_shape.split(","))

    if len(recon_chunk_shape) != 3:
        raise ValueError
    if len(vol_chunk_shape) != 3:
        raise ValueError

    if recon_range is None:
        do_reconstruct = False
    else:
        do_reconstruct = True
        recon_range = None if recon_range.lower() == "all" else parse_simple_slice(recon_range)

    write_projections = bool(write_projections)

    read_batch_size = 20 if read_batch_size is None else read_batch_size

    if config_path is not None and geometry_dir is not None:
        raise ValueError("Provide only one of config_path or geometry_dir.")

    if config_path is None and geometry_dir is None:
        raise ValueError("Either config_path or geometry_dir must be provided.")

    if tiff_dir is None:
        if config_path is None:
            raise ValueError("tiff_dir must be provided when using geometry_dir.")
        tiff_dir = Path(config_path).parent

    if config_path is not None:
        projection_source = NikonZeissTiffProjectionSource(
            config_path=config_path,
            tiff_dir=tiff_dir,
            read_batch_size=read_batch_size,
            processes=processes,
        )
    else:
        projection_source = PickleTiffProjectionSource(
            tiff_dir=tiff_dir,
            geometry_dir=geometry_dir,
            read_batch_size=read_batch_size,
            processes=processes,
        )

    processor = ChunkedCT(
        projection_source=projection_source,
        output_dir=output_dir,
        angle_chunks=angle_chunks,
        recon_chunk_shape=recon_chunk_shape,
        vol_chunk_shape=vol_chunk_shape,
        verbose=verbose,
        pad_factor=pad_factor,
    )
    if write_projections:
        processor.write_projection_chunks()

    if do_reconstruct:
        processor.reconstruct(chunks=recon_range, overwrite_on_mismatch=True)
