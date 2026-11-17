import itertools
import json
import math
import os
import shutil
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import dask.array as da
import numpy as np
import skimage as ski
import zarr
from cil.framework import AcquisitionData
from cil.io.utilities import compress_data, get_compressed_dtype
from cil.processors import Padder, TransmissionAbsorptionConverter
from cil.recon import FDK
from tifffile import imwrite
from tqdm import tqdm

from chunkct.sources.base import ProjectionSource


def save_center_slice(arr, path):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(arr[arr.shape[0] // 2, :, :], cmap="gray")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


class ChunkedCT:
    FDK_FILTERS = tuple(FDK.preset_filters.fget(None))

    def __init__(
        self,
        projection_source: ProjectionSource,
        output_dir,
        angle_chunks,
        recon_chunk_shape,
        vol_chunk_shape,
        verbose=1,
        pad_factor=None,
        filter: str = "ram-lak",
    ):
        """
        Perform CT reconstruction by chunking the projections and reconstruction volume.

        Parameters
        ==========
        projection_source:
            Object implementing ProjectionSource.

        config_path:
            Used to load geometry unless geometry_dir is provided.

        geometry_dir:
            Directory containing ag.pkl and ig.pkl. If provided, this is used instead of config_path geometry loading.

        output_dir:
            Output directory.

        angle_chunks:
            Number of chunks along the acquisition-angle dimension.

        vol_chunks:
            Number of chunks along z, y, x.

        verbose:
            Intermediate output level.

        pad_factor:
            Horizontal projection padding factor.

        filter:
            Filter to use for FDK reconstruction. Default is "ram-lak".
            See full set of options at https://tomographicimaging.github.io/CIL/nightly/recon/#cil.recon.FDK

        =====
        Usage:
            self.write_projection_chunks() followed by self.reconstruct()
            The former may be omitted if the projection chunks are already prepared.
        """
        self.projection_source = projection_source
        self.ag, self.ig = self.projection_source.get_geometries()

        self.angle_chunks = angle_chunks
        self.recon_chunk_shape = recon_chunk_shape
        self.vol_chunk_shape = vol_chunk_shape

        if len(self.recon_chunk_shape) != 3:
            raise ValueError("recon_chunk_shape must contain exactly three integers.")
        if len(self.vol_chunk_shape) != 3:
            raise ValueError("vol_chunk_shape must contain exactly three integers.")
        for r, v in zip(self.recon_chunk_shape, self.vol_chunk_shape, strict=True):
            if r % v != 0:
                raise ValueError("vol_chunk_shape must divide recon_chunk_shape on every axis.")

        self.recon_grid_shape = tuple(
            math.ceil(self.ig.shape[i] / self.recon_chunk_shape[i]) for i in range(3)
        )
        self.vol_grid_shape = tuple(
            math.ceil(self.ig.shape[i] / self.vol_chunk_shape[i]) for i in range(3)
        )

        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.recon_zarr_path = self.output_dir / "recon.zarr"
        self.proj_zarr_path = self.output_dir / "projections.zarr"

        self.pad_factor = pad_factor

        self.filter = filter
        if filter not in self.FDK_FILTERS:
            raise ValueError(f"Invalid filter: {filter!r}. Must be one of {self.FDK_FILTERS}.")

        self.verbose = 1 if verbose is None else verbose
        if self.verbose:
            os.makedirs(self.output_dir / "center", exist_ok=True)

        self.proj_shape = self.ag.shape

        n_angles = self.proj_shape[0]
        if not 1 <= self.angle_chunks <= n_angles:
            raise ValueError(f"angle_chunks must be between 1 and {n_angles}, got {angle_chunks}.")
        self.angle_chunk_size = math.ceil(n_angles / self.angle_chunks)
        # Chunks have equal size except a smaller last one, so fewer chunks than requested can
        # be enough: 30 angles in 7 chunks gives size 5, and 6 chunks of 5 already cover all 30.
        self.angle_chunks = math.ceil(n_angles / self.angle_chunk_size)
        if self.angle_chunks != angle_chunks:
            warnings.warn(
                f"{n_angles} angles in chunks of {self.angle_chunk_size} need only "
                f"{self.angle_chunks} angle chunks; using {self.angle_chunks} instead of "
                f"{angle_chunks}.",
                stacklevel=2,
            )

    def get_vol_chunk_slices(self, chunk_coords):
        slices = []

        for axis, chunk_index in enumerate(chunk_coords):
            start = chunk_index * self.recon_chunk_shape[axis]
            stop = min(start + self.recon_chunk_shape[axis], self.ig.shape[axis])
            slices.append(slice(start, stop))

        return tuple(slices)

    def write_projection_chunk(self, chunk_index):
        start = chunk_index * self.angle_chunk_size
        end = min((chunk_index + 1) * self.angle_chunk_size, self.proj_shape[0])

        chunk = self.projection_source.read_chunk(start, end)
        if chunk.dtype != np.uint16:
            raise TypeError(f"Chunk must have dtype np.uint16, got {chunk.dtype}")

        expected_shape = (end - start, *self.proj_shape[1:])
        if chunk.shape != expected_shape:
            raise ValueError(f"Chunk shape mismatch: expected {expected_shape}, got {chunk.shape}")

        proj_zarr = zarr.open_array(store=self.proj_zarr_path, mode="a")
        proj_zarr.blocks[chunk_index] = chunk

        if self.verbose >= 0:
            print(f"Finished writing projection chunk {chunk_index}.")

    def print_projection_compression_factor(self, proj_zarr):
        stored_bytes = proj_zarr.nbytes_stored()
        uncompressed_bytes = proj_zarr.nbytes
        compression_factor = uncompressed_bytes / stored_bytes

        print(
            f"Projection Zarr size: {stored_bytes / 2**30:.3f} GiB; "
            f"raw size: {uncompressed_bytes / 2**30:.3f} GiB; "
            f"compression factor: {compression_factor:.2f}x"
        )

    def _proj_chunk_exists(self, chunk_index):
        # zarr v3 LocalStore default chunk key: c/{i}/0/0 for a 3-D array
        return (self.proj_zarr_path / "c" / str(chunk_index) / "0" / "0").exists()

    def write_projection_chunks(self, overwrite=False):
        t = time.time()

        expected_shape = self.proj_shape
        expected_chunks = (self.angle_chunk_size, *self.proj_shape[1:])

        if overwrite and self.proj_zarr_path.exists():
            shutil.rmtree(self.proj_zarr_path)

        if self.proj_zarr_path.exists():
            proj_zarr = zarr.open_array(store=self.proj_zarr_path, mode="a")

            mismatches = []
            if proj_zarr.shape != expected_shape:
                mismatches.append(f"shape: existing {proj_zarr.shape}, expected {expected_shape}")
            if proj_zarr.chunks != expected_chunks:
                mismatches.append(
                    f"chunks: existing {proj_zarr.chunks}, expected {expected_chunks}"
                )

            if mismatches:
                raise ValueError(
                    "Existing projection Zarr metadata does not match the current configuration:\n"
                    + "\n".join(f"  - {m}" for m in mismatches)
                    + "\n\nRerun with overwrite=True to overwrite."
                )

            missing = [i for i in range(self.angle_chunks) if not self._proj_chunk_exists(i)]

            if self.verbose >= 0:
                n_existing = self.angle_chunks - len(missing)
                print(
                    f"Found existing projection Zarr: {n_existing}/{self.angle_chunks} chunks present. "
                    f"Writing {len(missing)} missing chunk(s)."
                )
        else:
            proj_zarr = zarr.open_array(
                store=self.proj_zarr_path,
                mode="w",
                dtype=np.uint16,
                shape=expected_shape,
                chunks=expected_chunks,
                zarr_format=3,
            )
            missing = list(range(self.angle_chunks))

        for i in missing:
            self.write_projection_chunk(i)

        if self.verbose > 0:
            self.print_projection_compression_factor(proj_zarr)

        if self.verbose >= 0:
            print(f"Writing projection chunks took {time.time() - t} s")

    def reconstruct_projection_chunk(self, ig, proj_chunk_index):
        start = proj_chunk_index * self.angle_chunk_size
        end = min((proj_chunk_index + 1) * self.angle_chunk_size, self.ag.shape[0])
        ag_chunk = self.ag.copy().set_angles(self.ag.angles[start:end])

        proj_zarr = zarr.open_array(store=self.proj_zarr_path, mode="r")
        chunk = proj_zarr.blocks[proj_chunk_index].astype(np.float32)
        expected_shape = chunk.shape
        chunk = self.projection_source.process_transmission_chunk(chunk)

        if chunk.dtype != np.float32:
            raise TypeError(f"process_transmission_chunk must return np.float32, got {chunk.dtype}")
        if chunk.shape != expected_shape:
            raise ValueError(
                f"process_transmission_chunk must preserve shape: "
                f"expected {expected_shape}, got {chunk.shape}"
            )

        data_chunk = AcquisitionData(array=chunk, deep_copy=False, geometry=ag_chunk)
        data_chunk = TransmissionAbsorptionConverter()(data_chunk)

        if self.pad_factor is not None:
            dim_horizontal = data_chunk.get_data_axes_order().index("horizontal")
            pad_width = round(self.pad_factor * data_chunk.shape[dim_horizontal])
            data_chunk = Padder.edge(pad_width={"horizontal": pad_width})(data_chunk)

        data_chunk.reorder(order="tigre")
        recon_chunk = FDK(data_chunk, image_geometry=ig, filter=self.filter).run(verbose=0)

        # CIL's FDK weights by 2*pi / (number of projections it is given), i.e. it treats
        # this chunk as if it were a complete scan. Rescale so the chunk contributes its
        # share of the full scan; the sum over chunks then equals the unchunked FDK.
        recon_array = recon_chunk.as_array()
        recon_array *= np.float32((end - start) / self.ag.shape[0])
        return recon_chunk

    def make_ig_chunk(self, chunk_coords):
        ig_chunk = self.ig.copy()
        for axis, dim_label in enumerate(["z", "y", "x"]):
            chunk = chunk_coords[axis]
            voxel_num = min(
                self.recon_chunk_shape[axis],
                self.ig.shape[axis] - chunk * self.recon_chunk_shape[axis],
            )
            setattr(ig_chunk, f"voxel_num_{dim_label}", voxel_num)
            ig_voxel_num = getattr(self.ig, f"voxel_num_{dim_label}")
            ig_chunk_voxel_num = getattr(ig_chunk, f"voxel_num_{dim_label}")
            ig_chunk_voxel_size = getattr(ig_chunk, f"voxel_size_{dim_label}")

            # center is in 'world' coordinates, hence the scaling with voxel_size
            initial_shift = (
                -ig_voxel_num / 2 + self.recon_chunk_shape[axis] / 2
            ) * ig_chunk_voxel_size
            center = (
                initial_shift
                + (
                    chunk * self.recon_chunk_shape[axis]
                    - (self.recon_chunk_shape[axis] - ig_chunk_voxel_num) / 2
                )
                * ig_chunk_voxel_size
            )
            setattr(ig_chunk, f"center_{dim_label}", center)

        return ig_chunk

    def reconstruct_volume_chunk(self, chunk_coords):
        t = time.time()
        str_chunk = "_".join(str(i) for i in chunk_coords)

        ig_chunk = self.make_ig_chunk(chunk_coords)
        vol_accumulated = np.zeros(ig_chunk.shape, dtype=np.float32)

        for proj_chunk_index in range(self.angle_chunks):
            recon_chunk = self.reconstruct_projection_chunk(ig_chunk, proj_chunk_index)
            vol_accumulated += recon_chunk.as_array()

            if self.verbose > 2:
                save_center_slice(
                    recon_chunk.as_array(),
                    self.output_dir / f"center/recon_vol_{str_chunk}_proj_{proj_chunk_index}.png",
                )

        if self.verbose > 1:
            save_center_slice(
                vol_accumulated, self.output_dir / f"center/recon_{str_chunk}_accumulated.png"
            )

        t_write = time.time()
        recon_zarr = zarr.open_array(store=self.recon_zarr_path, mode="a")
        recon_slices = self.get_vol_chunk_slices(chunk_coords)
        recon_zarr[recon_slices] = vol_accumulated

        if self.verbose >= 0:
            print(f"Finished volume chunk {chunk_coords}. Took {time.time() - t} s")
            print(f"  Writing the volume chunk to disk took {time.time() - t_write} s")

    def reconstruct(self, chunks=None, overwrite_on_mismatch=False):
        """
        chunks: slice object specifying the ravelled indices to reconstruct.
        """
        t = time.time()

        recon_zarr_shape = self.ig.shape
        recon_zarr_chunks = self.vol_chunk_shape

        ##### Validation of metadata ######
        if self.recon_zarr_path.exists():
            recon_zarr = zarr.open_array(store=self.recon_zarr_path, mode="a")

            mismatches = []

            if recon_zarr_shape != recon_zarr.shape:
                mismatches.append(
                    f"recon_zarr_shape: existing {recon_zarr.shape}, requested {recon_zarr_shape}"
                )
            if recon_zarr_chunks != recon_zarr.chunks:
                mismatches.append(
                    f"recon_zarr_chunks: existing {recon_zarr.chunks}, requested {recon_zarr_chunks}"
                )

            if mismatches:
                message = (
                    "Existing reconstruction Zarr metadata does not match the chunking configuration:\n"
                    + "\n".join(f"  - {m}" for m in mismatches)
                )

                if not overwrite_on_mismatch:
                    raise ValueError(
                        message
                        + "\n\nRefusing to continue because this could mix incompatible reconstruction output. "
                        + "Delete recon.zarr manually or rerun with overwrite_on_mismatch=True."
                    )
                print(message + "\n\nOverwriting recon.zarr since overwrite_on_mismatch=True")
                shutil.rmtree(self.recon_zarr_path)
        #######################

        zarr.open_array(
            store=self.recon_zarr_path,
            mode="a",
            dtype=np.float32,
            shape=recon_zarr_shape,
            chunks=recon_zarr_chunks,
            zarr_format=3,
        )
        full_range = list(itertools.product(*map(range, self.recon_grid_shape)))
        chunk_range = full_range[chunks] if chunks else full_range
        for chunk_coords in chunk_range:
            self.reconstruct_volume_chunk(chunk_coords)

        print(f"Chunked reconstruction took {time.time() - t} s")

    def _compression_scale_offset(self, z, compression, downsample_factor=(1, 1, 1)):
        """
        Compute the scale and offset used to rescale the volume into an unsigned
        integer type, mirroring cil.io.utilities.get_compression_scale_offset.
        The volume's min/max are computed lazily with dask so the full array
        never has to be loaded into memory at once.

        If downsample_factor is not all 1, the min/max are computed from the
        volume coarsened by local mean (matching what will actually be
        written), since local averaging shrinks the value range.
        """
        if compression is None:
            return 1.0, 0.0

        print("Computing scale/offset for compression. This may take a while for large volumes...")
        dtype = get_compressed_dtype(z, compression)
        save_range = np.iinfo(dtype).max

        arr = da.from_zarr(str(self.recon_zarr_path))
        if tuple(downsample_factor) != (1, 1, 1):
            arr = da.coarsen(
                np.mean, arr, {i: f for i, f in enumerate(downsample_factor)}, trim_excess=True
            )
        data_min, data_max = da.compute(arr.min(), arr.max())
        data_range = data_max - data_min

        if data_range > 0:
            scale = save_range / data_range
            offset = -data_min * scale
        else:
            scale = 1.0
            offset = 0.0

        return float(scale), float(offset)

    def zarr_recon_to_tiff(
        self, name="recon", processes=None, compression=None, downsample_factor=None
    ):
        """
        Convert from reconstructed zarr array to stack of tiffs.
        Assumes chunks are stored as vertical slices.

        Parameters
        ----------
        name:
            Filename prefix for the TIFF slices.
        processes:
            Number of processes for multiprocessing. None uses all available.
        compression: str, default None. Accepted values None, 'uint8', 'uint16'.
            Lossy compression to apply, mirroring cil.io.TIFFWriter. The default
            None writes the reconstruction unchanged (float32). 'uint8' or
            'uint16' rescales the whole volume to that unsigned integer type
            using a single scale/offset computed from the volume's min/max, and
            saves the scale/offset to '{name}_scaleoffset.json' in the TIFF
            output directory. The original data can be recovered as
            `(compressed_data - offset) / scale`, e.g. with
            cil.io.TIFFStackReader.read_rescaled().
        downsample_factor: int or tuple of 3 ints, default None.
            Reduce the resolution of the volume before writing, by averaging
            (local mean) blocks of voxels along z, y, x. A single int applies
            the same factor to all three axes. Default None writes the volume
            at full resolution. The scale/offset for `compression` is computed
            from the already-downsampled volume.
        """
        if compression not in (None, "uint8", "uint16"):
            raise ValueError(
                f"Invalid compression: {compression!r}. Must be one of None, 'uint8', 'uint16'."
            )

        if downsample_factor is None:
            downsample_factor = (1, 1, 1)
        elif isinstance(downsample_factor, int):
            downsample_factor = (downsample_factor,) * 3
        else:
            downsample_factor = tuple(downsample_factor)
        if len(downsample_factor) != 3 or any(f < 1 for f in downsample_factor):
            raise ValueError(
                f"Invalid downsample_factor: {downsample_factor!r}. Must be a positive int or a tuple of 3 positive ints."
            )
        z_factor, y_factor, x_factor = downsample_factor

        if compression is None:
            recon_tiff_path = self.output_dir / "recon_tiff"
        else:
            folder = "recon_tiff_" + compression
            recon_tiff_path = self.output_dir / folder
        os.makedirs(recon_tiff_path, exist_ok=True)
        self.recon_tiff_path = recon_tiff_path
        z = zarr.open(str(self.recon_zarr_path), mode="r")
        n_slices = math.ceil(
            z.shape[0] / z_factor
        )  # TODO: generalize to other kinds of chunk shapes

        dtype = get_compressed_dtype(z, compression)
        scale, offset = self._compression_scale_offset(z, compression, downsample_factor)

        args = [
            (
                self.recon_zarr_path,
                recon_tiff_path / f"{name}_slice_{idx:04d}.tiff",
                idx * z_factor,
                min((idx + 1) * z_factor, z.shape[0]),
                (y_factor, x_factor),
                scale,
                offset,
                dtype,
            )
            for idx in range(n_slices)
        ]

        with Pool(processes) as pool:
            list(
                tqdm(pool.imap(_write_tiff_slice, args), total=n_slices, desc="Writing TIFF slices")
            )

        if compression is not None:
            scaleoffset_path = recon_tiff_path / f"{name}_scaleoffset.json"
            with open(scaleoffset_path, "w") as f:
                json.dump({"scale": scale, "offset": offset}, f)
            if self.verbose >= 0:
                print(
                    f"Saved scale/offset {{'scale': {scale}, 'offset': {offset}}} to {scaleoffset_path}"
                )

        print(f"Saved {n_slices} TIFF slices to {recon_tiff_path}")


def _write_tiff_slice(args):
    zarr_path, tiff_path, z_start, z_end, xy_factor, scale, offset, dtype = args
    z = zarr.open(str(zarr_path), mode="r")
    block = np.array(z[z_start:z_end])

    y_factor, x_factor = xy_factor
    if block.shape[0] > 1 or y_factor > 1 or x_factor > 1:
        block = ski.transform.downscale_local_mean(block, (block.shape[0], y_factor, x_factor))

    slice_arr = compress_data(block[0], scale, offset, dtype)
    imwrite(str(tiff_path), slice_arr)
