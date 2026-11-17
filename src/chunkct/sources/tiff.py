import re
from abc import abstractmethod
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import skimage as ski

from chunkct.sources.base import ProjectionSource


class TiffProjectionSource(ProjectionSource):
    """
    Base class for projection sources backed by a directory of TIFF files.

    Subclasses must provide the corresponding CIL geometries.
    TIFFs are assumed to be uint16 transmission images.
    """

    def __init__(
        self,
        tiff_dir,
        read_batch_size=20,
        processes=None,
    ):
        self.tiff_dir = Path(tiff_dir)
        self.read_batch_size = read_batch_size
        self.processes = processes

        self.paths = self._find_paths()

        if len(self.paths) == 0:
            raise ValueError(f"No TIFF files found in {self.tiff_dir}")

        self.dtype = np.uint16
        self._normalization_constant = float(np.iinfo(self.dtype).max)

    @abstractmethod
    def get_geometries(self):
        raise NotImplementedError

    def process_transmission_chunk(self, chunk: np.ndarray) -> np.ndarray:
        chunk /= self._normalization_constant
        return chunk

    def _find_paths(self):
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(path):
            return [atoi(c) for c in re.split(r"(\d+)", path.name)]

        paths = [
            *self.tiff_dir.glob("*.tif"),
            *self.tiff_dir.glob("*.tiff"),
        ]

        return sorted(paths, key=natural_keys)

    @staticmethod
    def read_one(path):
        return ski.io.imread(path)

    def read_chunk(self, start, end):
        paths = self.paths[start:end]

        expected_count = end - start
        if len(paths) != expected_count:
            raise ValueError(
                f"Requested {expected_count} TIFF projections, but found {len(paths)} "
                f"for interval [{start}, {end})."
            )

        with Pool(self.processes) as pool:
            chunk = np.stack(
                pool.map(
                    TiffProjectionSource.read_one,
                    paths,
                    chunksize=self.read_batch_size,
                ),
                axis=0,
            )

        if chunk.dtype != np.uint16:
            raise TypeError(f"Expected TIFF chunk dtype np.uint16, got {chunk.dtype}")

        return chunk
