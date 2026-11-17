from abc import ABC, abstractmethod

import numpy as np


class ProjectionSource(ABC):
    """
    Base class for projection input sources.

    See the examples that subclass it for inspiration.
    """

    @abstractmethod
    def get_geometries(self):
        """
        Return the CIL geometries for this projection source.

        These should be the geometries for the equivalent full, unchunked
        reconstruction.

        Implementations may construct the geometries directly or load them from
        metadata files, pickle files, or another source-specific representation.

        Returns
        -------
        tuple
            (ag, ig)
            where ag is an AcquisitionGeometry and ig is an ImageGeometry from CIL's framework.
        """
        raise NotImplementedError

    @abstractmethod
    def read_chunk(self, start: int, end: int) -> np.ndarray:
        """
        Read raw projections in [start, end).

        Must return an array
            dtype: np.uint16
            shape: (end - start, detector_vertical, detector_horizontal)

        where detector_vertical and detector_horizontal are determined by
        ChunkedCT from ag.shape[1:].
        """
        raise NotImplementedError

    def process_transmission_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Optional hook for processing transmission data before conversion to
        absorption domain.

        Called after ChunkedCT has converted the raw projection chunk to np.float32.

        The default implementation returns the chunk unchanged. Override this
        method in subclasses to apply source-specific processing.

        Examples include normalization, dark/flat-field correction, clipping
        invalid values, replacing defective detector pixels, or other
        source-specific corrections.

        Parameters
        ----------
        chunk:
            Projection chunk as a np.float32 array with shape
            (chunk_angles, detector_vertical, detector_horizontal).

        Returns
        -------
        np.ndarray
            Processed transmission chunk. Must have the same shape and dtype as
            the input chunk.
        """
        return chunk
