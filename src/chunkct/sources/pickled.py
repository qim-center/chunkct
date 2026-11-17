import pickle
from pathlib import Path

from chunkct.sources.tiff import TiffProjectionSource


class PickleTiffProjectionSource(TiffProjectionSource):
    """
    TIFF projection source whose geometries are loaded from ag.pkl and ig.pkl.
    """

    def __init__(
        self,
        tiff_dir,
        geometry_dir,
        read_batch_size=20,
        processes=None,
    ):
        self.geometry_dir = Path(geometry_dir)

        super().__init__(
            tiff_dir=tiff_dir,
            read_batch_size=read_batch_size,
            processes=processes,
        )

        with open(self.geometry_dir / "ag.pkl", "rb") as f:
            self.ag = pickle.load(f)

        with open(self.geometry_dir / "ig.pkl", "rb") as f:
            self.ig = pickle.load(f)

    def get_geometries(self):
        return self.ag, self.ig
