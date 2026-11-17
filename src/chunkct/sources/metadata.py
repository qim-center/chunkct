from pathlib import Path

from cil.io import NikonDataReader, ZEISSDataReader

from chunkct.sources.tiff import TiffProjectionSource


def create_reader(file_name, roi=None):
    if file_name.endswith(".txrm"):
        DataReader = ZEISSDataReader
    elif file_name.endswith(".xtekct"):
        DataReader = NikonDataReader
    else:
        raise ValueError(
            "Unrecognizable CT metadata file. File extension should either be '.txrm' or '.xtekct'"
        )

    if roi is None:
        return DataReader(file_name=file_name)
    else:
        return DataReader(file_name=file_name, roi=roi)


class NikonZeissTiffProjectionSource(TiffProjectionSource):
    """
    TIFF projection source whose geometries are loaded from Nikon or ZEISS
    metadata using CIL.

    Projection data are read from TIFF files, not through CIL's read method.
    """

    def __init__(
        self,
        config_path,
        tiff_dir=None,
        read_batch_size=20,
        processes=None,
    ):
        self.config_path = Path(config_path)

        if tiff_dir is None:
            tiff_dir = self.config_path.parent

        super().__init__(
            tiff_dir=tiff_dir,
            read_batch_size=read_batch_size,
            processes=processes,
        )

        reader = create_reader(str(self.config_path))
        self.ag = reader.get_geometry()
        self.ig = self.ag.get_ImageGeometry()

    def get_geometries(self):
        return self.ag, self.ig
