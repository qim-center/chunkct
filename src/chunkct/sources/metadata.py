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


def read_nikon_white_level(xtekct_path) -> float:
    """Return ``WhiteLevel`` from a Nikon .xtekct file: the detector value for no attenuation."""
    with open(xtekct_path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("WhiteLevel="):
                return float(line.split("=", 1)[1])
    raise ValueError(f"No WhiteLevel entry found in {xtekct_path}")


class NikonZeissTiffProjectionSource(TiffProjectionSource):
    """
    TIFF projection source whose geometries are loaded from Nikon or ZEISS
    metadata using CIL.

    Projection data are read from TIFF files, not through CIL's read method.

    Normalisation to transmission (raw counts / I0):
    - Nikon: divides by ``WhiteLevel`` from the .xtekct, like CIL's NikonDataReader.
    - ZEISS: divides by 65535 (dtype maximum). CIL's ZEISSDataReader instead uses
      reference (flat-field) images; not implemented here yet.
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

        if self.config_path.suffix == ".xtekct":
            self._normalization_constant = read_nikon_white_level(self.config_path)

    def get_geometries(self):
        return self.ag, self.ig
