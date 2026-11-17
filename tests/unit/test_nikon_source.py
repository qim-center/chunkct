"""Tests for the Nikon source, on a tiny fake scan written to a temporary folder (CPU only).

A Nikon scan is a `.xtekct` text file with the scan settings plus one TIFF per projection in the
same folder. The fake scan has 4 projections of 8 x 10 pixels in which every pixel has the same raw
value, so the expected transmission is known exactly.
"""

import numpy as np
import pytest
from helpers import write_tiff_stack

from chunkct import NikonZeissTiffProjectionSource
from chunkct.sources.metadata import read_nikon_white_level

N_ANGLES, N_ROWS, N_COLS = 4, 8, 10
WHITE_LEVEL = 60000  # raw detector value with nothing in the beam (stored in the .xtekct)
RAW_VALUE = 30000  # raw value of every pixel in the fake projections


def write_xtekct(path, white_level=WHITE_LEVEL):
    """Write a minimal `.xtekct` with only the settings CIL's NikonDataReader needs.

    The distances and pixel sizes are arbitrary but valid; the tests only check shapes.
    `white_level=None` leaves out the WhiteLevel line.
    """
    settings = {
        "Name": "fake",
        "InputFolderName": "",
        "Projections": N_ANGLES,
        "DetectorPixelsX": N_COLS,
        "DetectorPixelsY": N_ROWS,
        "DetectorPixelSizeX": 0.2,
        "DetectorPixelSizeY": 0.2,
        "SrcToObject": 100,
        "SrcToDetector": 500,
        "InitialAngle": 0,
        "AngularStep": 360 / N_ANGLES,
        "ObjectOffsetX": 0,
        "ObjectRoll": 0,
    }
    if white_level is not None:
        settings["WhiteLevel"] = white_level
    lines = ["[XTekCT]", *(f"{key}={value}" for key, value in settings.items())]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def fake_scan(tmp_path):
    """Folder with `fake.xtekct` and `fake_0000.tif` … `fake_0003.tif`; returns the .xtekct path."""
    projections = np.full((N_ANGLES, N_ROWS, N_COLS), RAW_VALUE, dtype=np.uint16)
    write_tiff_stack(tmp_path, projections, stem="fake_", pad=4)
    return write_xtekct(tmp_path / "fake.xtekct")


def test_reads_white_level(tmp_path):
    path = write_xtekct(tmp_path / "scan.xtekct")
    assert read_nikon_white_level(path) == WHITE_LEVEL


def test_missing_white_level_is_an_error(tmp_path):
    path = write_xtekct(tmp_path / "scan.xtekct", white_level=None)
    with pytest.raises(ValueError, match="WhiteLevel"):
        read_nikon_white_level(path)


def test_geometry_matches_scan(fake_scan):
    # processes=1: read the TIFFs in this process instead of starting a worker pool
    source = NikonZeissTiffProjectionSource(config_path=fake_scan, processes=1)
    ag, _ = source.get_geometries()
    assert ag.shape == (N_ANGLES, N_ROWS, N_COLS)
    assert len(source.paths) == N_ANGLES


def test_transmission_is_raw_divided_by_white_level(fake_scan):
    """Same normalisation as CIL's NikonDataReader."""
    source = NikonZeissTiffProjectionSource(config_path=fake_scan, processes=1)
    raw = source.read_chunk(0, 2)
    transmission = source.process_transmission_chunk(raw.astype(np.float32))
    np.testing.assert_allclose(transmission, RAW_VALUE / WHITE_LEVEL)
