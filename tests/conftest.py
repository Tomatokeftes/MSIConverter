"""
Common test fixtures for thyra tests.
"""

import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List

import numpy as np
import pytest
from pyimzml.ImzMLWriter import ImzMLWriter

# Root directory of the tests
TEST_DIR = Path(__file__).parent.resolve()
# Test data directory
DATA_DIR = TEST_DIR / "data"


@pytest.fixture
def thyra_logs():
    """A context manager collecting records from a named Thyra logger.

    Deliberately not ``caplog``. ``setup_logging`` sets
    ``propagate = False`` on the ``thyra`` logger and that is
    process-global: once any test in the session has invoked the CLI,
    caplog's root handler never sees another Thyra record. A caplog
    assertion on Thyra's own logging therefore passes alone and fails in
    the full suite, which is the worst way for a test to be wrong.
    Attaching a handler to the named logger sidesteps propagation, so the
    result does not depend on which tests ran first.

    Usage::

        with thyra_logs("thyra.cli", logging.WARNING) as records:
            ...
        assert [r.getMessage() for r in records] == [...]
    """

    @contextmanager
    def capture(
        logger_name: str = "thyra", level: int = logging.INFO
    ) -> Iterator[List[logging.LogRecord]]:
        collected: List[logging.LogRecord] = []

        class _Collector(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                collected.append(record)

        logger = logging.getLogger(logger_name)
        handler = _Collector(level=level)
        previous = logger.level
        logger.addHandler(handler)
        if previous > level or previous == logging.NOTSET:
            logger.setLevel(level)
        try:
            yield collected
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous)

    return capture


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_reader():
    """Create a mock MSI reader for testing converters."""
    from thyra.core.base_reader import BaseMSIReader

    class MockMSIReader(BaseMSIReader):
        def __init__(self, data_path=None, **kwargs):
            super().__init__(data_path or Path("/mock/path"), **kwargs)
            self.closed = False

        def _create_metadata_extractor(self):
            # Create a mock metadata extractor
            from thyra.core.base_extractor import MetadataExtractor
            from thyra.metadata.types import ComprehensiveMetadata, EssentialMetadata

            class MockExtractor(MetadataExtractor):
                def _extract_essential_impl(self):
                    return EssentialMetadata(
                        dimensions=(3, 3, 1),
                        coordinate_bounds=(0.0, 2.0, 0.0, 2.0),
                        mass_range=(100.0, 1000.0),
                        pixel_size=None,
                        n_spectra=9,
                        total_peaks=900,  # 9 spectra * 100 peaks each
                        estimated_memory_gb=0.001,
                        source_path="/mock/path",
                    )

                def _extract_comprehensive_impl(self):
                    return ComprehensiveMetadata(
                        essential=self._extract_essential_impl(),
                        format_specific={"format": "mock"},
                        acquisition_params={},
                        instrument_info={"instrument": "test_instrument"},
                        raw_metadata={"source": "mock"},
                    )

            return MockExtractor(None)

        def get_common_mass_axis(self):
            return np.linspace(100, 1000, 100)  # 100 mass values

        def iter_spectra(self, batch_size=None):
            mass_axis = self.get_common_mass_axis()
            for x in range(3):
                for y in range(3):
                    # Create simple synthetic spectrum
                    intensities = np.zeros_like(mass_axis)
                    # Add a few peaks
                    intensities[x * 10 + 20] = 100.0  # Peak varying by x position
                    intensities[y * 10 + 50] = 200.0  # Peak varying by y position
                    yield ((x, y, 0), mass_axis, intensities)

        def close(self):
            self.closed = True

    return MockMSIReader()


@pytest.fixture
def create_minimal_imzml(temp_dir):
    """
    Create a minimal imzML file for testing.
    Returns a tuple of (imzml_path, ibd_path, mzs, intensities)
    """
    imzml_path = temp_dir / "minimal.imzML"
    ibd_path = temp_dir / "minimal.ibd"

    # Create small sample data
    coordinates = [(1, 1, 1), (1, 2, 1), (2, 1, 1), (2, 2, 1)]  # 2x2 grid
    mzs = np.linspace(100, 1000, 50)  # 50 m/z values

    # Create different intensities for each pixel
    all_intensities = []
    for i, (x, y, z) in enumerate(coordinates):
        intensities = np.zeros_like(mzs)
        # Create a few peaks with position-dependent intensity
        intensities[10] = 100.0 * x  # Peak intensity depends on x
        intensities[30] = 150.0 * y  # Peak intensity depends on y
        all_intensities.append(intensities)

    # Write imzML file
    with ImzMLWriter(str(imzml_path), mode="processed") as writer:
        for i, (x, y, z) in enumerate(coordinates):
            writer.addSpectrum(mzs, all_intensities[i], (x, y, z))

    return imzml_path, ibd_path, mzs, all_intensities


def _write_square_imzml(path, origin, side):
    """Write a ``side`` x ``side`` imzML whose lowest coordinate is ``origin``.

    Every pixel carries one peak whose intensity encodes its position, so a
    row that moved can be told from a row that went missing.

    Args:
        path: Where to write the ``.imzML`` (the ``.ibd`` goes beside it).
        origin: The smallest x and y written to the file. ``1`` is what the
            specification says; ``0`` is what some exporters write.
        side: Pixels per side.

    Returns:
        ``(path, mzs, expected)`` where ``expected`` maps the 0-based
        ``(x, y)`` the store should hold to that pixel's intensity.
    """
    mzs = np.linspace(100.0, 200.0, 5)
    expected = {}

    with ImzMLWriter(str(path), mode="processed") as writer:
        for row in range(side):
            for col in range(side):
                intensity = np.zeros_like(mzs)
                # Distinct per pixel, and never zero: a pixel whose only
                # peak is zero gets no row at all, which would hide the
                # very loss this fixture exists to detect.
                intensity[2] = 100.0 + 10.0 * row + col
                writer.addSpectrum(mzs, intensity, (origin + col, origin + row, 1))
                expected[(col, row)] = intensity[2]

    return path, mzs, expected


@pytest.fixture
def zero_based_imzml(temp_dir):
    """A 3x3 imzML numbered from 0 -- the shape of issue #244.

    The imzML specification numbers pixels from 1, and the reader used to
    subtract a constant 1. On a file written 0-based that produced
    ``x = -1`` for the first column, which the converter's grid guard
    dropped: the store came out 4 rows of a 3x3 acquisition, with a
    warning naming a 2x2 grid the file never declared, and exit 0.
    """
    return _write_square_imzml(temp_dir / "zero_based.imzML", origin=0, side=3)


@pytest.fixture
def one_based_imzml(temp_dir):
    """The same 3x3 acquisition written the way the specification says."""
    return _write_square_imzml(temp_dir / "one_based.imzML", origin=1, side=3)


@pytest.fixture
def cropped_one_based_imzml(temp_dir):
    """A 1-based acquisition whose leftmost column is x = 5, not x = 1.

    The reason x and y do not simply rebase on their observed minimum the
    way z does: this file is a legitimately cropped region of the slide,
    and sliding it to the origin would change its grid width, every
    ``obs`` coordinate and its pixel footprint -- on a file that converts
    correctly today.
    """
    return _write_square_imzml(temp_dir / "cropped.imzML", origin=5, side=3)


@pytest.fixture
def mock_waters_data(temp_dir):
    """Create a mock Waters .raw directory structure.

    Creates the minimal structure needed for WatersReader validation:
    a directory with a _FUNC001.DAT file.
    """
    raw_dir = temp_dir / "mock.raw"
    raw_dir.mkdir()
    (raw_dir / "_FUNC001.DAT").write_bytes(b"\x00" * 64)
    return raw_dir
