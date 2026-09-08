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


@pytest.fixture
def mock_bruker_data(temp_dir):
    """
    Create a mock Bruker data directory structure.
    This is just for structure testing - won't contain real data.
    """
    bruker_dir = temp_dir / "mock.d"
    bruker_dir.mkdir(parents=True)

    # Create minimal analysis.tsf file
    with open(bruker_dir / "analysis.tsf", "w") as f:
        f.write("Mock Bruker TSF file for testing")

    # Create a fake SQLite DB file
    with open(bruker_dir / "analysis.sqlite", "w") as f:
        f.write("Mock SQLite DB file")

    return bruker_dir


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
