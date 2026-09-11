"""The Bruker reader against a real acquisition on disk.

These tests run only when ``THYRA_BRUKER_TDF_DATASET`` names a real TIMS
acquisition -- the same TDF ``.d`` directory
``tests/integration/test_bruker_tdf_synthetic.py`` wants, so one variable
serves both files. With the variable unset the whole class skips.

The vendor library is bundled for Windows and Linux only; anywhere it cannot
be loaded these tests skip rather than fail. A dataset path that does not
exist, or is not a Bruker acquisition, still fails loudly -- the reader
rejects it before it ever reaches the library.
"""

import os
from pathlib import Path

import pytest

from thyra.core.registry import detect_format, get_reader_class
from thyra.readers.bruker import BrukerReader
from thyra.utils.bruker_exceptions import SDKError

REAL_DATASET = os.environ.get("THYRA_BRUKER_TDF_DATASET")


def _open(path: Path) -> BrukerReader:
    try:
        return BrukerReader(path)
    except (SDKError, OSError) as exc:  # the vendor library is not loadable here
        pytest.skip(f"Bruker library not loadable on this platform: {exc}")


@pytest.mark.skipif(
    not REAL_DATASET,
    reason="Set THYRA_BRUKER_TDF_DATASET to a TIMS .d directory to run",
)
class TestBrukerRealData:
    """Test BrukerReader with a real dataset (optional, opt-in by env var)."""

    @pytest.fixture
    def bruker_data_path(self) -> Path:
        """The acquisition named by THYRA_BRUKER_TDF_DATASET."""
        return Path(REAL_DATASET)  # type: ignore[arg-type]

    def test_bruker_reader_instantiation(self, bruker_data_path):
        """Test that BrukerReader can be instantiated with real data."""
        reader = _open(bruker_data_path)
        assert reader.data_path == bruker_data_path
        reader.close()

    def test_bruker_reader_context_manager(self, bruker_data_path):
        """Test BrukerReader works as context manager."""
        with _open(bruker_data_path) as reader:
            assert reader.data_path == bruker_data_path
            assert hasattr(reader, "close")

    def test_pixel_size_detection(self, bruker_data_path):
        """Test automatic pixel size detection."""
        with _open(bruker_data_path) as reader:
            essential_metadata = reader.get_essential_metadata()
            pixel_size = essential_metadata.pixel_size
            assert pixel_size is not None
            assert isinstance(pixel_size, tuple)
            assert len(pixel_size) == 2
            assert all(isinstance(x, (int, float)) and x > 0 for x in pixel_size)

    def test_cli_workflow_simulation(self, bruker_data_path):
        """Test the workflow that was originally failing in CLI."""
        # This simulates the exact workflow from __main__.py that was failing
        input_format = detect_format(bruker_data_path)
        assert input_format == "bruker"

        reader_class = get_reader_class(input_format)
        assert reader_class == BrukerReader

        # This line was causing the original error
        reader = _open(bruker_data_path)
        assert reader is not None

        # Test the pixel size detection that was failing
        essential_metadata = reader.get_essential_metadata()
        pixel_size = essential_metadata.pixel_size
        assert pixel_size is not None

        reader.close()

    def test_basic_functionality(self, bruker_data_path):
        """Test basic reader functionality with real data."""
        with _open(bruker_data_path) as reader:
            # Test metadata
            metadata = reader.get_comprehensive_metadata()
            assert hasattr(metadata, "essential")
            assert hasattr(metadata.essential, "source_path")
            assert str(metadata.essential.source_path) == str(bruker_data_path)

            # Test dimensions
            dimensions = reader.shape
            assert isinstance(dimensions, tuple)
            assert len(dimensions) == 3
            assert all(isinstance(x, int) and x > 0 for x in dimensions)

            # Test common mass axis (just check it doesn't crash)
            mass_axis = reader.get_common_mass_axis()
            assert hasattr(mass_axis, "__len__")
