"""
Integration tests using real Bruker dataset.

These tests ensure that the BrukerReader works correctly with actual data,
particularly for constructor calls and pixel size detection.

NOTE: These tests are optional and will skip if the real dataset is not available.
They are primarily for local development and validation.
"""

import os
from pathlib import Path

import pytest

from msiconvert.core.registry import detect_format, get_reader_class
from msiconvert.readers.bruker import BrukerReader


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Real Bruker dataset not available in CI environment",
)
class TestBrukerRealData:
    """Test BrukerReader with real dataset (optional, skipped in CI)."""

    @pytest.fixture
    def bruker_data_path(self):
        """Path to real Bruker test dataset."""
        # Try multiple possible locations for the dataset
        possible_paths = [
            Path(__file__).parent.parent.parent
            / "data"
            / "20231109_PEA_NEDC.d",  # Local data folder
            Path(
                "C:/Users/P70078823/Desktop/MSIConverter/data/20231109_PEA_NEDC.d"
            ),  # Absolute path
        ]

        for data_path in possible_paths:
            if data_path.exists():
                return data_path

        pytest.skip(
            f"Bruker test dataset not found. Tried: {[str(p) for p in possible_paths]}"
        )
        return None

    def test_bruker_reader_instantiation(self, bruker_data_path):
        """Test that BrukerReader can be instantiated with real data."""
        reader = BrukerReader(bruker_data_path)
        assert reader.data_path == bruker_data_path
        reader.close()

    def test_bruker_reader_context_manager(self, bruker_data_path):
        """Test BrukerReader works as context manager."""
        with BrukerReader(bruker_data_path) as reader:
            assert reader.data_path == bruker_data_path
            assert hasattr(reader, "close")

    def test_pixel_size_detection(self, bruker_data_path):
        """Test automatic pixel size detection."""
        with BrukerReader(bruker_data_path) as reader:
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
        reader = reader_class(bruker_data_path)
        assert reader is not None

        # Test the pixel size detection that was failing
        essential_metadata = reader.get_essential_metadata()
        pixel_size = essential_metadata.pixel_size
        assert pixel_size is not None

        reader.close()

    def test_basic_functionality(self, bruker_data_path):
        """Test basic reader functionality with real data."""
        with BrukerReader(bruker_data_path) as reader:
            # Test metadata
            metadata = reader.get_comprehensive_metadata()
            assert hasattr(metadata, 'essential')
            assert hasattr(metadata.essential, 'source_path')
            assert str(metadata.essential.source_path) == str(bruker_data_path)

            # Test dimensions
            dimensions = reader.shape
            assert isinstance(dimensions, tuple)
            assert len(dimensions) == 3
            assert all(isinstance(x, int) and x > 0 for x in dimensions)

            # Test common mass axis (just check it doesn't crash)
            mass_axis = reader.get_common_mass_axis()
            assert hasattr(mass_axis, "__len__")
