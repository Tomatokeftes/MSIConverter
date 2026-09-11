"""
Tests for the base reader and converter classes.
"""

from pathlib import Path

import numpy as np
import pytest

from thyra.core.base_converter import BaseMSIConverter
from thyra.core.base_reader import BaseMSIReader


class TestBaseMSIReader:
    """Test the base MSI reader abstract class."""

    def test_abstract_methods(self):
        """Test that abstract methods are properly defined."""
        # Subclassing without implementing methods should fail
        with pytest.raises(TypeError):

            class IncompleteReader(BaseMSIReader):
                pass

            IncompleteReader(Path("/test/path"))

    def test_implementation(self):
        """Test that implementing all required methods works."""

        # Define a minimal implementation
        class MinimalReader(BaseMSIReader):
            def _create_metadata_extractor(self):
                from thyra.core.base_extractor import MetadataExtractor
                from thyra.metadata.types import (
                    ComprehensiveMetadata,
                    EssentialMetadata,
                )

                class TestExtractor(MetadataExtractor):
                    def _extract_essential_impl(self):
                        return EssentialMetadata(
                            dimensions=(1, 1, 1),
                            coordinate_bounds=(0.0, 0.0, 0.0, 0.0),
                            mass_range=(100.0, 200.0),
                            pixel_size=None,
                            n_spectra=1,
                            total_peaks=100,
                            estimated_memory_gb=0.001,
                            source_path="/test/path",
                        )

                    def _extract_comprehensive_impl(self):
                        return ComprehensiveMetadata(
                            essential=self._extract_essential_impl(),
                            format_specific={},
                            acquisition_params={},
                            instrument_info={},
                            raw_metadata={"test": "metadata"},
                        )

                return TestExtractor(None)

            def get_common_mass_axis(self):
                return np.array([100.0, 200.0])

            def iter_spectra(self):
                yield ((0, 0, 0), np.array([100.0]), np.array([1.0]))

            def close(self):
                pass

        # Should instantiate without errors
        reader = MinimalReader(Path("/test/path"))

        # Check method functionality
        essential = reader.get_essential_metadata()
        assert essential.dimensions == (1, 1, 1)
        assert essential.mass_range == (100.0, 200.0)
        assert essential.source_path == "/test/path"
        np.testing.assert_array_equal(
            reader.get_common_mass_axis(), np.array([100.0, 200.0])
        )

        # Test iterator
        coords, mzs, intensities = next(reader.iter_spectra())
        assert coords == (0, 0, 0)
        np.testing.assert_array_equal(mzs, np.array([100.0]))
        np.testing.assert_array_equal(intensities, np.array([1.0]))


class TestBaseMSIConverter:
    """Test the base MSI converter abstract class."""

    def test_abstract_methods(self):
        """Test that abstract methods are properly defined."""

        # Mock reader for testing
        class MockReader(BaseMSIReader):
            def _create_metadata_extractor(self):
                from thyra.core.base_extractor import MetadataExtractor
                from thyra.metadata.types import (
                    ComprehensiveMetadata,
                    EssentialMetadata,
                )

                class TestExtractor(MetadataExtractor):
                    def _extract_essential_impl(self):
                        return EssentialMetadata(
                            dimensions=(1, 1, 1),
                            coordinate_bounds=(0.0, 0.0, 0.0, 0.0),
                            mass_range=(0.0, 0.0),
                            pixel_size=None,
                            n_spectra=1,
                            total_peaks=0,
                            estimated_memory_gb=0.001,
                            source_path="/test/path",
                        )

                    def _extract_comprehensive_impl(self):
                        return ComprehensiveMetadata(
                            essential=self._extract_essential_impl(),
                            format_specific={},
                            acquisition_params={},
                            instrument_info={},
                            raw_metadata={},
                        )

                return TestExtractor(None)

            def get_common_mass_axis(self):
                return np.array([])

            def iter_spectra(self, batch_size=None):
                yield ((0, 0, 0), np.array([]), np.array([]))

            def close(self):
                pass

        # Subclassing without implementing methods should fail
        with pytest.raises(TypeError):

            class IncompleteConverter(BaseMSIConverter):
                pass

            IncompleteConverter(MockReader(Path("/test/path")), "test.out")

    def test_utility_methods(self):
        """Test the utility methods in the base converter."""
        # Create test classes and converter
        converter = self._create_test_converter()

        # Initialize for testing utility methods
        converter._initialize_conversion()

        # Run all utility method tests
        self._test_coordinate_conversion(converter)
        self._test_mass_mapping(converter)
        self._test_dataframe_creation(converter)

    def _create_test_converter(self):
        """Create a converter with test classes for utility method testing."""

        # Create a minimal implementation for testing
        class MinimalConverter(BaseMSIConverter):
            def _create_data_structures(self):
                return {}

            def _save_output(self, data_structures):
                return True

        # Create a mock reader
        mock_reader = self._create_mock_reader()

        return MinimalConverter(mock_reader, Path("test.out"))

    def _create_mock_reader(self):
        """Create a mock reader for testing."""

        class MockReader(BaseMSIReader):
            def _create_metadata_extractor(self):
                from thyra.core.base_extractor import MetadataExtractor
                from thyra.metadata.types import (
                    ComprehensiveMetadata,
                    EssentialMetadata,
                )

                class TestExtractor(MetadataExtractor):
                    def _extract_essential_impl(self):
                        return EssentialMetadata(
                            dimensions=(2, 2, 1),
                            coordinate_bounds=(0.0, 1.0, 0.0, 1.0),
                            mass_range=(100.0, 300.0),
                            pixel_size=None,
                            n_spectra=4,
                            total_peaks=8,
                            estimated_memory_gb=0.001,
                            source_path="/test/path",
                        )

                    def _extract_comprehensive_impl(self):
                        return ComprehensiveMetadata(
                            essential=self._extract_essential_impl(),
                            format_specific={},
                            acquisition_params={},
                            instrument_info={},
                            raw_metadata={"source": "test"},
                        )

                return TestExtractor(None)

            def get_common_mass_axis(self):
                return np.array([100.0, 200.0, 300.0])

            def iter_spectra(self, batch_size=None):
                for x in range(2):
                    for y in range(2):
                        yield (
                            (x, y, 0),
                            np.array([100.0, 300.0]),
                            np.array([1.0, 2.0]),
                        )

            def close(self):
                pass

        return MockReader(Path("/test/path"))

    def _test_coordinate_conversion(self, converter):
        """Test coordinate to index conversion."""
        assert converter._get_pixel_index(0, 0, 0) == 0
        assert converter._get_pixel_index(1, 0, 0) == 1
        assert converter._get_pixel_index(0, 1, 0) == 2
        assert converter._get_pixel_index(1, 1, 0) == 3

    def _test_mass_mapping(self, converter):
        """Test mass mapping functionality."""
        mz_indices = converter._map_mass_to_indices(np.array([100.0, 300.0]))
        np.testing.assert_array_equal(mz_indices, np.array([0, 2]))

    def _test_dataframe_creation(self, converter):
        """Test dataframe creation methods."""
        # Test coordinates dataframe
        coords_df = converter._create_coordinates_dataframe()
        assert len(coords_df) == 4
        assert "x" in coords_df.columns
        assert "y" in coords_df.columns
        assert "z" in coords_df.columns

        # Test mass dataframe
        mass_df = converter._create_mass_dataframe()
        assert len(mass_df) == 3
        assert "mz" in mass_df.columns
