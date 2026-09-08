# tests/unit/converters/test_streaming_converter.py
"""Tests for the streaming SpatialData converter."""

import tempfile
from pathlib import Path
from typing import Generator, Optional, Tuple
from unittest.mock import MagicMock

import anndata
import numpy as np
import pytest

from thyra.converters.spatialdata.streaming_converter import (
    SPATIALDATA_AVAILABLE,
    StreamingSpatialDataConverter,
)


class MockEssentialMetadata:
    """Mock essential metadata for testing."""

    def __init__(
        self,
        dimensions: Tuple[int, int, int] = (10, 10, 1),
        n_spectra: int = 100,
        mass_range: Tuple[float, float] = (100.0, 1000.0),
        pixel_size: Optional[Tuple[float, float]] = (20.0, 20.0),
        peaks_per_spectrum: int = 500,
    ):
        self.dimensions = dimensions
        self.n_spectra = n_spectra
        self.mass_range = mass_range
        self.pixel_size = pixel_size
        self.coordinate_bounds = (0, dimensions[0], 0, dimensions[1])
        self.estimated_memory_gb = 0.1
        self.source_path = Path("/mock/path.imzML")
        self.total_peaks = n_spectra * peaks_per_spectrum
        self.is_3d = dimensions[2] > 1  # 3D if z > 1
        self.has_pixel_size = pixel_size is not None
        # Per-pixel peak counts for streaming converter
        n_pixels = dimensions[0] * dimensions[1] * dimensions[2]
        self.peak_counts_per_pixel = np.full(
            n_pixels, peaks_per_spectrum, dtype=np.int32
        )


class MockMSIReader:
    """Mock MSI reader for testing the streaming converter."""

    def __init__(
        self,
        dimensions: Tuple[int, int, int] = (10, 10, 1),
        peaks_per_spectrum: int = 500,
        mass_range: Tuple[float, float] = (100.0, 1000.0),
    ):
        self.dimensions = dimensions
        self.peaks_per_spectrum = peaks_per_spectrum
        self.mass_range = mass_range
        self.n_spectra = dimensions[0] * dimensions[1] * dimensions[2]
        self.data_path = Path("/mock/path.imzML")
        self.has_shared_mass_axis = False  # Mock data is processed mode

        # Generate consistent random data
        np.random.seed(42)

    def get_essential_metadata(self) -> MockEssentialMetadata:
        """Return mock essential metadata."""
        return MockEssentialMetadata(
            dimensions=self.dimensions,
            n_spectra=self.n_spectra,
            mass_range=self.mass_range,
            peaks_per_spectrum=self.peaks_per_spectrum,
        )

    def get_comprehensive_metadata(self) -> MagicMock:
        """Return mock comprehensive metadata."""
        mock = MagicMock()
        mock.format_specific = {}
        mock.acquisition_params = {}
        mock.instrument_info = {}
        mock.raw_metadata = {}
        mock.essential = self.get_essential_metadata()
        return mock

    def iter_spectra(
        self, batch_size: Optional[int] = None
    ) -> Generator[Tuple[Tuple[int, int, int], np.ndarray, np.ndarray], None, None]:
        """Yield mock spectra for all pixels.

        Peaks are drawn from the very axis :meth:`get_common_mass_axis`
        reports, not merely from the same mass RANGE. Without resampling,
        ``BaseMSIConverter._map_mass_to_indices`` requires every peak to
        land within 1e-6 Da of a common-axis point and raises for any
        that do not, so m/z values that just fall inside the range would
        abort the conversion. A mock whose two methods describe
        different mass axes is not a stand-in for any real reader.
        """
        n_x, n_y, n_z = self.dimensions
        common_mass_axis = self.get_common_mass_axis()

        for z in range(n_z):
            for y in range(n_y):
                for x in range(n_x):
                    # Sample distinct axis positions: a duplicate index would
                    # write the same (pixel, m/z) cell twice.
                    peak_indices = np.sort(
                        np.random.choice(
                            common_mass_axis.size,
                            self.peaks_per_spectrum,
                            replace=False,
                        )
                    )
                    mzs = common_mass_axis[peak_indices]
                    # Generate random intensities
                    intensities = np.random.exponential(1000, self.peaks_per_spectrum)

                    yield (x, y, z), mzs, intensities

    def get_optical_image_paths(self):
        """Return empty list (no optical images)."""
        return []

    def get_common_mass_axis(self) -> np.ndarray:
        """Return a common mass axis for testing."""
        min_mz, max_mz = self.mass_range
        # Create a common mass axis with reasonable resolution
        return np.linspace(min_mz, max_mz, 10000)

    def get_peak_counts_per_pixel(self) -> np.ndarray:
        """Return per-pixel peak counts for CSR indptr construction."""
        n_x, n_y, n_z = self.dimensions
        n_pixels = n_x * n_y * n_z
        # All pixels have the same number of peaks
        return np.full(n_pixels, self.peaks_per_spectrum, dtype=np.int32)

    def get_region_map(self):
        """Return None (single-region mock)."""
        return None

    def get_region_info(self):
        """Return None (single-region mock)."""
        return None

    def close(self):
        """Mock close method."""
        pass

    def reset(self):
        """Reset reader state for re-iteration."""
        # Reset the random seed for consistent data across iterations
        np.random.seed(42)


@pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)
def test_resampling_plan_uses_real_bins_for_width_based_config():
    """Regression for #87: when ``target_bins`` is None, the bin count must be
    resolved from ``width_at_mz`` rather than a hardcoded 10,000 placeholder.

    That fallback used to pick the wrong streaming method; there is one
    route now, but the resolved count is what the memory guard reads
    before the axis is built (issue #251), so an axis scored at 10,000
    bins instead of 180,000 would be waved through by a guard sized for
    the wrong number.
    """
    reader = MockMSIReader(
        dimensions=(50, 50, 1),
        peaks_per_spectrum=200,
        mass_range=(100.0, 1000.0),
    )
    # Width-based config: bins NOT specified, only width_at_mz.
    # With CONSTANT axis (mock falls through to DefaultDetector) and
    # 5 mDa bin width over 100-1000 m/z, the resolved axis is 180,000 bins.
    config = {
        "method": "tic_preserving",
        "target_bins": None,
        "width_at_mz": 0.005,
        "reference_mz": 1000.0,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = StreamingSpatialDataConverter(
            reader=reader,
            output_path=Path(tmpdir) / "out.zarr",
            use_csc="auto",
            resampling_config=config,
        )
        _, _, _, target_bins = converter._resolve_resampling_plan()

    assert target_bins == pytest.approx(180_000, rel=0.01), (
        f"Expected ~180,000 bins from the 5 mDa width, got {target_bins:,}. "
        "Likely regression of #87 (10k hardcoded fallback)."
    )


@pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)
def test_converter_initialization():
    """Test that the converter initializes correctly."""
    reader = MockMSIReader()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_output.zarr"

        converter = StreamingSpatialDataConverter(
            reader=reader,
            output_path=output_path,
            dataset_id="test_dataset",
            pixel_size_um=20.0,
            use_csc=True,
        )

        assert converter.dataset_id == "test_dataset"


@pytest.mark.integration
@pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)
class TestStreamingSpatialDataConverter:
    """Tests for StreamingSpatialDataConverter."""

    def test_basic_conversion(self):
        """Test basic conversion with small synthetic dataset."""
        reader = MockMSIReader(
            dimensions=(5, 5, 1),  # 25 pixels
            peaks_per_spectrum=100,
            mass_range=(200.0, 800.0),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.zarr"

            converter = StreamingSpatialDataConverter(
                reader=reader,
                output_path=output_path,
                dataset_id="test_dataset",
                pixel_size_um=20.0,
                use_csc=True,  # Use CSC format
            )

            # Run conversion
            success = converter.convert()
            assert success, "Conversion should succeed"

            # Check output exists
            assert output_path.exists(), "Output path should exist"

            # Try to load the output
            from spatialdata import SpatialData

            sdata = SpatialData.read(str(output_path))

            # Verify structure - CSC mode only creates tables
            assert len(sdata.tables) > 0, "Should have tables"

            # Check table content
            table_key = list(sdata.tables.keys())[0]
            adata = sdata.tables[table_key]

            assert adata.shape[0] == 25, "Should have 25 pixels (5x5)"
            # Mass bins come from the reader's common mass axis (10000)
            assert adata.shape[1] == 10000, "Should have 10000 mass bins"

    def test_chunked_processing(self):
        """Test that chunked processing produces correct output."""
        reader = MockMSIReader(
            dimensions=(10, 10, 1),  # 100 pixels
            peaks_per_spectrum=200,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_chunked.zarr"

            converter = StreamingSpatialDataConverter(
                reader=reader,
                output_path=output_path,
                dataset_id="chunked_test",
                use_csc=True,
            )

            success = converter.convert()
            assert success

            # Verify all pixels are present
            from spatialdata import SpatialData

            sdata = SpatialData.read(str(output_path))
            table_key = list(sdata.tables.keys())[0]
            adata = sdata.tables[table_key]

            assert adata.shape[0] == 100, "All 100 pixels should be present"

    def test_sparse_matrix_format(self):
        """Test that CSC mode creates correct sparse matrix format."""
        reader = MockMSIReader(
            dimensions=(3, 3, 1),  # 9 pixels
            peaks_per_spectrum=50,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_csc.zarr"

            converter = StreamingSpatialDataConverter(
                reader=reader,
                output_path=output_path,
                dataset_id="csc_test",
                use_csc=True,
            )

            success = converter.convert()
            assert success

            from spatialdata import SpatialData

            sdata = SpatialData.read(str(output_path))

            # Check table exists and has correct format
            table_key = list(sdata.tables.keys())[0]
            adata = sdata.tables[table_key]

            # Check it's a sparse matrix
            from scipy import sparse

            assert sparse.issparse(adata.X), "Matrix should be sparse"
            assert isinstance(adata.X, sparse.csc_matrix), "Should be CSC format"

            # Check shape
            assert adata.shape[0] == 9, "Should have 9 pixels (3x3)"

    def test_mass_axis_preservation(self):
        """Test that common mass axis from reader is preserved."""
        reader = MockMSIReader(
            dimensions=(2, 2, 1),
            peaks_per_spectrum=100,
            mass_range=(100.0, 500.0),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_mass_axis.zarr"

            converter = StreamingSpatialDataConverter(
                reader=reader,
                output_path=output_path,
                dataset_id="mass_axis_test",
                use_csc=True,
            )

            success = converter.convert()
            assert success

            from spatialdata import SpatialData

            sdata = SpatialData.read(str(output_path))
            table_key = list(sdata.tables.keys())[0]
            adata = sdata.tables[table_key]

            # Check mass axis matches reader's common mass axis
            expected_bins = len(reader.get_common_mass_axis())
            assert adata.shape[1] == expected_bins

            # Check m/z values in var
            mz_values = adata.var["mz"].values
            assert len(mz_values) == expected_bins
            assert mz_values[0] >= 100.0
            assert mz_values[-1] <= 1000.0  # Reader's default max

    def test_tic_image_and_shapes_created(self):
        """Test that TIC image and pixel shapes are created in CSC mode."""
        reader = MockMSIReader(
            dimensions=(4, 4, 1),  # 16 pixels
            peaks_per_spectrum=50,
            mass_range=(200.0, 600.0),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_tic_shapes.zarr"

            converter = StreamingSpatialDataConverter(
                reader=reader,
                output_path=output_path,
                dataset_id="tic_shapes_test",
                pixel_size_um=25.0,
                use_csc=True,  # Force PCS streaming path
            )

            success = converter.convert()
            assert success, "Conversion should succeed"

            from spatialdata import SpatialData

            sdata = SpatialData.read(str(output_path))

            # Verify TIC image exists
            assert len(sdata.images) > 0, "Should have at least one image (TIC)"
            tic_keys = [k for k in sdata.images.keys() if "tic" in k.lower()]
            assert len(tic_keys) > 0, "Should have TIC image"

            # Check TIC image dimensions
            tic_key = tic_keys[0]
            tic_image = sdata.images[tic_key]
            # TIC should be (c, y, x) with c=1
            assert tic_image.shape[0] == 1, "TIC should have 1 channel"
            assert tic_image.shape[1] == 4, "TIC height should match n_y"
            assert tic_image.shape[2] == 4, "TIC width should match n_x"

            # Verify shapes exist
            assert len(sdata.shapes) > 0, "Should have shapes"
            shape_key = list(sdata.shapes.keys())[0]
            shapes = sdata.shapes[shape_key]

            # Should have one shape per pixel
            assert len(shapes) == 16, "Should have 16 shapes (4x4 grid)"

            # Check shapes are valid polygons
            from shapely.geometry import Polygon

            for geom in shapes.geometry:
                assert isinstance(geom, Polygon), "Each shape should be a polygon"
                assert geom.is_valid, "Shape should be valid"
                # Each shape should be a square pixel
                # MockReader's default pixel_size is (20.0, 20.0) which overrides
                # the specified pixel_size_um due to auto-detection
                area = geom.area
                expected_area = 20.0 * 20.0  # mock reader's pixel_size^2
                assert (
                    abs(area - expected_area) < 0.01
                ), "Shape area should match pixel size"


@pytest.mark.integration
@pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)
def test_streaming_converter_memory_efficiency():
    """Test that streaming converter uses bounded memory."""
    # This is a basic check - actual memory profiling would need
    # more sophisticated tools

    reader = MockMSIReader(
        dimensions=(20, 20, 1),  # 400 pixels
        peaks_per_spectrum=500,
    )

    # Resampling config is required for memory-efficient conversion
    # Without it, raw mass axis collection can explode memory
    resampling_config = {
        "method": "tic_preserving",
        "target_bins": 1000,
        "axis_type": "constant",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_memory.zarr"

        converter = StreamingSpatialDataConverter(
            reader=reader,
            output_path=output_path,
            use_csc=True,
            resampling_config=resampling_config,
        )

        # Should complete without memory issues
        success = converter.convert()
        assert success


@pytest.mark.integration
@pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)
def test_single_pixel_dataset():
    """Test conversion with a single pixel (1x1) dataset."""
    reader = MockMSIReader(
        dimensions=(1, 1, 1),  # Single pixel
        peaks_per_spectrum=50,
        mass_range=(200.0, 600.0),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_single_pixel.zarr"

        converter = StreamingSpatialDataConverter(
            reader=reader,
            output_path=output_path,
            dataset_id="single_pixel_test",
            use_csc=True,
        )

        success = converter.convert()
        assert success, "Single pixel conversion should succeed"

        from spatialdata import SpatialData

        sdata = SpatialData.read(str(output_path))
        table_key = list(sdata.tables.keys())[0]
        adata = sdata.tables[table_key]

        assert adata.shape[0] == 1, "Should have exactly 1 pixel"
        assert len(sdata.shapes) > 0, "Should have shapes"
        assert len(list(sdata.shapes.values())[0]) == 1, "Should have 1 shape"


@pytest.mark.integration
@pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)
def test_rectangular_grid():
    """Test conversion with non-square (rectangular) grid."""
    reader = MockMSIReader(
        dimensions=(10, 5, 1),  # 10x5 = 50 pixels, rectangular
        peaks_per_spectrum=100,
        mass_range=(100.0, 500.0),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_rectangular.zarr"

        converter = StreamingSpatialDataConverter(
            reader=reader,
            output_path=output_path,
            dataset_id="rectangular_test",
            use_csc=True,
        )

        success = converter.convert()
        assert success, "Rectangular grid conversion should succeed"

        from spatialdata import SpatialData

        sdata = SpatialData.read(str(output_path))
        table_key = list(sdata.tables.keys())[0]
        adata = sdata.tables[table_key]

        assert adata.shape[0] == 50, "Should have 50 pixels (10x5)"

        # Check TIC image has correct shape
        tic_keys = [k for k in sdata.images.keys() if "tic" in k.lower()]
        if tic_keys:
            tic_image = sdata.images[tic_keys[0]]
            # TIC shape should be (c, y, x) = (1, 5, 10)
            assert tic_image.shape[1] == 5, "TIC height should be 5"
            assert tic_image.shape[2] == 10, "TIC width should be 10"


@pytest.mark.integration
@pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)
def test_auto_use_csc_mode_with_resampling():
    """``use_csc="auto"`` converts, and the size it reports is the counted one."""
    reader = MockMSIReader(
        dimensions=(50, 50, 1),  # 2500 pixels
        peaks_per_spectrum=200,
    )

    resampling_config = {
        "method": "tic_preserving",
        "target_bins": 5000,  # Custom target bins
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_auto_resample.zarr"

        converter = StreamingSpatialDataConverter(
            reader=reader,
            output_path=output_path,
            dataset_id="auto_resample_test",
            use_csc="auto",
            resampling_config=resampling_config,
        )

        assert converter.convert() is True

        # The route converts on the resolved axis. What the log reports
        # about its size is the counted non-zeros rather than the dense
        # product of pixels and bins (issue #254, pinned in
        # test_routing_estimate.py).
        table = anndata.read_zarr(output_path / "tables" / "auto_resample_test_z0")
        assert table.shape == (2500, 5000)
        assert int(table.X.nnz) > 0


class MockMSIReaderWithOptical(MockMSIReader):
    """Mock reader that provides optical images."""

    def __init__(self, optical_images=None, **kwargs):
        super().__init__(**kwargs)
        self._optical_images = optical_images or []

    def get_optical_image_paths(self):
        """Return mock optical image paths."""
        return self._optical_images


@pytest.mark.integration
@pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)
def test_optical_image_loading():
    """Test that optical images are loaded when include_optical=True."""
    import tifffile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock TIFF image
        optical_path = Path(tmpdir) / "optical_test.tif"
        mock_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        tifffile.imwrite(str(optical_path), mock_image)

        reader = MockMSIReaderWithOptical(
            dimensions=(3, 3, 1),
            peaks_per_spectrum=50,
            optical_images=[optical_path],
        )

        output_path = Path(tmpdir) / "test_optical.zarr"

        converter = StreamingSpatialDataConverter(
            reader=reader,
            output_path=output_path,
            dataset_id="optical_test",
            use_csc=True,
            include_optical=True,
        )

        success = converter.convert()
        assert success, "Conversion with optical images should succeed"

        from spatialdata import SpatialData

        sdata = SpatialData.read(str(output_path))

        # Check for optical image
        optical_keys = [k for k in sdata.images.keys() if "optical" in k.lower()]
        assert len(optical_keys) > 0, "Should have optical image"


@pytest.mark.integration
@pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)
def test_optical_image_rgb():
    """Test loading RGB optical images (3D TIFF)."""
    import tifffile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock RGB TIFF image
        optical_path = Path(tmpdir) / "optical_rgb.tif"
        mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        tifffile.imwrite(str(optical_path), mock_image)

        reader = MockMSIReaderWithOptical(
            dimensions=(3, 3, 1),
            peaks_per_spectrum=50,
            optical_images=[optical_path],
        )

        output_path = Path(tmpdir) / "test_optical_rgb.zarr"

        converter = StreamingSpatialDataConverter(
            reader=reader,
            output_path=output_path,
            dataset_id="optical_rgb_test",
            use_csc=True,
            include_optical=True,
        )

        success = converter.convert()
        assert success, "Conversion with RGB optical images should succeed"

        from spatialdata import SpatialData

        sdata = SpatialData.read(str(output_path))

        # Check optical image has 3 channels
        optical_keys = [k for k in sdata.images.keys() if "optical" in k.lower()]
        assert len(optical_keys) > 0
        optical_img = sdata.images[optical_keys[0]]
        assert optical_img.shape[0] == 3, "RGB image should have 3 channels"


@pytest.mark.integration
@pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)
def test_no_optical_images_when_disabled():
    """Test that optical images are not loaded when include_optical=False."""
    import tifffile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock TIFF image
        tiff_path = Path(tmpdir) / "microscope_image.tif"
        mock_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        tifffile.imwrite(str(tiff_path), mock_image)

        reader = MockMSIReaderWithOptical(
            dimensions=(3, 3, 1),
            peaks_per_spectrum=50,
            optical_images=[tiff_path],
        )

        output_path = Path(tmpdir) / "test_disabled_img.zarr"

        converter = StreamingSpatialDataConverter(
            reader=reader,
            output_path=output_path,
            dataset_id="disabled_img_test",
            use_csc=True,
            include_optical=False,  # Explicitly disable
        )

        success = converter.convert()
        assert success

        from spatialdata import SpatialData

        sdata = SpatialData.read(str(output_path))

        # Should not have optical images (only TIC)
        optical_keys = [k for k in sdata.images.keys() if "optical" in k.lower()]
        assert len(optical_keys) == 0, "Should not have optical images"
        # Verify TIC image still exists
        tic_keys = [k for k in sdata.images.keys() if "tic" in k.lower()]
        assert len(tic_keys) > 0, "Should still have TIC image"


@pytest.mark.integration
@pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)
def test_larger_chunk_write():
    """Test that larger datasets trigger chunk write logic."""
    # Create dataset large enough to trigger multiple chunk writes
    reader = MockMSIReader(
        dimensions=(15, 15, 1),  # 225 pixels
        peaks_per_spectrum=100,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_large_chunks.zarr"

        converter = StreamingSpatialDataConverter(
            reader=reader,
            output_path=output_path,
            dataset_id="large_chunk_test",
            use_csc=True,
        )

        success = converter.convert()
        assert success, "Large chunk conversion should succeed"

        from spatialdata import SpatialData

        sdata = SpatialData.read(str(output_path))
        table_key = list(sdata.tables.keys())[0]
        adata = sdata.tables[table_key]

        assert adata.shape[0] == 225, "All 225 pixels should be present"


class MockMSIReaderWithControlledIntensities(MockMSIReader):
    """Mock reader that generates controlled intensities for threshold testing.

    Uses the same m/z values from get_common_mass_axis() to ensure consistency.
    Supports intensity_threshold filtering at the reader level (like real readers).
    """

    def __init__(self, *args, intensity_threshold=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-compute fixed m/z values from common mass axis
        self._fixed_mzs = self.get_common_mass_axis()[: self.peaks_per_spectrum]
        self._intensity_threshold = intensity_threshold

    def iter_spectra(self, batch_size=None):
        """Generate spectra with alternating low/high intensities.

        Applies intensity_threshold filtering if set (simulating reader-level filtering).
        """
        n_x, n_y, n_z = self.dimensions

        for z in range(n_z):
            for y in range(n_y):
                for x in range(n_x):
                    # Create intensities where ~50% are below threshold (0.5)
                    # and ~50% are above (10.0)
                    intensities = np.where(
                        np.arange(self.peaks_per_spectrum) % 2 == 0,
                        0.5,  # Below threshold of 1.0
                        10.0,  # Above threshold
                    )
                    mzs = self._fixed_mzs.copy()
                    intensities = intensities.astype(np.float64)

                    # Apply intensity threshold filtering (like real readers do)
                    if self._intensity_threshold is not None:
                        mask = intensities >= self._intensity_threshold
                        mzs = mzs[mask]
                        intensities = intensities[mask]

                    yield (x, y, z), mzs, intensities


@pytest.mark.integration
@pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)
def test_intensity_threshold_filtering():
    """Test that intensity_threshold filters out low intensity values.

    Note: As of v1.13.0, intensity_threshold is handled at the reader level,
    not the converter level. This test verifies that reader-level filtering works.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # First, convert WITHOUT threshold (threshold passed to reader)
        reader_no_thresh = MockMSIReaderWithControlledIntensities(
            dimensions=(3, 3, 1),  # 9 pixels
            peaks_per_spectrum=100,
            mass_range=(100.0, 500.0),
            intensity_threshold=None,  # No filtering at reader level
        )
        output_no_thresh = Path(tmpdir) / "no_threshold.zarr"
        converter_no_thresh = StreamingSpatialDataConverter(
            reader=reader_no_thresh,
            output_path=output_no_thresh,
            dataset_id="no_threshold_test",
            use_csc=True,
        )
        success = converter_no_thresh.convert()
        assert success, "Conversion without threshold should succeed"

        # Then convert WITH threshold (threshold passed to reader)
        reader_with_thresh = MockMSIReaderWithControlledIntensities(
            dimensions=(3, 3, 1),
            peaks_per_spectrum=100,
            mass_range=(100.0, 500.0),
            intensity_threshold=1.0,  # Filter values < 1.0 at reader level
        )
        output_with_thresh = Path(tmpdir) / "with_threshold.zarr"
        converter_with_thresh = StreamingSpatialDataConverter(
            reader=reader_with_thresh,
            output_path=output_with_thresh,
            dataset_id="threshold_test",
            use_csc=True,
        )
        success = converter_with_thresh.convert()
        assert success, "Conversion with threshold should succeed"

        # Compare results
        from spatialdata import SpatialData

        sdata_no_thresh = SpatialData.read(str(output_no_thresh))
        sdata_with_thresh = SpatialData.read(str(output_with_thresh))

        adata_no_thresh = sdata_no_thresh.tables[list(sdata_no_thresh.tables.keys())[0]]
        adata_with_thresh = sdata_with_thresh.tables[
            list(sdata_with_thresh.tables.keys())[0]
        ]

        # With threshold, we should have roughly half the non-zero entries
        nnz_no_thresh = adata_no_thresh.X.nnz
        nnz_with_thresh = adata_with_thresh.X.nnz

        # The threshold version should have fewer entries
        assert (
            nnz_with_thresh < nnz_no_thresh
        ), f"Threshold should reduce entries: {nnz_with_thresh} >= {nnz_no_thresh}"

        # Should be roughly half (with some tolerance for edge effects)
        ratio = nnz_with_thresh / nnz_no_thresh
        assert 0.4 < ratio < 0.6, f"Expected ~50% reduction, got {ratio:.2%}"
