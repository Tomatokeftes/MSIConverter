"""Tests for the Waters metadata extractor."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from thyra.metadata.extractors.waters_extractor import WatersMetadataExtractor
from thyra.readers.waters.imaging_grid import ImagingGrid
from thyra.readers.waters.masslynx_lib import FunctionType, ScanInfoData


def _make_scan_info(x_mm, y_mm, has_pos=True):
    """Create a ScanInfoData."""
    return ScanInfoData(
        ms_level=1,
        polarity=0,
        drift_scan_count=0,
        is_profile=0,
        precursor_mz=0.0,
        rt=1.0,
        laser_x_pos=x_mm if has_pos else -1.0,
        laser_y_pos=y_mm if has_pos else -1.0,
    )


def _make_grid_and_ml(n_x=3, n_y=2, n_peaks=5):
    """Create a mock MassLynxLib, handle, and ImagingGrid for testing.

    Returns:
        Tuple of (mock_ml, handle, grid, function_types, ms_functions).
    """
    # Build x/y index maps
    x_positions = [100.0 * (i + 1) for i in range(n_x)]
    y_positions = [100.0 * (i + 1) for i in range(n_y)]
    x_index_map = {v: i for i, v in enumerate(x_positions)}
    y_index_map = {v: i for i, v in enumerate(y_positions)}

    # Build scan map: 1 MS function, n_x * n_y scans
    scan_map = {}
    scan_idx = 0
    for yi, y_um in enumerate(y_positions):
        for xi, x_um in enumerate(x_positions):
            x_mm = x_um / 1000.0
            y_mm = y_um / 1000.0
            scan_map[(0, scan_idx)] = _make_scan_info(x_mm, y_mm)
            scan_idx += 1

    n_scans = n_x * n_y

    lateral_width = x_positions[-1] - x_positions[0] if n_x > 1 else 0.0
    lateral_height = y_positions[-1] - y_positions[0] if n_y > 1 else 0.0
    pixel_size_x = lateral_width / n_x if n_x > 1 else 0.0
    pixel_size_y = lateral_height / n_y if n_y > 1 else 0.0

    grid = ImagingGrid(
        x_index_map=x_index_map,
        y_index_map=y_index_map,
        pixel_count_x=n_x,
        pixel_count_y=n_y,
        pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y,
        lateral_width=lateral_width,
        lateral_height=lateral_height,
        scan_map=scan_map,
    )

    # Mock MassLynxLib
    mock_ml = MagicMock()
    mock_ml.get_number_of_scans_in_function.return_value = n_scans
    mock_ml.is_raw_spectrum_profile.return_value = False  # centroid
    mock_ml.get_acquisition_date.return_value = "2026-01-15"
    mock_ml.is_lockmass_corrected.return_value = False
    mock_ml.get_lockmass_function.return_value = -1
    mock_ml.get_number_of_functions.return_value = 1
    mock_ml.get_acquisition_range.return_value = (100.0, 1000.0)

    # Each spectrum returns n_peaks m/z values
    mzs = np.linspace(100.0, 1000.0, n_peaks)
    intensities = np.random.default_rng(42).uniform(10.0, 500.0, n_peaks)
    mock_ml.read_spectrum.return_value = (mzs, intensities)

    handle = "mock_handle"
    function_types = {0: FunctionType.MS}
    ms_functions = [0]

    return mock_ml, handle, grid, function_types, ms_functions


class TestWatersMetadataExtractorEssential:
    """Test essential metadata extraction."""

    def test_dimensions(self):
        """Test that dimensions match the imaging grid."""
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml(n_x=3, n_y=2)
        extractor = WatersMetadataExtractor(
            mock_ml, handle, Path("/test/data.raw"), grid, ft, ms
        )
        essential = extractor.get_essential()

        assert essential.dimensions == (3, 2, 1)

    def test_coordinate_bounds(self):
        """Test coordinate bounds are 0-based pixel indices."""
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml(n_x=4, n_y=3)
        extractor = WatersMetadataExtractor(
            mock_ml, handle, Path("/test/data.raw"), grid, ft, ms
        )
        essential = extractor.get_essential()

        assert essential.coordinate_bounds == (0.0, 3.0, 0.0, 2.0)

    def test_mass_range(self):
        """Test mass range from scanned spectra."""
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml(n_x=2, n_y=2, n_peaks=10)
        extractor = WatersMetadataExtractor(
            mock_ml, handle, Path("/test/data.raw"), grid, ft, ms
        )
        essential = extractor.get_essential()

        assert essential.mass_range[0] == pytest.approx(100.0)
        assert essential.mass_range[1] == pytest.approx(1000.0)

    def test_spectra_count(self):
        """Test that n_spectra matches the number of positioned scans."""
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml(n_x=3, n_y=2)
        extractor = WatersMetadataExtractor(
            mock_ml, handle, Path("/test/data.raw"), grid, ft, ms
        )
        essential = extractor.get_essential()

        assert essential.n_spectra == 6  # 3 * 2

    def test_total_peaks(self):
        """Test total peak count."""
        n_peaks = 20
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml(n_x=2, n_y=2, n_peaks=n_peaks)
        extractor = WatersMetadataExtractor(
            mock_ml, handle, Path("/test/data.raw"), grid, ft, ms
        )
        essential = extractor.get_essential()

        assert essential.total_peaks == 4 * n_peaks  # 4 pixels * 20 peaks each

    def test_pixel_size(self):
        """Test pixel size from imaging grid."""
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml(n_x=3, n_y=2)
        extractor = WatersMetadataExtractor(
            mock_ml, handle, Path("/test/data.raw"), grid, ft, ms
        )
        essential = extractor.get_essential()

        assert essential.pixel_size is not None
        assert essential.pixel_size[0] > 0
        assert essential.pixel_size[1] > 0

    def test_spectrum_type_centroid(self):
        """Test centroid spectrum type detection."""
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml()
        mock_ml.is_raw_spectrum_profile.return_value = False
        extractor = WatersMetadataExtractor(
            mock_ml, handle, Path("/test/data.raw"), grid, ft, ms
        )
        essential = extractor.get_essential()

        assert essential.spectrum_type == "centroid spectrum"

    def test_spectrum_type_profile(self):
        """A profile-acquired file read as the profile trace."""
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml()
        mock_ml.is_raw_spectrum_profile.return_value = True
        extractor = WatersMetadataExtractor(
            mock_ml, handle, Path("/test/data.raw"), grid, ft, ms, use_centroid=False
        )
        essential = extractor.get_essential()

        assert essential.spectrum_type == "profile spectrum"

    def test_spectrum_type_is_what_the_reader_delivers(self):
        """The same profile-acquired file, read through the vendor centroider."""
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml()
        mock_ml.is_raw_spectrum_profile.return_value = True
        extractor = WatersMetadataExtractor(
            mock_ml, handle, Path("/test/data.raw"), grid, ft, ms, use_centroid=True
        )
        essential = extractor.get_essential()

        assert essential.spectrum_type == "centroid spectrum"

    def test_source_path(self):
        """Test source path is preserved."""
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml()
        extractor = WatersMetadataExtractor(
            mock_ml, handle, Path("/test/data.raw"), grid, ft, ms
        )
        essential = extractor.get_essential()

        assert essential.source_path == str(Path("/test/data.raw"))

    def test_caching(self):
        """Test that essential metadata is cached."""
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml()
        extractor = WatersMetadataExtractor(
            mock_ml, handle, Path("/test/data.raw"), grid, ft, ms
        )
        e1 = extractor.get_essential()
        e2 = extractor.get_essential()

        assert e1 is e2

    def test_peak_counts_per_pixel(self):
        """Test per-pixel peak count array."""
        n_peaks = 15
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml(n_x=2, n_y=2, n_peaks=n_peaks)
        extractor = WatersMetadataExtractor(
            mock_ml, handle, Path("/test/data.raw"), grid, ft, ms
        )
        essential = extractor.get_essential()

        assert essential.peak_counts_per_pixel is not None
        assert len(essential.peak_counts_per_pixel) == 4
        assert all(c == n_peaks for c in essential.peak_counts_per_pixel)


class TestWatersMetadataExtractorComprehensive:
    """Test comprehensive metadata extraction."""

    def test_comprehensive_contains_essential(self):
        """Test that comprehensive metadata includes essential."""
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml()
        extractor = WatersMetadataExtractor(
            mock_ml, handle, Path("/test/data.raw"), grid, ft, ms
        )
        comprehensive = extractor.get_comprehensive()

        assert comprehensive.essential is not None
        assert comprehensive.essential.dimensions == (3, 2, 1)

    def test_format_specific(self):
        """Test Waters-specific metadata fields."""
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml()
        extractor = WatersMetadataExtractor(
            mock_ml, handle, Path("/test/data.raw"), grid, ft, ms
        )
        comprehensive = extractor.get_comprehensive()

        fs = comprehensive.format_specific
        # "format" is what DataCharacteristics.from_metadata reads for its
        # is_waters_raw flag; "data_format" predates it and stays for the
        # stored format-specific block.
        assert fs["format"] == "Waters MassLynx raw"
        assert fs["data_format"] == "waters_raw"
        assert fs["is_imaging"] is True
        assert fs["pixel_count_x"] == 3
        assert fs["pixel_count_y"] == 2
        assert fs["ms_functions"] == [0]

    def test_instrument_info(self):
        """The key must be "manufacturer" -- the one the detector chain reads.

        This said "vendor" once, which nothing downstream consumed, so the
        fact that a file was Waters never reached the resampling decision.
        """
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml()
        extractor = WatersMetadataExtractor(
            mock_ml, handle, Path("/test/data.raw"), grid, ft, ms
        )
        comprehensive = extractor.get_comprehensive()

        assert comprehensive.instrument_info["manufacturer"] == "Waters"

    def test_acquisition_params(self):
        """Test acquisition parameters extraction."""
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml()
        extractor = WatersMetadataExtractor(
            mock_ml, handle, Path("/test/data.raw"), grid, ft, ms
        )
        comprehensive = extractor.get_comprehensive()

        params = comprehensive.acquisition_params
        assert "acquisition_date" in params
        assert params["acquisition_date"] == "2026-01-15"
        assert "is_lockmass_corrected" in params

    def test_raw_metadata_imaging_grid(self):
        """Test raw metadata contains imaging grid details."""
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml()
        extractor = WatersMetadataExtractor(
            mock_ml, handle, Path("/test/data.raw"), grid, ft, ms
        )
        comprehensive = extractor.get_comprehensive()

        raw = comprehensive.raw_metadata
        assert "imaging_grid" in raw
        assert raw["imaging_grid"]["pixel_count_x"] == 3
        assert raw["imaging_grid"]["pixel_count_y"] == 2


class TestWatersMetadataExtractorEdgeCases:
    """Test edge cases in metadata extraction."""

    def test_read_spectrum_error_handled(self):
        """Test that spectrum read errors during metadata scan are handled."""
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml(n_x=2, n_y=1)

        # First spectrum succeeds, second fails
        mzs = np.array([100.0, 500.0, 1000.0])
        ints = np.array([10.0, 20.0, 30.0])
        mock_ml.read_spectrum.side_effect = [
            (mzs, ints),
            RuntimeError("DLL crash"),
        ]

        extractor = WatersMetadataExtractor(
            mock_ml, handle, Path("/test/data.raw"), grid, ft, ms
        )
        essential = extractor.get_essential()

        # Should still have data from the first spectrum
        assert essential.n_spectra == 1
        assert essential.total_peaks == 3


class TestNoFabricatedMassRange:
    """A dataset with no readable spectra must not acquire an invented range.

    ``_scan_all_ms_spectra`` used to substitute ``(0.0, 1000.0)`` behind one
    warning that never mentioned a mass range. That value reached disk as
    ``uns/essential_metadata/mass_range`` and, with resampling on, sized the
    whole common axis -- so a file whose real range is 100-200 came out with
    a 0-1000 axis and no indication that the number was made up.

    ``test_mass_range`` above asserts ``approx(100.0)``, which is the real
    range of its fixture; it never exercised the fallback and so could not
    have caught its removal.
    """

    def _extractor(self, mock_ml, handle, grid, ft, ms):
        return WatersMetadataExtractor(
            mock_ml, handle, Path("/test/data.raw"), grid, ft, ms
        )

    def test_no_readable_scan_raises(self):
        """Every read fails: there is nothing to derive a range from."""
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml(n_x=2, n_y=2)
        mock_ml.read_spectrum.side_effect = RuntimeError("DLL crash")

        with pytest.raises(ValueError, match="Could not determine the mass range"):
            self._extractor(mock_ml, handle, grid, ft, ms).get_essential()

    def test_every_scan_empty_raises(self):
        """Reads succeed but return no peaks -- the truncated-data shape."""
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml(n_x=2, n_y=2)
        mock_ml.read_spectrum.return_value = (np.array([]), np.array([]))

        with pytest.raises(ValueError, match="Could not determine the mass range"):
            self._extractor(mock_ml, handle, grid, ft, ms).get_essential()

    def test_the_error_names_the_dataset(self):
        """So the message identifies which acquisition failed."""
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml(n_x=2, n_y=2)
        mock_ml.read_spectrum.return_value = (np.array([]), np.array([]))

        with pytest.raises(ValueError, match=r"data\.raw"):
            self._extractor(mock_ml, handle, grid, ft, ms).get_essential()

    def test_one_readable_scan_is_enough(self):
        """The guard must not fire while any real measurement survives."""
        mock_ml, handle, grid, ft, ms = _make_grid_and_ml(n_x=2, n_y=1)
        mock_ml.read_spectrum.side_effect = [
            (np.array([300.0, 400.0]), np.array([1.0, 2.0])),
            (np.array([]), np.array([])),
        ]
        # No acquisition range, so the stored span is the range; with one
        # reported, that setting would rightly be used instead.
        mock_ml.get_acquisition_range.return_value = None

        essential = self._extractor(mock_ml, handle, grid, ft, ms).get_essential()

        assert essential.mass_range == pytest.approx((300.0, 400.0))


class TestWatersResamplingDetection:
    """From the extractor's real output to the detector chain's answer.

    Built through ``_resampling_metadata_dict``, the same shaping the preview
    and the converter use, so what is asserted is the contract the two sides
    actually meet on. Before ``WatersDetector`` existed, the answer hinged on
    ``is_raw_spectrum_profile``: centroid landed on ``CentroidImzMLDetector``
    by accident, profile fell to ``DefaultDetector``'s CONSTANT axis.
    """

    @staticmethod
    def _chain_answer(is_profile, use_centroid, instrument=None):
        from thyra.preview import _resampling_metadata_dict
        from thyra.resampling.decision_tree import ResamplingDecisionTree

        mock_ml, handle, grid, ft, ms = _make_grid_and_ml()
        mock_ml.is_raw_spectrum_profile.return_value = is_profile
        extractor = WatersMetadataExtractor(
            mock_ml,
            handle,
            Path("/test/data.raw"),
            grid,
            ft,
            ms,
            instrument=instrument,
            use_centroid=use_centroid,
        )
        comprehensive = extractor.get_comprehensive()
        metadata = _resampling_metadata_dict(comprehensive.essential, comprehensive)
        tree = ResamplingDecisionTree()
        return (
            tree.select_axis_type(metadata),
            tree.select_strategy(metadata),
            tree.select_reference_width(metadata),
            tree.select_tof_law(metadata),
        )

    @pytest.mark.parametrize("is_profile", [True, False])
    def test_vendor_centroid_bins_on_reflector_tof(self, is_profile):
        """What the reader delivers decides, not what the file was acquired in."""
        from thyra.resampling.types import AxisType, ResamplingMethod

        axis, method, width, law = self._chain_answer(is_profile, use_centroid=True)
        assert axis is AxisType.REFLECTOR_TOF
        assert method is ResamplingMethod.NEAREST_NEIGHBOR
        assert width == (0.002, 1000.0)
        assert law is None

    def test_profile_trace_interpolates_on_linear_tof(self):
        from thyra.resampling.types import AxisType, ResamplingMethod

        axis, method, width, _ = self._chain_answer(True, use_centroid=False)
        assert axis is AxisType.LINEAR_TOF
        assert method is ResamplingMethod.TIC_PRESERVING
        # Not an MRT and no digitiser constants: the MRT width, with a warning.
        assert width == (0.0013, 1000.0)

    def test_mrt_profile_trace_is_pinned_at_1_3_mda(self):
        from thyra.readers.waters.instrument import WatersInstrument
        from thyra.resampling.types import AxisType, ResamplingMethod

        mrt = WatersInstrument(is_mrt=True, decided_by="OpticMode=MRT")
        axis, method, width, _ = self._chain_answer(
            True, use_centroid=False, instrument=mrt
        )
        assert axis is AxisType.LINEAR_TOF
        assert method is ResamplingMethod.TIC_PRESERVING
        assert width == (0.0013, 1000.0)

    def test_mrt_vendor_centroid_follows_the_measured_width_law(self):
        from thyra.readers.waters.instrument import WatersInstrument
        from thyra.resampling.mass_axis import MRT_TOF_LAW
        from thyra.resampling.types import AxisType, ResamplingMethod

        mrt = WatersInstrument(is_mrt=True, decided_by="OpticMode=MRT")
        axis, method, width, law = self._chain_answer(
            True, use_centroid=True, instrument=mrt
        )
        assert axis is AxisType.TOF
        assert method is ResamplingMethod.NEAREST_NEIGHBOR
        assert width is None
        assert law == MRT_TOF_LAW

    def test_centroid_acquired_file_cannot_deliver_a_trace(self):
        """Asked for the profile of a centroid-mode file: centroids, so reflector."""
        from thyra.resampling.types import AxisType, ResamplingMethod

        axis, method, _, _ = self._chain_answer(False, use_centroid=False)
        assert axis is AxisType.REFLECTOR_TOF
        assert method is ResamplingMethod.NEAREST_NEIGHBOR
