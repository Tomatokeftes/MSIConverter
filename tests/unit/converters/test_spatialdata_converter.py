"""
Tests for the SpatialData converter.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from thyra.converters.spatialdata import (
    SpatialDataConverter,
    StreamingSpatialDataConverter,
)


def _create_mock_extractor(dims):
    """Create mock metadata extractor for test reader."""
    from thyra.core.base_extractor import MetadataExtractor
    from thyra.metadata.types import ComprehensiveMetadata, EssentialMetadata

    class MockExtractor(MetadataExtractor):
        def __init__(self, dims):
            super().__init__(None)
            self._dims = dims

        def _extract_essential_impl(self):
            n_spectra = self._dims[0] * self._dims[1] * self._dims[2]
            return EssentialMetadata(
                dimensions=self._dims,
                coordinate_bounds=(0.0, 2.0, 0.0, 2.0),
                mass_range=(100.0, 1000.0),
                pixel_size=None,
                n_spectra=n_spectra,
                total_peaks=n_spectra * 100,  # 100 peaks per spectrum
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

    return MockExtractor(dims)


def _generate_spectrum_data(dimensions):
    """Generate spectrum data for mock reader.

    Two peaks per pixel on a 100-bin axis, one moving with x and one with
    y, so a stored row names the pixel it came from.
    """
    mass_axis = np.linspace(100, 1000, 100)
    for z in range(dimensions[2]):
        for y in range(dimensions[1]):
            for x in range(dimensions[0]):
                intensities = np.zeros_like(mass_axis)
                intensities[x * 10 + 20] = 100.0
                intensities[y * 10 + 50] = 200.0
                yield ((x, y, z), mass_axis, intensities)


def create_mock_reader_with_dimensions(dimensions):
    """Helper to create mock reader with specific dimensions."""
    from pathlib import Path

    from thyra.core.base_reader import BaseMSIReader

    class MockMSIReader(BaseMSIReader):
        def __init__(self, dims, **kwargs):
            super().__init__(Path("/mock/path"), **kwargs)
            self._dimensions = dims
            self.closed = False

        def _create_metadata_extractor(self):
            return _create_mock_extractor(self._dimensions)

        def get_common_mass_axis(self):
            return np.linspace(100, 1000, 100)

        def iter_spectra(self, batch_size=None):
            return _generate_spectrum_data(self._dimensions)

        def close(self):
            self.closed = True

    return MockMSIReader(dimensions)


def _run_passes(converter):
    """Initialise, plan and run both passes; hand back the data structures."""
    converter._initialize_conversion()
    data_structures = converter._create_data_structures()
    converter._process_spectra(data_structures)
    return data_structures


class TestSpatialDataConverter:
    """Test the SpatialData converter functionality."""

    def test_initialization(self, temp_dir):
        """Test converter initialization."""
        output_path = temp_dir / "test_output.zarr"

        # Create mock reader with multiple z-slices for true 3D
        mock_reader = create_mock_reader_with_dimensions((3, 3, 2))

        # Initialize converter
        converter = SpatialDataConverter(
            mock_reader,
            output_path,
            dataset_id="test_dataset",
            pixel_size_um=2.5,
            handle_3d=True,
        )

        # Check initialization: the registered name is the streaming
        # converter, there being one converter.
        assert isinstance(converter, StreamingSpatialDataConverter)
        assert converter.reader == mock_reader
        assert converter.output_path == output_path
        assert converter.dataset_id == "test_dataset"
        assert converter.pixel_size_um == 2.5
        assert converter.handle_3d is True

    def test_plans_one_table_per_plane(self, temp_dir):
        """Without 3D handling, a two-plane dataset gets two plane tables."""
        output_path = temp_dir / "test_output.zarr"
        mock_reader = create_mock_reader_with_dimensions((3, 3, 2))

        converter = SpatialDataConverter(
            mock_reader, output_path, handle_3d=False, dataset_id="test_dataset"
        )
        converter._initialize_conversion()
        data_structures = converter._create_data_structures()

        assert data_structures["mode"] == "2d_slices"
        units = data_structures["units"]
        assert [u.key for u in units] == ["test_dataset_z0", "test_dataset_z1"]
        assert [u.region_key for u in units] == [
            "test_dataset_z0_pixels",
            "test_dataset_z1_pixels",
        ]
        assert [u.plane for u in units] == [0, 1]
        assert all(u.n_grid == 9 for u in units)
        assert all(u.tic.shape == (3, 3) for u in units)
        assert isinstance(data_structures["var_df"], pd.DataFrame)
        assert len(data_structures["var_df"]) == 100
        for key in ("tables", "shapes", "images"):
            assert data_structures[key] == {}

    def test_plans_one_volume_table(self, temp_dir):
        """With 3D handling, the same dataset is one volume table."""
        output_path = temp_dir / "test_output.zarr"
        mock_reader = create_mock_reader_with_dimensions((3, 3, 2))

        converter = SpatialDataConverter(mock_reader, output_path, handle_3d=True)
        converter._initialize_conversion()
        data_structures = converter._create_data_structures()

        assert data_structures["mode"] == "3d_volume"
        (unit,) = data_structures["units"]
        assert unit.key == "msi_dataset"
        assert unit.region_key == "msi_dataset_pixels"
        assert unit.plane is None
        assert unit.n_grid == 18
        assert unit.tic.shape == (2, 3, 3)

    def test_the_passes_scatter_each_spectrum_into_its_own_plane(self, temp_dir):
        """A pixel's peaks land in its plane's matrix, on its row, at its bins."""
        output_path = temp_dir / "test_output.zarr"
        mock_reader = create_mock_reader_with_dimensions((3, 3, 2))
        converter = SpatialDataConverter(
            mock_reader, output_path, handle_3d=False, dataset_id="test_dataset"
        )
        try:
            data_structures = _run_passes(converter)

            for unit in data_structures["units"]:
                matrix = unit.assembly.matrix().tocsr()
                assert matrix.shape == (9, 100)
                grid = 1 * 3 + 1  # pixel (1, 1) on this plane
                row = int(unit.row_of_grid[grid])
                assert row >= 0
                dense = matrix[row].toarray().ravel()
                assert dense[30] == 100.0  # x * 10 + 20
                assert dense[60] == 200.0  # y * 10 + 50
                assert np.count_nonzero(dense) == 2, "zeros are not stored"
        finally:
            converter._release_table_scratch()

    def test_the_volume_row_index_carries_z(self, temp_dir):
        """Two pixels at the same (x, y) on different planes get different rows."""
        output_path = temp_dir / "test_output.zarr"
        mock_reader = create_mock_reader_with_dimensions((3, 3, 2))
        converter = SpatialDataConverter(mock_reader, output_path, handle_3d=True)
        try:
            data_structures = _run_passes(converter)

            (unit,) = data_structures["units"]
            matrix = unit.assembly.matrix().tocsr()
            assert matrix.shape == (18, 100)
            rows = [int(unit.row_of_grid[z * 9 + 1 * 3 + 1]) for z in (0, 1)]
            assert rows[0] != rows[1]
            for row in rows:
                dense = matrix[row].toarray().ravel()
                assert dense[30] == 100.0
                assert dense[60] == 200.0
            obs = converter._table_obs(unit)
            assert obs["z"].tolist() == [0] * 9 + [1] * 9
        finally:
            converter._release_table_scratch()

    def test_finalize_builds_the_plane_elements(self, temp_dir):
        """Tables, shapes and TIC images, one set per plane, for real."""
        output_path = temp_dir / "test_output.zarr"
        mock_reader = create_mock_reader_with_dimensions((3, 3, 2))
        converter = SpatialDataConverter(
            mock_reader, output_path, handle_3d=False, dataset_id="test_dataset"
        )
        try:
            data_structures = _run_passes(converter)
            converter._finalize_data(data_structures)

            assert set(data_structures["tables"]) == {
                "test_dataset_z0",
                "test_dataset_z1",
            }
            assert set(data_structures["shapes"]) == {
                "test_dataset_z0_pixels",
                "test_dataset_z1_pixels",
            }
            assert set(data_structures["images"]) == {
                "test_dataset_z0_tic",
                "test_dataset_z1_tic",
            }
            table = data_structures["tables"]["test_dataset_z0"]
            assert table.shape == (9, 100)
            assert list(table.obs.columns) == [
                "y",
                "x",
                "region",
                "spatial_x",
                "spatial_y",
                "region_number",
                "instance_key",
            ]
            assert data_structures["images"]["test_dataset_z0_tic"].shape == (1, 3, 3)
            assert len(data_structures["shapes"]["test_dataset_z0_pixels"]) == 9
        finally:
            converter._release_table_scratch(data_structures["tables"])

    def test_finalize_builds_the_volume_elements(self, temp_dir):
        """One table with depth in obs, one 3D TIC image."""
        output_path = temp_dir / "test_output.zarr"
        mock_reader = create_mock_reader_with_dimensions((3, 3, 2))
        converter = SpatialDataConverter(mock_reader, output_path, handle_3d=True)
        try:
            data_structures = _run_passes(converter)
            converter._finalize_data(data_structures)

            assert set(data_structures["tables"]) == {"msi_dataset"}
            table = data_structures["tables"]["msi_dataset"]
            assert table.shape == (18, 100)
            assert list(table.obs.columns) == [
                "x",
                "y",
                "z",
                "region",
                "spatial_x",
                "spatial_y",
                "spatial_z",
                "region_number",
                "instance_key",
            ]
            assert data_structures["images"]["msi_dataset_tic"].shape == (1, 2, 3, 3)
        finally:
            converter._release_table_scratch(data_structures["tables"])

    @patch(
        "thyra.converters.spatialdata.base_spatialdata_converter.zarr.consolidate_metadata"
    )
    @patch("thyra.converters.spatialdata.base_spatialdata_converter." "SpatialData")
    def test_save_output(
        self, mock_spatial_data_class, mock_consolidate, mock_reader, temp_dir
    ):
        """Test saving output."""
        output_path = temp_dir / "test_output.zarr"

        # Create a customized mock_sdata that behaves more like the real thing
        class MockSpatialData:
            def __init__(self, **kwargs):
                self.tables = kwargs.get("tables", {})
                self.shapes = kwargs.get("shapes", {})
                self.images = kwargs.get("images", {})
                self.metadata = {}

            def write(self, path):
                return True

        # Set up our mock to use the custom class
        mock_spatial_data_class.side_effect = MockSpatialData

        converter = SpatialDataConverter(mock_reader, output_path)

        # Mock the add_metadata method to avoid any issues there
        converter.add_metadata = MagicMock()

        # Simple data structures
        data_structures = {
            "tables": {"table1": "mock_table"},
            "shapes": {"shape1": "mock_shape"},
            "images": {},  # Add images key to match converter expectations
        }

        # Call the method directly
        result = converter._save_output(data_structures)

        # Check results
        assert result is True  # The method should return True

        # Don't test the specifics of the mocks, just that the general flow
        # works
        assert converter.add_metadata.called

    def test_add_metadata(self, mock_reader, temp_dir):
        """Test adding metadata to SpatialData object."""
        output_path = temp_dir / "test_output.zarr"

        # Create mock SpatialData
        mock_sdata = MagicMock()
        mock_sdata.metadata = {}

        # Initialize converter
        converter = SpatialDataConverter(
            mock_reader,
            output_path,
            dataset_id="test_dataset",
            pixel_size_um=2.0,
        )
        converter._initialize_conversion()

        # Add metadata
        converter.add_metadata(mock_sdata)

        # Check metadata
        assert mock_sdata.metadata["conversion_info"]["dataset_id"] == "test_dataset"
        assert mock_sdata.metadata["conversion_info"]["pixel_size_um"] == 2.0
        assert "conversion_info" in mock_sdata.metadata

    @patch(
        "thyra.converters.spatialdata.base_spatialdata_converter.zarr.consolidate_metadata"
    )
    @patch("thyra.converters.spatialdata.base_spatialdata_converter." "SpatialData")
    def test_convert_end_to_end(
        self, mock_spatial_data, mock_consolidate, mock_reader, temp_dir
    ):
        """Test the full conversion process."""
        output_path = temp_dir / "test_output.zarr"

        # Set up mocks
        mock_sdata = MagicMock()
        mock_spatial_data.return_value = mock_sdata

        # Initialize converter
        converter = SpatialDataConverter(mock_reader, output_path)

        # Mock some internal methods to avoid complexity
        converter._finalize_data = MagicMock()

        # Run conversion
        result = converter.convert()

        # Check result
        assert result is True
        assert converter._finalize_data.called
        assert mock_spatial_data.called
        assert mock_sdata.write.called


class TestSparseFormatIsGone:
    """CSC is the one layout written, and ``sparse_format`` no longer selects.

    The keyword chose between CSC and CSR until v3.22, honoured by the
    in-memory converters only -- so its meaning depended on ``streaming``,
    and on the streaming route a ``csr`` request had spent a release
    silently producing CSC. It is removed rather than mirrored: every
    consumer here reads columns, and a caller who wants rows can call
    ``X.tocsr()`` on what they read back for one conversion in memory.

    Removing it from the signature is not enough on its own. An unknown
    keyword falls through ``**kwargs`` into ``BaseMSIConverter.options``
    without a word, which would reproduce exactly the silent-CSC failure
    the removal was meant to end, so passing it has to raise.
    """

    def test_the_keyword_is_refused_rather_than_swallowed(self, temp_dir):
        mock_reader = create_mock_reader_with_dimensions((3, 3, 1))

        with pytest.raises(ValueError, match=r"sparse_format was removed"):
            SpatialDataConverter(
                mock_reader,
                temp_dir / "test_output.zarr",
                dataset_id="test_dataset",
                pixel_size_um=2.5,
                sparse_format="csr",
            )

    def test_csc_is_refused_too(self, temp_dir):
        """Even the value that matches what is written.

        Accepting ``"csc"`` would leave a keyword that does nothing, which
        is the shape of option this removal is clearing out. The error
        costs the caller one deleted argument.
        """
        mock_reader = create_mock_reader_with_dimensions((3, 3, 1))

        with pytest.raises(ValueError, match=r"sparse_format was removed"):
            SpatialDataConverter(
                mock_reader,
                temp_dir / "test_output.zarr",
                dataset_id="test_dataset",
                sparse_format="csc",
            )

    def test_the_selector_is_gone_from_the_converters(self, temp_dir):
        """A leftover ``_sparse_format`` would read as a live choice.

        Left behind, the next person sets it and the branch that consumed
        it no longer exists.
        """
        mock_reader = create_mock_reader_with_dimensions((3, 3, 1))
        converter = SpatialDataConverter(
            mock_reader,
            temp_dir / "test_output.zarr",
            dataset_id="test_dataset",
            pixel_size_um=2.5,
        )

        assert not hasattr(converter, "_sparse_format")
        assert "sparse_format" not in converter.options


class TestNormalizeResamplingConfig:
    """Tests for _normalize_resampling_config."""

    def test_passthrough_dataclass(self):
        """Passing a ResamplingConfig returns it unchanged."""
        from thyra.converters.spatialdata.base_spatialdata_converter import (
            _normalize_resampling_config,
        )
        from thyra.resampling.types import ResamplingConfig

        cfg = ResamplingConfig(target_bins=500)
        result = _normalize_resampling_config(cfg)
        assert result is cfg

    def test_dict_target_bins(self):
        """Dict with target_bins converts correctly."""
        from thyra.converters.spatialdata.base_spatialdata_converter import (
            _normalize_resampling_config,
        )
        from thyra.resampling.types import ResamplingConfig

        result = _normalize_resampling_config({"target_bins": 2000})
        assert isinstance(result, ResamplingConfig)
        assert result.target_bins == 2000

    def test_dict_method_nearest_neighbor(self):
        """Dict method string maps to ResamplingMethod enum."""
        from thyra.converters.spatialdata.base_spatialdata_converter import (
            _normalize_resampling_config,
        )
        from thyra.resampling.types import ResamplingMethod

        result = _normalize_resampling_config({"method": "nearest_neighbor"})
        assert result.method == ResamplingMethod.NEAREST_NEIGHBOR

    def test_dict_auto_method_becomes_none(self):
        """method='auto' maps to None (use default)."""
        from thyra.converters.spatialdata.base_spatialdata_converter import (
            _normalize_resampling_config,
        )

        result = _normalize_resampling_config({"method": "auto"})
        assert result.method is None

    def test_dict_width_at_mz_maps_to_mass_width_da(self):
        """Dict key width_at_mz maps to ResamplingConfig.mass_width_da."""
        from thyra.converters.spatialdata.base_spatialdata_converter import (
            _normalize_resampling_config,
        )

        result = _normalize_resampling_config({"width_at_mz": 0.01})
        assert result.mass_width_da == 0.01

    def test_dict_reference_mz_default(self):
        """reference_mz defaults to 1000.0 when not supplied."""
        from thyra.converters.spatialdata.base_spatialdata_converter import (
            _normalize_resampling_config,
        )

        result = _normalize_resampling_config({})
        assert result.reference_mz == 1000.0

    def test_dict_reference_mz_custom(self):
        """Explicit reference_mz is preserved."""
        from thyra.converters.spatialdata.base_spatialdata_converter import (
            _normalize_resampling_config,
        )

        result = _normalize_resampling_config({"reference_mz": 500.0})
        assert result.reference_mz == 500.0

    def test_dict_axis_type_string(self):
        """axis_type string maps to AxisType enum."""
        from thyra.converters.spatialdata.base_spatialdata_converter import (
            _normalize_resampling_config,
        )
        from thyra.resampling.types import AxisType

        result = _normalize_resampling_config({"axis_type": "orbitrap"})
        assert result.axis_type == AxisType.ORBITRAP


class TestTableObs:
    """Tests for the obs table built from a unit's kept rows."""

    def _unit(self, temp_dir, dimensions, pixel_size_um=10.0, handle_3d=False):
        """A planned table unit with every grid position occupied."""
        mock_reader = create_mock_reader_with_dimensions(dimensions)
        converter = SpatialDataConverter(
            mock_reader,
            temp_dir / "test_output.zarr",
            dataset_id="ds",
            pixel_size_um=pixel_size_um,
            handle_3d=handle_3d,
        )
        converter._dimensions = dimensions
        converter._common_mass_axis = np.linspace(100.0, 1000.0, 5)
        unit = converter._plan_tables()[0]
        unit.occupancy[:] = True
        unit.finish_counting()
        return converter, unit

    def test_2d_pixel_count(self, temp_dir):
        """A plane produces n_x * n_y rows."""
        converter, unit = self._unit(temp_dir, (4, 3, 1))
        assert len(converter._table_obs(unit)) == 12

    def test_3d_pixel_count(self, temp_dir):
        """A volume produces n_x * n_y * n_z rows."""
        converter, unit = self._unit(temp_dir, (2, 3, 4), handle_3d=True)
        assert len(converter._table_obs(unit)) == 24

    def test_2d_x_range(self, temp_dir):
        """x coordinates span [0, n_x-1]."""
        converter, unit = self._unit(temp_dir, (5, 4, 1))
        df = converter._table_obs(unit)
        assert df["x"].min() == 0
        assert df["x"].max() == 4

    def test_2d_y_range(self, temp_dir):
        """y coordinates span [0, n_y-1]."""
        converter, unit = self._unit(temp_dir, (5, 4, 1))
        df = converter._table_obs(unit)
        assert df["y"].min() == 0
        assert df["y"].max() == 3

    def test_2d_has_no_z_columns(self, temp_dir):
        """A plane table carries no depth: it has none."""
        converter, unit = self._unit(temp_dir, (3, 3, 1))
        df = converter._table_obs(unit)
        assert "z" not in df.columns
        assert "spatial_z" not in df.columns

    def test_3d_z_range(self, temp_dir):
        """z coordinates span [0, n_z-1] for a volume."""
        converter, unit = self._unit(temp_dir, (2, 2, 3), handle_3d=True)
        df = converter._table_obs(unit)
        assert df["z"].min() == 0
        assert df["z"].max() == 2

    def test_unique_coordinates(self, temp_dir):
        """Every (x, y, z) combination is unique."""
        converter, unit = self._unit(temp_dir, (3, 4, 2), handle_3d=True)
        df = converter._table_obs(unit)
        tuples = list(zip(df["x"], df["y"], df["z"]))
        assert len(tuples) == len(set(tuples))

    def test_spatial_coords_scale_with_pixel_size(self, temp_dir):
        """spatial_x / spatial_y are x / y multiplied by pixel_size_um."""
        converter, unit = self._unit(temp_dir, (3, 3, 1), pixel_size_um=25.0)
        df = converter._table_obs(unit)
        np.testing.assert_array_equal(df["spatial_x"], df["x"] * 25.0)
        np.testing.assert_array_equal(df["spatial_y"], df["y"] * 25.0)

    def test_instance_id_index(self, temp_dir):
        """The dataframe index is named instance_id."""
        converter, unit = self._unit(temp_dir, (2, 2, 1))
        df = converter._table_obs(unit)
        assert df.index.name == "instance_id"

    def test_kept_rows_keep_their_grid_index(self, temp_dir):
        """Dropping positions compacts the offsets, never the identities."""
        converter, unit = self._unit(temp_dir, (3, 2, 1))
        unit.occupancy[:] = False
        unit.occupancy[[1, 4]] = True
        unit.finish_counting()
        df = converter._table_obs(unit)
        assert df.index.tolist() == ["1", "4"]
        assert df["x"].tolist() == [1, 1]
        assert df["y"].tolist() == [0, 1]

    def test_region_number_without_region_map(self, temp_dir):
        """Without a region_map, region_number is 1 for all pixels."""
        converter, unit = self._unit(temp_dir, (2, 2, 1))
        df = converter._table_obs(unit)
        assert (df["region_number"] == 1).all()

    def test_region_number_with_region_map(self, temp_dir):
        """With a region_map, pixels get the correct region number."""
        converter, unit = self._unit(temp_dir, (2, 2, 1))
        converter._region_map = {(0, 0): 0, (1, 0): 0, (0, 1): 1, (1, 1): 1}
        df = converter._table_obs(unit)
        row_x0_y0 = df[(df["x"] == 0) & (df["y"] == 0)]
        row_x0_y1 = df[(df["x"] == 0) & (df["y"] == 1)]
        assert row_x0_y0["region_number"].iloc[0] == 0
        assert row_x0_y1["region_number"].iloc[0] == 1
