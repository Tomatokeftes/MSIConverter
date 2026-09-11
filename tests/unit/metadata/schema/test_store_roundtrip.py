"""The block a converter writes must read back and validate.

The uns parity suite (tests/unit/converters/test_uns_provenance_parity.py)
asserts the block is identical across all four write paths; this file
covers the consumer side: reading it out of a real store, validating
it, and exporting METASPACE metadata from it.
"""

import json

import pytest
from click.testing import CliRunner

from tests.fixtures.mock_msi_generator import MockMSIConfig, MockMSIReader
from thyra.metadata.schema import (
    MSI_METADATA_SCHEMA_VERSION,
    read_msi_metadata_blocks,
    validate_document,
)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    from thyra.converters.spatialdata import SpatialDataConverter

    output = tmp_path_factory.mktemp("schema_store") / "out.zarr"
    converter = SpatialDataConverter(
        reader=MockMSIReader(
            MockMSIConfig(n_x=4, n_y=4, n_mz_bins=200, peaks_per_spectrum=(10, 20))
        ),
        output_path=output,
        dataset_id="mock",
        pixel_size_um=10.0,
    )
    assert converter.convert() is True
    return output


class TestStoreRoundTrip:
    def test_block_reads_back_and_validates(self, store):
        blocks = read_msi_metadata_blocks(store)
        assert blocks, "converted store carries no msi_metadata block"
        for block in blocks.values():
            assert block["schema_version"] == MSI_METADATA_SCHEMA_VERSION
            assert block["ms_analysis"]["pixel_size_um"] == {"x": 10.0, "y": 10.0}
            meta, issues = validate_document(block)
            assert meta is not None
            assert not [i for i in issues if i.severity == "error"]

    def test_processing_history_records_the_conversion(self, store):
        from thyra import __version__

        for block in read_msi_metadata_blocks(store).values():
            steps = block["processing"]
            assert steps[0]["name"] == "conversion"
            assert steps[0]["software"] == {
                "name": "thyra",
                "version": __version__,
            }
            # No resampling was configured, so no resampling step.
            assert [s["name"] for s in steps] == ["conversion"]

    def test_root_attrs_carry_the_explicit_affine(self, store):
        import zarr

        cs_global = dict(zarr.open_group(str(store), mode="r").attrs)[
            "coordinate_systems"
        ]["global"]
        assert cs_global["raster_to_global_affine"] == [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

    def test_validate_cli_passes_on_a_real_store(self, store):
        from thyra.metadata.schema.cli import validate_command

        result = CliRunner().invoke(validate_command, [str(store)])
        assert result.exit_code == 0, result.output

    def test_export_metaspace_cli_works_on_a_real_store(self, store, tmp_path):
        from thyra.metadata.schema.cli import export_metaspace_command

        output = tmp_path / "submission.json"
        result = CliRunner().invoke(
            export_metaspace_command, [str(store), "-o", str(output)]
        )
        assert result.exit_code == 0, result.output
        document = json.loads(output.read_text(encoding="utf-8"))
        assert document["MS_Analysis"]["Pixel_Size"] == {"Xaxis": 10, "Yaxis": 10}

    def test_resampling_step_serialises_the_config(self, tmp_path):
        from thyra.converters.spatialdata import SpatialDataConverter

        converter = SpatialDataConverter(
            reader=MockMSIReader(
                MockMSIConfig(n_x=4, n_y=4, n_mz_bins=200, peaks_per_spectrum=(10, 20))
            ),
            output_path=tmp_path / "out.zarr",
            dataset_id="mock",
            pixel_size_um=10.0,
            resampling_config={"method": "nearest_neighbor", "target_bins": 100},
        )
        steps = converter._processing_provenance()
        assert [s.name for s in steps] == ["conversion", "mass axis resampling"]
        parameters = steps[1].parameters
        assert parameters["method"] == "nearest_neighbor"
        assert parameters["target_bins"] == 100
        # Unset config fields are dropped, not serialised as nulls.
        assert "min_mz" not in parameters

    def test_resolved_resampling_lands_in_the_stored_step(self, tmp_path):
        from thyra.converters.spatialdata import SpatialDataConverter

        output = tmp_path / "resampled.zarr"
        converter = SpatialDataConverter(
            reader=MockMSIReader(
                MockMSIConfig(n_x=4, n_y=4, n_mz_bins=200, peaks_per_spectrum=(10, 20))
            ),
            output_path=output,
            dataset_id="mock",
            pixel_size_um=10.0,
            resampling_config={"method": "nearest_neighbor", "target_bins": 100},
        )
        assert converter.convert() is True

        block = read_msi_metadata_blocks(output)["mock_z0"]
        step = next(
            s for s in block["processing"] if s["name"] == "mass axis resampling"
        )
        # The step declares what was DONE: the resolved axis, not just
        # the requested config.
        parameters = step["parameters"]
        assert parameters["method"] == "nearest_neighbor"
        assert parameters["target_bins"] == 100
        assert parameters["axis_type"]
        assert parameters["min_mz"] < parameters["max_mz"]

    def test_a_zarr_group_that_is_not_spatialdata_is_reported_cleanly(self, tmp_path):
        import zarr

        zarr.open_group(str(tmp_path / "plain.zarr"), mode="a")
        with pytest.raises(ValueError, match="tables"):
            read_msi_metadata_blocks(tmp_path / "plain.zarr")
