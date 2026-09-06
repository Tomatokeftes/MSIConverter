"""The ``ms_analysis.ion_mobility`` block: model rules and auto-population."""

import pytest
from pydantic import ValidationError

from thyra.metadata.schema import build_msi_metadata
from thyra.metadata.schema.models import (
    MSI_METADATA_SCHEMA_VERSION,
    IonMobility,
    MobilityGrid,
    MSIMetadata,
)
from thyra.metadata.schema.validate import validate_document
from thyra.metadata.types import ComprehensiveMetadata, EssentialMetadata


def _comprehensive(format_specific):
    essential = EssentialMetadata(
        dimensions=(3, 2, 1),
        coordinate_bounds=(0.0, 2.0, 0.0, 1.0),
        mass_range=(50.0, 1000.0),
        pixel_size=(20.0, 20.0),
        n_spectra=6,
        total_peaks=492,
        estimated_memory_gb=0.0,
        source_path="synthetic_tims.d",
    )
    return ComprehensiveMetadata(
        essential=essential,
        format_specific=format_specific,
        acquisition_params={},
        instrument_info={"instrument_name": "timsTOF fleX"},
        raw_metadata={},
    )


TDF_REPORT = {
    "present": True,
    "separation": "inverse reduced ion mobility",
    "separation_accession": "MS:1002815",
    "unit": "volt-second per square centimeter",
    "unit_accession": "MS:1002814",
    "num_scans_min": 240,
    "num_scans_max": 240,
    "one_over_k0_range": [0.9, 1.99],
}


class TestModel:
    def test_absent_mobility_carries_no_axis(self):
        with pytest.raises(ValidationError, match="present is False"):
            IonMobility(present=False, num_scans=240)

    def test_present_false_alone_is_fine(self):
        assert IonMobility(present=False).present is False

    def test_num_scans_must_be_positive(self):
        with pytest.raises(ValidationError):
            IonMobility(present=True, num_scans=0)

    def test_absent_mobility_names_no_resolved_table(self):
        with pytest.raises(ValidationError, match="present is False"):
            IonMobility(present=False, resolved_table="x_z0_mobility")

    def test_grid_needs_extent(self):
        with pytest.raises(ValidationError, match="upper"):
            MobilityGrid(law="linear", lower=1.2, upper=1.0, n_channels=256)
        with pytest.raises(ValidationError):
            MobilityGrid(law="linear", lower=1.0, upper=1.2, n_channels=0)

    def test_grid_round_trips(self):
        block = IonMobility(
            present=True,
            resolved_table="x_z0_mobility",
            grid=MobilityGrid(law="linear", lower=1.0, upper=1.29, n_channels=256),
        )
        dumped = block.model_dump(mode="json", exclude_none=True)
        assert dumped["grid"] == {
            "law": "linear",
            "lower": 1.0,
            "upper": 1.29,
            "n_channels": 256,
        }
        assert IonMobility.model_validate(dumped) == block


class TestBuilder:
    def test_tdf_report_becomes_a_full_block(self):
        block = build_msi_metadata(
            _comprehensive({"ion_mobility": TDF_REPORT}),
            pixel_size_um=(20.0, 20.0),
            source_format="bruker",
        )
        mobility = block.ms_analysis.ion_mobility
        assert mobility is not None
        assert mobility.present is True
        assert mobility.separation == "inverse reduced ion mobility"
        assert mobility.separation_term.accession == "MS:1002815"
        assert mobility.separation_term.name == "inverse reduced ion mobility"
        assert mobility.unit_term.accession == "MS:1002814"
        assert (mobility.range_lower, mobility.range_upper) == (0.9, 1.99)
        assert mobility.num_scans == 240

    def test_tsf_report_says_no_mobility(self):
        block = build_msi_metadata(
            _comprehensive({"ion_mobility": {"present": False}}),
            pixel_size_um=(20.0, 20.0),
            source_format="bruker",
        )
        assert block.ms_analysis.ion_mobility == IonMobility(present=False)

    def test_formats_that_say_nothing_leave_it_unset(self):
        block = build_msi_metadata(
            _comprehensive({}), pixel_size_um=(20.0, 20.0), source_format="imzml"
        )
        assert block.ms_analysis.ion_mobility is None

    def test_unknown_accessions_are_dropped_not_invented(self):
        report = dict(TDF_REPORT, separation_accession="MS:9999999")
        block = build_msi_metadata(
            _comprehensive({"ion_mobility": report}),
            pixel_size_um=(20.0, 20.0),
            source_format="bruker",
        )
        assert block.ms_analysis.ion_mobility.separation_term is None
        assert block.ms_analysis.ion_mobility.present is True

    def test_resolved_table_is_named_and_the_grid_stays_unset(self):
        block = build_msi_metadata(
            _comprehensive({"ion_mobility": TDF_REPORT}),
            pixel_size_um=(20.0, 20.0),
            source_format="bruker",
            mobility_resolved_table="tims_z0_mobility",
        )
        mobility = block.ms_analysis.ion_mobility
        assert mobility.resolved_table == "tims_z0_mobility"
        assert mobility.grid is None
        assert mobility.num_scans == 240

    def test_without_a_table_nothing_is_named(self):
        block = build_msi_metadata(
            _comprehensive({"ion_mobility": TDF_REPORT}),
            pixel_size_um=(20.0, 20.0),
            source_format="bruker",
        )
        assert block.ms_analysis.ion_mobility.resolved_table is None
        assert (
            "resolved_table" not in block.to_uns_dict()["ms_analysis"]["ion_mobility"]
        )

    def test_a_written_table_proves_the_dimension(self):
        # A reader that reported nothing still gets a present block when
        # the converter wrote a resolved table beside the summed one.
        block = build_msi_metadata(
            _comprehensive({}),
            pixel_size_um=(20.0, 20.0),
            source_format="imzml",
            mobility_resolved_table="mob_z0_mobility",
        )
        mobility = block.ms_analysis.ion_mobility
        assert mobility.present is True
        assert mobility.resolved_table == "mob_z0_mobility"

    def test_built_block_round_trips_and_validates(self):
        block = build_msi_metadata(
            _comprehensive({"ion_mobility": TDF_REPORT}),
            pixel_size_um=(20.0, 20.0),
            source_format="bruker",
        )
        as_uns = block.to_uns_dict()
        assert as_uns["schema_version"] == MSI_METADATA_SCHEMA_VERSION == "0.4.0"
        assert as_uns["ms_analysis"]["ion_mobility"]["num_scans"] == 240
        restored = MSIMetadata.model_validate(as_uns)
        assert restored.ms_analysis.ion_mobility == block.ms_analysis.ion_mobility
        model, issues = validate_document(as_uns)
        assert model is not None
        assert not [issue for issue in issues if issue.severity == "error"]
