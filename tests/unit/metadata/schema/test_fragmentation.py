"""What a store records about the fragmentation of its spectra.

An MS/MS imaging run measures fragments, so the m/z axis of the table
means fragment m/z. Nothing about the axis says so: before this block
existed, a converted MS/MS store was indistinguishable from an MS1 one,
and the precursor, the isolation width and the collision energy were
simply lost. These tests pin what is recorded instead.

The vocabulary is mzPeak's, checked against the reference archive at
HUPO-PSI/mzPeak ``502c3a4``: an ``isolation_window`` of a target plus a
lower and an upper offset -- not a single width -- with ``ms_level``
``MS:1000511``, the isolation terms ``MS:1000827``/``828``/``829``, and
activation as ``MS:1000133`` with ``MS:1000045`` collision energy in
electronvolts. Naming the same quantity differently from the archival
layer is the failure this pins against.
"""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from thyra.core.msms import (
    COLLISION_INDUCED_DISSOCIATION_ACCESSION,
    FragmentationSchedule,
    IsolationWindow,
    windows_overlap,
)
from thyra.metadata.schema import MSI_METADATA_SCHEMA_VERSION, Fragmentation
from thyra.metadata.schema import IsolationWindow as IsolationWindowModel
from thyra.metadata.schema import build_msi_metadata, validate_document
from thyra.metadata.schema.builder import _build_fragmentation

# The 15-window PASEF schedule measured on a real MALDI acquisition
# (220425_MSMS_pos_brain1.d): every pixel isolates all fifteen, each in
# its own slice of the mobility ramp.
_PASEF_WINDOWS = (
    (313.275, 1.0, 34.307, 2933, 3015),
    (353.320, 1.0, 35.172, 2832, 2901),
    (936.578, 1.0, 49.638, 994, 1127),
)


def _pasef_schedule() -> FragmentationSchedule:
    return FragmentationSchedule(
        ms_level=2,
        windows=tuple(
            IsolationWindow.from_full_width(
                mz, width, collision_energy=ce, scan_begin=lo, scan_end=hi
            )
            for mz, width, ce, lo, hi in _PASEF_WINDOWS
        ),
        dissociation_accession=COLLISION_INDUCED_DISSOCIATION_ACCESSION,
        source="bruker_tdf",
    )


class TestIsolationWindow:
    def test_a_full_width_is_halved_into_two_offsets(self):
        """mzPeak stores offsets; Bruker reports one width. Halving is the bridge.

        Asserted rather than assumed because the two spellings differ by a
        factor of two: a reader that passed the full width straight through
        would describe a window twice as wide as the quadrupole's, which no
        validator could catch -- both are plausible numbers.
        """
        window = IsolationWindow.from_full_width(1046.54, 1.5)

        assert window.lower_offset == 0.75
        assert window.upper_offset == 0.75
        assert window.target == 1046.54

    def test_an_unreported_width_stays_unreported(self):
        window = IsolationWindow.from_full_width(700.0, None)

        assert window.lower_offset is None and window.upper_offset is None

    def test_a_window_without_scans_is_not_mobility_resolved(self):
        assert not IsolationWindow(target=700.0).is_mobility_resolved
        assert IsolationWindow(
            target=700.0, scan_begin=10, scan_end=20
        ).is_mobility_resolved


class TestSchedule:
    def test_one_precursor_per_pixel_does_not_merge(self):
        schedule = FragmentationSchedule(
            ms_level=2, windows=(IsolationWindow(target=1046.54),)
        )

        assert schedule.is_msms
        assert not schedule.merges_precursors

    def test_several_precursors_per_pixel_merge(self):
        """The property that makes a summed MS/MS spectrum a chimera."""
        assert _pasef_schedule().merges_precursors

    def test_an_ms1_run_is_not_msms(self):
        schedule = FragmentationSchedule(ms_level=1)

        assert not schedule.is_msms and not schedule.merges_precursors

    def test_the_uns_block_carries_no_colons_in_its_keys(self):
        """Zarr writes a key as a directory name; Windows forbids the colon.

        Accessions are values here, never keys -- the same rule the
        mobility blocks follow, and the reason a store with a CV term as
        a key is unwritable on Windows rather than merely ugly.
        """
        block = _pasef_schedule().to_uns()

        assert block and not any(":" in key for key in block)

    def test_the_uns_block_is_arrays_not_lists_of_objects(self):
        """Parallel arrays survive the zarr writer; a list of dicts does not.

        Stored as objects they come back as numpy arrays of Python
        ``repr`` strings -- unparseable, and a deepcopy of one segfaults
        numpy 2.1-2.2 (numpy#28609), killing every consumer that copies
        the table.
        """
        block = _pasef_schedule().to_uns()

        for key in (
            "isolation_window_target",
            "isolation_window_lower_offset",
            "isolation_window_upper_offset",
            "collision_energy",
        ):
            assert isinstance(block[key], np.ndarray), key
            assert block[key].dtype == np.float64, key
            assert block[key].shape == (len(_PASEF_WINDOWS),), key
        assert block["scan_begin"].dtype == np.int64
        assert block["scan_end"].dtype == np.int64

    def test_the_uns_block_names_the_mzpeak_terms(self):
        block = _pasef_schedule().to_uns()

        assert block["ms_level_accession"] == "MS:1000511"
        assert block["isolation_window_target_accession"] == "MS:1000827"
        assert block["isolation_window_lower_offset_accession"] == "MS:1000828"
        assert block["isolation_window_upper_offset_accession"] == "MS:1000829"
        assert block["collision_energy_accession"] == "MS:1000045"
        assert block["collision_energy_unit_accession"] == "UO:0000266"
        assert block["dissociation_accession"] == "MS:1000133"

    def test_an_ms1_block_claims_no_precursor(self):
        block = FragmentationSchedule(ms_level=1).to_uns()

        assert block["ms_level"] == 1
        assert block["n_windows"] == 0
        assert "isolation_window_target" not in block

    def test_scan_ranges_are_only_written_when_every_window_has_one(self):
        """A partial scan range is worse than none: it reads as a full one."""
        mixed = FragmentationSchedule(
            ms_level=2,
            windows=(
                IsolationWindow(target=300.0, scan_begin=10, scan_end=20),
                IsolationWindow(target=400.0),
            ),
        )

        assert "scan_begin" not in mixed.to_uns()


class TestWindowsOverlap:
    def test_disjoint_mobility_windows_do_not_overlap(self):
        """What makes a PASEF frame separable by mobility alone."""
        assert not windows_overlap(_pasef_schedule().windows)

    def test_shared_scans_overlap(self):
        assert windows_overlap(
            (
                IsolationWindow(target=300.0, scan_begin=10, scan_end=30),
                IsolationWindow(target=400.0, scan_begin=20, scan_end=40),
            )
        )

    def test_windows_without_a_scan_range_count_as_overlapping(self):
        """Not separable is the honest answer when the ranges are unknown."""
        assert windows_overlap(
            (IsolationWindow(target=300.0), IsolationWindow(target=400.0))
        )


class TestBuilder:
    def test_a_reader_that_says_nothing_leaves_the_block_unset(self):
        """ "Not reported" is not "MS1", and must not be recorded as it."""
        assert _build_fragmentation(None) is None
        assert _build_fragmentation({}) is None

    def test_an_ms1_report_is_recorded_as_absent(self):
        block = _build_fragmentation(
            FragmentationSchedule(ms_level=1).to_extractor_report()
        )

        assert block is not None and block.present is False
        assert block.ms_level == 1 and not block.windows

    def test_the_pasef_report_round_trips_into_the_model(self):
        block = _build_fragmentation(_pasef_schedule().to_extractor_report())

        assert block is not None and block.present and block.ms_level == 2
        assert block.merges_precursors and block.constant_across_pixels
        assert block.dissociation_term.accession == "MS:1000133"
        assert [w.target for w in block.windows] == [313.275, 353.320, 936.578]
        assert block.windows[0].lower_offset == 0.5
        assert block.windows[0].scan_begin == 2933

    def test_a_window_without_a_usable_target_is_dropped(self):
        """A precursor list is what a consumer acts on; never invent one."""
        block = _build_fragmentation(
            {
                "present": True,
                "ms_level": 2,
                "windows": [{"collision_energy": 30.0}, {"isolation_window_target": 0}],
            }
        )

        assert block is not None and block.windows == []

    def test_it_lands_in_ms_analysis(self):
        meta = build_msi_metadata(
            None,
            pixel_size_um=(20.0, 20.0),
            fragmentation=_pasef_schedule().to_extractor_report(),
        )

        assert meta.ms_analysis.fragmentation is not None
        assert meta.schema_version == MSI_METADATA_SCHEMA_VERSION == "0.5.0"

    def test_the_demultiplexed_sibling_is_named_here_too(self):
        """Both sibling kinds must be discoverable from the versioned block.

        ``ion_mobility.resolved_table`` has named the mobility sibling since
        0.3.0. A consumer reading only this block would otherwise find one
        kind of sibling and not the other, and have to fall back to Thyra's
        own unversioned ``uns`` arrays to learn the MS/MS one exists.
        """
        meta = build_msi_metadata(
            None,
            pixel_size_um=(20.0, 20.0),
            fragmentation=_pasef_schedule().to_extractor_report(),
            msms_resolved_table="msi_z0_msms",
        )

        assert meta.ms_analysis.fragmentation.resolved_table == "msi_z0_msms"

    def test_it_stays_unset_when_no_sibling_was_written(self):
        block = _build_fragmentation(_pasef_schedule().to_extractor_report())

        assert block.resolved_table is None

    def test_a_survey_acquisition_never_names_one(self):
        """An MS1 run has no precursors, so it cannot have them split apart."""
        block = _build_fragmentation({"present": False, "ms_level": 1}, "msi_z0_msms")

        assert block.present is False and block.resolved_table is None


class TestModelValidators:
    def test_absent_fragmentation_cannot_be_ms2(self):
        with pytest.raises(ValidationError, match="ms_level"):
            Fragmentation(present=False, ms_level=2)

    def test_absent_fragmentation_cannot_carry_a_precursor(self):
        with pytest.raises(ValidationError, match="precursor fields"):
            Fragmentation(
                present=False,
                ms_level=1,
                windows=[IsolationWindowModel(target=700.0)],
            )

    def test_present_fragmentation_cannot_be_ms1(self):
        with pytest.raises(ValidationError, match="ms_level"):
            Fragmentation(present=True, ms_level=1)

    def test_merging_needs_more_than_one_window(self):
        with pytest.raises(ValidationError, match="more than one"):
            Fragmentation(
                present=True,
                ms_level=2,
                merges_precursors=True,
                windows=[IsolationWindowModel(target=700.0)],
            )

    def test_a_backwards_scan_range_is_rejected(self):
        with pytest.raises(ValidationError, match="scan_end"):
            IsolationWindowModel(target=700.0, scan_begin=30, scan_end=10)


class TestStorageEncoding:
    def test_windows_are_stored_as_json_not_as_objects(self):
        """The same rule ``processing`` follows, for the same reason.

        A list of objects does not round-trip through AnnData/zarr: it
        comes back as a numpy array of Python ``repr`` strings, which is
        neither parseable nor safe to deepcopy. Encoding it as JSON is
        what keeps the precursor list readable.
        """
        meta = build_msi_metadata(
            None,
            pixel_size_um=(20.0, 20.0),
            fragmentation=_pasef_schedule().to_extractor_report(),
        )
        stored = meta.to_uns_dict()["ms_analysis"]["fragmentation"]["windows"]

        assert isinstance(stored, str)
        assert [w["target"] for w in json.loads(stored)] == [313.275, 353.320, 936.578]

    def test_a_document_validates_with_windows_either_way(self):
        """``validate_document`` accepts the stored spelling and the parsed one."""
        meta = build_msi_metadata(
            None,
            pixel_size_um=(20.0, 20.0),
            fragmentation=_pasef_schedule().to_extractor_report(),
        )

        as_stored, issues = validate_document(meta.to_uns_dict())
        assert not [i for i in issues if i.severity == "error"], issues
        assert len(as_stored.ms_analysis.fragmentation.windows) == 3

        parsed = meta.model_dump(mode="json", exclude_none=True)
        as_parsed, issues = validate_document(parsed)
        assert not [i for i in issues if i.severity == "error"], issues
        assert len(as_parsed.ms_analysis.fragmentation.windows) == 3

    def test_unparseable_windows_are_an_error_not_a_crash(self):
        meta = build_msi_metadata(
            None,
            pixel_size_um=(20.0, 20.0),
            fragmentation=_pasef_schedule().to_extractor_report(),
        )
        doc = meta.to_uns_dict()
        doc["ms_analysis"]["fragmentation"]["windows"] = "{not json"

        result, issues = validate_document(doc)

        assert result is None
        assert any("windows" in issue.location for issue in issues)
