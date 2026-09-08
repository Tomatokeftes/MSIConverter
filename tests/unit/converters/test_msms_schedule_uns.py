"""The fragmentation schedule must reach the store, on every write path.

``uns["msms_schedule"]`` is what lets a consumer tell fragment m/z from
intact m/z. It is composed once in ``build_uns_metadata`` so every route
renders it, the same way the mobility blocks are -- the write paths have
drifted before (see ``test_uns_provenance_parity``), and a block added to
one and forgotten on another diverges silently.

The mock source is MS1; the schedule is supplied by overriding the reader
contract's :meth:`get_fragmentation`, which is how any format will report
it. That keeps this about the converter's handling rather than about one
vendor's SQL (which ``test_bruker_fragmentation`` covers).
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pytest

from tests.fixtures.mock_msi_generator import MockMSIConfig, MockMSIReader
from thyra.converters.spatialdata.base_spatialdata_converter import (
    SPATIALDATA_AVAILABLE,
)
from thyra.core.msms import (
    COLLISION_INDUCED_DISSOCIATION_ACCESSION,
    FragmentationSchedule,
    IsolationWindow,
)

pytestmark = pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)

_DATASET_ID = "mock"
_TABLE_NAME = f"{_DATASET_ID}_z0"


def _schedule(n_windows: int) -> FragmentationSchedule:
    return FragmentationSchedule(
        ms_level=2,
        windows=tuple(
            IsolationWindow.from_full_width(
                300.0 + 50.0 * i,
                1.0,
                collision_energy=30.0 + i,
                scan_begin=100 * i,
                scan_end=100 * i + 50,
            )
            for i in range(n_windows)
        ),
        dissociation_accession=COLLISION_INDUCED_DISSOCIATION_ACCESSION,
        source="stub",
    )


class _FragmentingReader(MockMSIReader):
    """The mock source, reporting a fragmentation schedule."""

    schedule: Optional[FragmentationSchedule] = None

    def get_fragmentation(self) -> Optional[FragmentationSchedule]:
        return self.schedule


def _reader(schedule: Optional[FragmentationSchedule]) -> _FragmentingReader:
    reader = _FragmentingReader(
        MockMSIConfig(n_x=4, n_y=4, n_mz_bins=500, peaks_per_spectrum=(20, 40))
    )
    reader.schedule = schedule
    return reader


def _convert(output_path, schedule):
    from thyra.converters.spatialdata.streaming_converter import (
        StreamingSpatialDataConverter,
    )

    common: Dict[str, Any] = {
        "reader": _reader(schedule),
        "output_path": output_path,
        "dataset_id": _DATASET_ID,
        "pixel_size_um": 10.0,
    }
    converter = StreamingSpatialDataConverter(**common, use_csc=True)
    assert converter.convert() is True
    return output_path


def _read_uns(output_path) -> Dict[str, Any]:
    import anndata as ad
    import zarr

    group = zarr.open_group(str(output_path), mode="r")
    return ad.io.read_elem(group["tables"][_TABLE_NAME]["uns"])


class TestStoredBlock:
    def test_the_schedule_reaches_the_store(self, tmp_path):
        out = _convert(tmp_path / "msms.zarr", _schedule(3))

        block = _read_uns(out)["msms_schedule"]

        assert block["ms_level"] == 2
        assert block["n_windows"] == 3
        assert bool(block["merges_precursors"]) is True
        np.testing.assert_allclose(
            np.asarray(block["isolation_window_target"]), [300.0, 350.0, 400.0]
        )
        np.testing.assert_allclose(
            np.asarray(block["isolation_window_lower_offset"]), [0.5, 0.5, 0.5]
        )
        assert not any(":" in key for key in block)

    def test_an_ms1_source_gets_no_block(self, tmp_path):
        """Absence is the signal: an MS1 store must not carry an empty one."""
        out = _convert(tmp_path / "ms1.zarr", FragmentationSchedule(ms_level=1))

        assert "msms_schedule" not in _read_uns(out)

    def test_a_reader_that_says_nothing_gets_no_block(self, tmp_path):
        out = _convert(tmp_path / "silent.zarr", None)

        assert "msms_schedule" not in _read_uns(out)

    def test_the_versioned_block_agrees_with_it(self, tmp_path):
        """``msi_metadata`` and ``msms_schedule`` describe the same acquisition.

        Two blocks from one source: the versioned schema document and the
        array block beside it. They are composed separately, so they can
        disagree, which would leave a consumer no way to tell which is
        right.
        """
        import json

        out = _convert(tmp_path / "both.zarr", _schedule(3))
        uns = _read_uns(out)

        fragmentation = uns["msi_metadata"]["ms_analysis"]["fragmentation"]
        windows = json.loads(fragmentation["windows"])

        assert fragmentation["ms_level"] == uns["msms_schedule"]["ms_level"]
        assert len(windows) == uns["msms_schedule"]["n_windows"]
        np.testing.assert_allclose(
            [w["target"] for w in windows],
            np.asarray(uns["msms_schedule"]["isolation_window_target"]),
        )

    def test_uns_holds_no_string_arrays(self, tmp_path):
        """A string array anywhere in ``uns`` segfaults a numpy 2.1-2.2 copy.

        The precursor list is a list of objects, which is exactly the
        shape that comes back as one unless it is encoded as JSON first.
        """

        def _string_arrays(value, path):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield from _string_arrays(child, f"{path}.{key}")
            elif isinstance(value, np.ndarray) and value.dtype.kind in "TUSO":
                yield path

        out = _convert(tmp_path / "strings.zarr", _schedule(3))

        assert not list(_string_arrays(_read_uns(out), "uns"))


class TestChimeraWarning:
    def test_merging_precursors_is_said_out_loud(self, tmp_path, caplog):
        """Silence here is the failure: a chimeric spectrum looks ordinary.

        Nothing in the output shows that one pixel's spectrum holds
        fragments of several precursors, so it is worth a WARNING rather
        than leaving it for a reader of the peaks to work out.
        """
        with caplog.at_level(logging.WARNING):
            _convert(tmp_path / "chimera.zarr", _schedule(3))

        merged = [r for r in caplog.records if "isolates 3 precursors" in r.message]
        assert merged and merged[0].levelno == logging.WARNING
        assert "msms_schedule" in merged[0].message

    def test_a_single_precursor_is_not_warned_about(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            _convert(tmp_path / "single.zarr", _schedule(1))

        assert not [r for r in caplog.records if "precursors per pixel" in r.message]

    def test_it_is_said_once_per_conversion(self, tmp_path, caplog):
        """The schedule is read once; the warning must not repeat per pixel."""
        with caplog.at_level(logging.WARNING):
            _convert(tmp_path / "once.zarr", _schedule(3))

        assert (
            len([r for r in caplog.records if "isolates 3 precursors" in r.message])
            == 1
        )
