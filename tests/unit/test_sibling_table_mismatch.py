"""What is said when a sibling table will not add up to the summed one.

Two halves of issue #253.

``--msms-table`` defaults to on (design decision D2), but the CLI passes
the keyword only when the flag is given, and ``_lossless_spectrum_for``
read it as off-by-default -- so the common path, the default-on table,
never reached the warning that says a ``vendor_centroid`` summed
spectrum will not add up to it.

And the same disagreement between the same two tables was a WARNING for
the mobility grid and an INFO for the MS/MS split.

Issue #256 rides along here: ``--handle-3d`` and ``--z-spacing`` on a
single-slice acquisition skipped both promises docs/cli.md makes, because
``_is_volume`` is false for one plane and the log line was at DEBUG.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Iterator, List

import numpy as np
import pytest

from thyra.convert import _force_scan_sum, _lossless_spectrum_for


@contextmanager
def _records(logger_name: str, level: int = logging.INFO) -> Iterator[List]:
    """Collect records from a named logger, whatever the logging state.

    Deliberately not ``caplog``: ``setup_logging`` sets
    ``propagate = False`` on the ``thyra`` logger process-wide, so a
    caplog assertion here passes alone and fails in the full suite.
    """
    collected: List[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            collected.append(record)

    logger = logging.getLogger(logger_name)
    handler = _Collector(level=level)
    previous = logger.level
    logger.addHandler(handler)
    if previous > level or previous == logging.NOTSET:
        logger.setLevel(level)
    try:
        yield collected
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


class TestWhichTablesNeedTheLosslessSpectrum:
    def test_the_default_on_msms_table_counts_as_asked_for(self):
        """The CLI omits the keyword unless the flag was typed."""
        assert _lossless_spectrum_for({}) == ["demultiplexed MS/MS"]

    def test_opting_out_takes_it_off_the_list(self):
        assert _lossless_spectrum_for({"msms_table": False}) == []

    def test_the_grid_is_off_by_default(self):
        assert "mobility grid" not in _lossless_spectrum_for({})

    def test_both_can_be_wanted_at_once(self):
        assert _lossless_spectrum_for({"mobility_grid": True}) == [
            "mobility grid",
            "demultiplexed MS/MS",
        ]


class TestTheStartupWarning:
    def test_scan_sum_says_nothing(self):
        with _records("thyra.convert", logging.WARNING) as records:
            _force_scan_sum("bruker", {"tdf_spectrum": "scan_sum"}, ["mobility grid"])
        assert not records

    def test_the_vendor_centroid_warns(self):
        with _records("thyra.convert", logging.WARNING) as records:
            _force_scan_sum(
                "bruker", {"tdf_spectrum": "vendor_centroid"}, ["mobility grid"]
            )
        assert len(records) == 1
        message = records[0].getMessage()
        assert message.startswith("A mobility grid table is written")

    def test_two_tables_agree_with_their_verb(self):
        """ "A mobility grid and demultiplexed MS/MS table was asked for"."""
        with _records("thyra.convert", logging.WARNING) as records:
            _force_scan_sum(
                "bruker",
                {"tdf_spectrum": "vendor_centroid"},
                ["mobility grid", "demultiplexed MS/MS"],
            )
        message = records[0].getMessage()
        assert message.startswith(
            "A mobility grid table and a demultiplexed MS/MS table are written"
        )

    def test_a_format_that_ignores_the_mode_is_not_warned_about(self):
        """The CLI already says --tdf-spectrum is ignored on imzML (#260)."""
        with _records("thyra.convert", logging.WARNING) as records:
            _force_scan_sum(
                "imzml", {"tdf_spectrum": "vendor_centroid"}, ["demultiplexed MS/MS"]
            )
        assert not records


def _table(rows):
    """An AnnData stand-in: a matrix whose row sums are the ion current."""
    table = SimpleNamespace(X=np.asarray(rows, dtype=np.float64))
    table.uns = {}
    return table


class TestTheMismatchIsAWarningForEverySibling:
    """The grid says it at WARNING; the split said the same thing at INFO."""

    _MODULE = "thyra.converters.spatialdata.base_spatialdata_converter"

    def _record(self, split_rows, summed_rows):
        from thyra.converters.spatialdata.base_spatialdata_converter import (
            BaseSpatialDataConverter,
        )

        table = _table(split_rows)
        with _records(self._MODULE, logging.INFO) as records:
            BaseSpatialDataConverter._record_demultiplexed_current(
                table, _table(summed_rows), "msi_z0"
            )
        return table, records

    def test_a_disagreement_is_a_warning(self):
        table, records = self._record([[11.0, 0.0]], [[10.0, 0.0]])

        assert [r.levelno for r in records] == [logging.WARNING]
        assert "1.1000x" in records[0].getMessage()
        assert table.uns["demultiplexed_current"]["current_ratio"] == pytest.approx(1.1)

    def test_an_exact_split_is_silent_and_still_recorded(self):
        table, records = self._record([[10.0, 0.0]], [[10.0, 0.0]])

        assert not records
        assert table.uns["demultiplexed_current"]["current_ratio"] == pytest.approx(1.0)

    def test_the_grid_reports_the_same_disagreement_the_same_way(self):
        from thyra.converters.spatialdata.base_spatialdata_converter import (
            BaseSpatialDataConverter,
        )

        table = _table([[11.0, 0.0]])
        with _records(self._MODULE, logging.INFO) as records:
            BaseSpatialDataConverter._record_mobility_marginal(
                table, _table([[10.0, 0.0]]), "msi_z0"
            )

        assert [r.levelno for r in records] == [logging.WARNING]


class TestASinglePlaneVolume:
    """Issue #256: docs/cli.md promises both of these lines."""

    def _converter(self, z_arg, handle_3d=True, n_z=1):
        from thyra.core.base_converter import BaseMSIConverter, ZSpacingSource

        stub = SimpleNamespace(
            handle_3d=handle_3d,
            _dimensions=(4, 4, n_z),
            _z_spacing_um_arg=z_arg,
            z_spacing_um=10.0,
            z_spacing_source=ZSpacingSource.USER_PROVIDED,
            pixel_size_um=10.0,
        )
        stub._log_single_plane_volume = (
            lambda: BaseMSIConverter._log_single_plane_volume(stub)
        )
        # A property, so it cannot be inherited by a namespace stand-in.
        stub._is_volume = bool(handle_3d and n_z > 1)
        return BaseMSIConverter, stub

    def test_handle_3d_alone_says_the_volume_has_one_plane(self):
        cls, stub = self._converter(None)
        with _records("thyra.core.base_converter", logging.INFO) as records:
            cls._log_z_spacing(stub)
        assert any("1 plane" in r.getMessage() for r in records)

    def test_a_z_spacing_is_logged_as_ignored(self):
        cls, stub = self._converter(20.0)
        with _records("thyra.core.base_converter", logging.WARNING) as records:
            cls._log_z_spacing(stub)
        message = records[0].getMessage()
        assert "20 um is ignored" in message

    def test_a_plain_2d_conversion_stays_quiet(self):
        cls, stub = self._converter(None, handle_3d=False)
        with _records("thyra.core.base_converter", logging.INFO) as records:
            cls._log_z_spacing(stub)
        assert not records

    def test_a_real_volume_still_reports_its_spacing(self):
        cls, stub = self._converter(20.0, n_z=3)
        with _records("thyra.core.base_converter", logging.INFO) as records:
            cls._log_z_spacing(stub)
        assert any("z spacing" in r.getMessage() for r in records)
        assert not any("1 plane" in r.getMessage() for r in records)


@pytest.mark.parametrize("wanted", [["mobility grid"], ["demultiplexed MS/MS"]])
def test_no_mode_at_all_says_nothing(wanted):
    with _records("thyra.convert", logging.WARNING) as records:
        _force_scan_sum("bruker", {}, wanted)
    assert not records
