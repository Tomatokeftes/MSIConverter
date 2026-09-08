"""Tests for Bruker region selection by name vs DB number (#89)."""

import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest

from thyra.readers.bruker.timstof.timstof_reader import BrukerReader

_MODULE_LOGGER_NAME = "thyra.readers.bruker.timstof.timstof_reader"


@contextmanager
def _capture_module_logs() -> Iterator[List[str]]:
    """Attach a handler directly to the reader's module logger.

    pytest's ``caplog`` relies on log propagation to the root handler, which
    can be silently disabled by other tests in the session. Attaching a
    handler to the named logger directly captures records regardless of
    propagation state, making this assertion robust to test order.
    """
    records: List[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    module_logger = logging.getLogger(_MODULE_LOGGER_NAME)
    handler = _Capture(level=logging.INFO)
    previous_level = module_logger.level
    module_logger.addHandler(handler)
    module_logger.setLevel(logging.INFO)
    try:
        yield records
    finally:
        module_logger.removeHandler(handler)
        module_logger.setLevel(previous_level)


class _NameResolutionHarness:
    """Stand-in for a fully-initialised BrukerReader.

    Exposes only the attributes/methods used by ``_resolve_requested_region``
    and ``_log_region_mapping``, so we can test them without the heavy SDK
    initialisation path.
    """

    _mis_metadata: Dict[str, Any]
    _region_info: List[Tuple[int, int]]
    _requested_region: Optional[Any]

    # Reuse the methods under test directly from the real class so any future
    # behaviour change is automatically picked up.
    _get_region_name_map = BrukerReader._get_region_name_map
    _log_region_mapping = BrukerReader._log_region_mapping
    _resolve_requested_region = BrukerReader._resolve_requested_region
    _check_region_on_single_region_set = BrukerReader._check_region_on_single_region_set
    _select_region = BrukerReader._select_region


def _harness(
    areas: List[str],
    region_info: List[Tuple[int, int]],
    requested: Optional[Any],
) -> _NameResolutionHarness:
    h = _NameResolutionHarness()
    h._mis_metadata = {"areas": [{"name": n} for n in areas]}
    h._region_info = region_info
    h._requested_region = requested
    return h


def test_get_region_name_map_index_based() -> None:
    """The i-th .mis Area maps positionally to RegionNumber i."""
    h = _harness(["01", "02", "03"], [(0, 10), (1, 20), (2, 30)], None)
    assert h._get_region_name_map() == {0: "01", 1: "02", 2: "03"}


def test_resolve_int_passes_through() -> None:
    h = _harness(["01", "02", "03"], [(0, 10), (1, 20), (2, 30)], 2)
    assert h._resolve_requested_region() == 2


def test_resolve_string_matches_area_name() -> None:
    """User passing '03' (the .mis label) should land on RegionNumber 2 even
    though numerically '03' would be 3 — names take precedence over int parse.
    """
    h = _harness(
        ["01", "02", "03", "04", "05", "06"],
        [(0, 10), (1, 20), (2, 30), (3, 40), (4, 50), (5, 60)],
        "03",
    )
    assert h._resolve_requested_region() == 2


def test_resolve_string_falls_back_to_int_when_name_unknown() -> None:
    h = _harness(["A", "B", "C"], [(0, 10), (1, 20), (2, 30)], "1")
    assert h._resolve_requested_region() == 1


def test_resolve_string_raises_when_name_unknown_and_not_int() -> None:
    h = _harness(["A", "B", "C"], [(0, 10), (1, 20), (2, 30)], "doesnotexist")
    with pytest.raises(ValueError, match="not a recognised .mis Area Name"):
        h._resolve_requested_region()


def test_resolve_none_returns_none() -> None:
    h = _harness(["01", "02"], [(0, 10), (1, 20)], None)
    assert h._resolve_requested_region() is None


def test_log_region_mapping_emits_pairs() -> None:
    h = _harness(
        ["01", "02", "03"],
        [(0, 10), (1, 20), (2, 30)],
        None,
    )
    with _capture_module_logs() as records:
        h._log_region_mapping()
    text = "\n".join(records)
    assert "Region mapping" in text
    assert "RegionNumber 0 -> Area '01'" in text
    assert "RegionNumber 1 -> Area '02'" in text
    assert "RegionNumber 2 -> Area '03'" in text


def test_log_region_mapping_skips_single_region() -> None:
    h = _harness(["only"], [(0, 100)], None)
    with _capture_module_logs() as records:
        h._log_region_mapping()
    assert all("Region mapping" not in line for line in records)


def test_log_region_mapping_skips_when_no_areas() -> None:
    h = _harness([], [(0, 10), (1, 20)], None)
    with _capture_module_logs() as records:
        h._log_region_mapping()
    assert all("Region mapping" not in line for line in records)


class TestTheIntegerFallbackIsLogged:
    """Issue #252: '02' matches no Area Name and selects a different area.

    The name path has always logged what it resolved. The fallback said
    nothing at all, so a run that meant Area '02' and got Area '04' left
    only ``Region 2: 736 frames selected`` behind.
    """

    def test_the_failed_name_lookup_is_named(self) -> None:
        h = _harness(["01", "03", "04"], [(0, 10), (1, 20), (2, 30)], "02")
        with _capture_module_logs() as records:
            assert h._resolve_requested_region() == 2
        text = "\n".join(records)
        assert "matches no .mis Area Name" in text
        assert "RegionNumber 2" in text
        assert "01, 03, 04" in text

    def test_the_name_path_still_logs_what_it_resolved(self) -> None:
        h = _harness(["01", "03", "04"], [(0, 10), (1, 20), (2, 30)], "03")
        with _capture_module_logs() as records:
            assert h._resolve_requested_region() == 1
        assert any("Area Name" in line for line in records)


class TestASingleRegionSetStillChecksTheRequest:
    """Issue #252: ``_select_region`` returned before the request was read.

    On a single-area set ``--region foo`` converted everything at exit 0,
    while the same word on a three-area set was refused.
    """

    def test_an_unparseable_request_is_refused(self) -> None:
        h = _harness(["01"], [(0, 713)], "foo")
        with pytest.raises(ValueError, match="not a recognised .mis Area Name"):
            h._select_region()

    def test_a_region_that_is_not_there_is_refused(self) -> None:
        h = _harness(["01"], [(0, 713)], "3")
        with pytest.raises(ValueError, match="Region 3 not found"):
            h._select_region()

    def test_the_only_region_is_accepted_and_converts_everything(self) -> None:
        h = _harness(["01"], [(0, 713)], "01")
        with _capture_module_logs() as records:
            assert h._select_region() == (None, None)
        assert any("only region" in line for line in records)

    def test_no_request_is_still_silent_and_unfiltered(self) -> None:
        h = _harness(["01"], [(0, 713)], None)
        assert h._select_region() == (None, None)

    def test_a_request_with_no_region_information_is_refused(self) -> None:
        h = _harness([], [], "01")
        with pytest.raises(ValueError, match="no region information"):
            h._select_region()

    def test_no_request_with_no_region_information_still_converts(self) -> None:
        h = _harness([], [], None)
        assert h._select_region() == (None, None)
