"""Tests for the .mis file parser and discovery helpers."""

import logging
from pathlib import Path

import pytest

from thyra.readers.bruker.mis_parser import find_mis_file_for_d_folder, parse_mis_file


def _write_mis(tmp_path: Path, name: str, raster: str = "5,5") -> Path:
    mis = tmp_path / name
    mis.write_text(
        f"""<?xml version="1.0"?>
<ImagingSequence>
<ImageFile>img.tif</ImageFile>
<Raster>{raster}</Raster>
<Area Name="01"><Point>10,20</Point><Point>30,40</Point></Area>
</ImagingSequence>
"""
    )
    return mis


def test_parse_mis_extracts_raster(tmp_path: Path) -> None:
    mis = _write_mis(tmp_path, "sample.mis", raster="5,5")
    data = parse_mis_file(mis)
    assert data["raster"] == [5, 5]


def test_parse_mis_handles_rectangular_raster(tmp_path: Path) -> None:
    mis = _write_mis(tmp_path, "sample.mis", raster="10,20")
    data = parse_mis_file(mis)
    assert data["raster"] == [10, 20]


def test_find_mis_prefers_matching_stem(tmp_path: Path) -> None:
    """When several .mis files sit in the parent, prefer the one whose stem
    matches the .d folder stem.
    """
    d_folder = tmp_path / "sample_A.d"
    d_folder.mkdir()
    _write_mis(tmp_path, "sample_A.mis", raster="5,5")
    _write_mis(tmp_path, "sample_B.mis", raster="50,50")

    found = find_mis_file_for_d_folder(d_folder)
    assert found is not None
    assert found.name == "sample_A.mis"


def test_find_mis_falls_back_to_any(tmp_path: Path) -> None:
    d_folder = tmp_path / "sample.d"
    d_folder.mkdir()
    other = _write_mis(tmp_path, "different_name.mis", raster="7,7")

    found = find_mis_file_for_d_folder(d_folder)
    assert found == other


def test_find_mis_returns_none_when_missing(tmp_path: Path) -> None:
    d_folder = tmp_path / "sample.d"
    d_folder.mkdir()
    assert find_mis_file_for_d_folder(d_folder) is None


def test_entity_bearing_mis_does_not_expand(tmp_path: Path, thyra_logs) -> None:
    """An XML entity in a .mis is refused, not expanded.

    The only test in this file that fails before the defusedxml swap: on the
    unguarded `xml.etree` import the same file parsed happily and yielded
    ``{"raster": [5, 5]}``. defusedxml raises EntitiesForbidden, which is a
    ValueError rather than a ParseError, so the except clause has to name it
    or the exception escapes parse_mis_file into three callers that do not
    catch it.

    Not caplog: setup_logging sets propagate=False on the `thyra` logger
    process-globally, so a caplog assertion here would pass alone and fail
    after any test that has invoked the CLI. See the thyra_logs fixture.
    """
    pytest.importorskip("defusedxml")
    mis = tmp_path / "entity.mis"
    mis.write_text(
        """<?xml version="1.0"?>
<!DOCTYPE ImagingSequence [<!ENTITY r "5,5">]>
<ImagingSequence><Raster>&r;</Raster></ImagingSequence>
"""
    )

    with thyra_logs("thyra.readers.bruker.mis_parser", logging.WARNING) as records:
        data = parse_mis_file(mis)

    assert data == {}
    assert any("Failed to parse .mis file" in r.getMessage() for r in records)


def test_internal_subset_dtd_still_parses(tmp_path: Path) -> None:
    """A .mis carrying a plain DTD is not collateral damage of the above.

    defusedxml's parse() defaults are forbid_dtd=False, forbid_entities=True,
    forbid_external=True. Only the entity declaration is refused; a document
    type declaration on its own still reads. Passes before and after, and is
    here so a later tightening to forbid_dtd=True cannot pass unnoticed.
    """
    mis = tmp_path / "dtd.mis"
    mis.write_text(
        """<?xml version="1.0"?>
<!DOCTYPE ImagingSequence [<!ELEMENT Raster (#PCDATA)>]>
<ImagingSequence><Raster>5,5</Raster></ImagingSequence>
"""
    )

    assert parse_mis_file(mis)["raster"] == [5, 5]
