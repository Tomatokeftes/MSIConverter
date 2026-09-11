"""Tests for the .mis file parser and discovery helpers."""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from thyra.errors import ConversionRefused
from thyra.readers.bruker.mis_parser import (
    _extract_areas,
    find_mis_file_for_d_folder,
    parse_mis_file,
)
from thyra.readers.bruker.timstof.timstof_reader import BrukerReader

RECTANGULAR_AREA = (
    '<Area Type="0" Name="01"><Point>10,20</Point><Point>30,40</Point></Area>'
)

# A real FlexImaging polygon, ten points, from the acquisition that produced
# issue #84: its bounding box corners are on different vertices from the first
# two points, so a parser that reads only two points gets it wrong.
POLYGON_AREA = """<Area Type="3" Name="01">
    <Point>24470,4585</Point>
    <Point>24420,3818</Point>
    <Point>24862,3543</Point>
    <Point>25353,3168</Point>
    <Point>26228,3043</Point>
    <Point>26753,4193</Point>
    <Point>26462,5485</Point>
    <Point>25362,5777</Point>
    <Point>24737,4943</Point>
    <Point>24487,4552</Point>
</Area>"""


def _write_mis(
    tmp_path: Path,
    name: str,
    raster: str = "5,5",
    area: str = RECTANGULAR_AREA,
) -> Path:
    mis = tmp_path / name
    mis.write_text(
        f"""<?xml version="1.0"?>
<ImagingSequence>
<ImageFile>img.tif</ImageFile>
<Raster>{raster}</Raster>
{area}
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


def test_extract_areas_rectangular() -> None:
    """Area extraction with rectangular (Type=0) 2-point areas.

    Moved here from test_rapiflex_reader.py, which called the same assertions
    against RapiflexReader._extract_areas. It passes on both sides of the
    parser merge: the shared parser already carried this logic. What it guards
    is the coverage, not the merge -- it is now asserted on the one code path
    Rapiflex, timsTOF, solariX and BrukerMetadataExtractor share.
    """
    root = ET.fromstring(f"<Root>{RECTANGULAR_AREA}</Root>")
    metadata: dict = {}
    _extract_areas(root, metadata)

    assert len(metadata["areas"]) == 1
    area = metadata["areas"][0]
    assert area["name"] == "01"
    assert area["p1"] == [10, 20]
    assert area["p2"] == [30, 40]


def test_extract_areas_polygon() -> None:
    """Area extraction with polygon (Type=3) N-point areas.

    The regression test for issue #84 / PR #85, which fixed a bounding box
    computed from the first two <Point> elements only. It was written against
    the Rapiflex copy of the parser and stayed there, so the shared parser --
    the one three readers and the metadata extractor use -- carried the fix
    with no test on it. Like the rectangular case it passes before and after
    the merge; it moves so the fix is covered where the code now lives.
    """
    root = ET.fromstring(f"<Root>{POLYGON_AREA}</Root>")
    metadata: dict = {}
    _extract_areas(root, metadata)

    assert len(metadata["areas"]) == 1
    area = metadata["areas"][0]
    assert area["name"] == "01"
    # Bounding box should span ALL points, not just the first two.
    assert area["p1"] == [24420, 3043]
    assert area["p2"] == [26753, 5777]


def test_parse_mis_extracts_polygon_area_bounding_box(tmp_path: Path) -> None:
    """The polygon bounding box survives the public entry point too.

    test_extract_areas_polygon calls the private helper with a tree built in
    memory. This one goes through parse_mis_file from a file on disk, so the
    XML parser, the .//Area search and the helper are exercised together.
    Passes before and after the merge, like the two above it.
    """
    mis = _write_mis(tmp_path, "polygon.mis", area=POLYGON_AREA)
    data = parse_mis_file(mis)

    assert data["areas"] == [{"name": "01", "p1": [24420, 3043], "p2": [26753, 5777]}]


def _write_entity_mis(tmp_path: Path, name: str = "entity.mis") -> Path:
    mis = tmp_path / name
    mis.write_text(
        """<?xml version="1.0"?>
<!DOCTYPE ImagingSequence [<!ENTITY r "5,5">]>
<ImagingSequence><Raster>&r;</Raster></ImagingSequence>
"""
    )
    return mis


def test_entity_bearing_mis_does_not_expand(tmp_path: Path) -> None:
    """An XML entity in a .mis is refused, not expanded.

    The only test in this file that fails before the defusedxml swap: on the
    unguarded `xml.etree` import the same file parsed happily and yielded
    ``{"raster": [5, 5]}``. defusedxml raises EntitiesForbidden, which is a
    ValueError rather than a ParseError, so the except clause has to name it
    or the exception escapes parse_mis_file into three callers that do not
    catch it.

    No importorskip on defusedxml: it is a hard dependency, and an
    importorskip would turn the one security test in this file into a
    silent pass on exactly the install where the hole is open.
    """
    with pytest.raises(ConversionRefused):
        parse_mis_file(_write_entity_mis(tmp_path))


def test_the_refusal_names_the_file_and_the_reason(tmp_path: Path) -> None:
    """A security refusal must not read like an empty file.

    This started as a warning and an empty dict, which none of the four
    consumers (the Rapiflex, timsTOF and solariX readers, and
    BrukerMetadataExtractor) checks for. The acquisition then ran with no
    areas, no teaching points and no raster step, and the first visible
    symptom was a later ``--region <name>`` failing as "no such region" --
    a message about a region list, pointing away from the file that
    emptied it. So the two things the message has to carry are which file
    and why.
    """
    mis = _write_entity_mis(tmp_path, "brain_section.mis")

    with pytest.raises(ConversionRefused) as excinfo:
        parse_mis_file(mis)

    message = str(excinfo.value)
    assert "brain_section.mis" in message
    assert "entit" in message.lower()


def test_malformed_xml_is_still_only_a_warning(tmp_path: Path, thyra_logs) -> None:
    """Not well-formed is not the same claim as refused.

    A truncated or corrupt .mis costs the optical alignment, which every
    caller already treats as optional -- most acquisitions have no .mis at
    all -- so it stays a warning and an empty result, exactly as before.
    The refusal above is the case where Thyra decided not to read a file it
    could have read, and that decision is the one that has to be audible.

    Not caplog: setup_logging sets propagate=False on the `thyra` logger
    process-globally, so a caplog assertion here would pass alone and fail
    after any test that has invoked the CLI. See the thyra_logs fixture.
    """
    mis = tmp_path / "truncated.mis"
    mis.write_text('<?xml version="1.0"?>\n<ImagingSequence><Raster>5,5')

    with thyra_logs("thyra.readers.bruker.mis_parser", logging.WARNING) as records:
        data = parse_mis_file(mis)

    assert data == {}
    assert any("Failed to parse .mis file" in r.getMessage() for r in records)


def test_the_timstof_reader_does_not_swallow_the_refusal(tmp_path: Path) -> None:
    """The one consumer with a ``except ValueError`` anywhere near it.

    ``BrukerReader._parse_mis_alignment`` wraps
    ``get_teaching_points_file()`` in ``except (ValueError, OSError):
    return {}`` for non-standard folder layouts. ``ConversionRefused`` is a
    ``ValueError``, so widening that try by two lines to cover the
    ``parse_mis_file`` call under it would restore the silence with no
    other visible change -- and this is the consumer where the silence
    hurt most, because ``_parse_mis_alignment`` runs in ``__init__``
    before ``_select_region``, and the areas it fills are what resolves
    ``--region <name>``.

    Called on an uninitialised instance: the method reads nothing but
    ``get_teaching_points_file``, and constructing the reader properly
    would need the vendor library.
    """
    reader = BrukerReader.__new__(BrukerReader)
    mis = _write_entity_mis(tmp_path)
    reader.get_teaching_points_file = lambda: mis  # type: ignore[method-assign]

    with pytest.raises(ConversionRefused, match="entity.mis"):
        reader._parse_mis_alignment()


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
