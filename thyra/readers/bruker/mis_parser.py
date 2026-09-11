"""Standalone parser for Bruker FlexImaging .mis XML files.

The .mis file contains teaching point calibration, acquisition area definitions,
and optical image references. It is used by both Rapiflex and timsTOF workflows
for aligning MSI data with optical images.

Parsing goes through ``defusedxml``, which is a hard dependency: see the
import below for why there is no stdlib fallback.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# defusedxml, unconditionally, and never xml.etree here. The stdlib parser
# expands entity declarations -- measured on a .mis carrying
# ``<!ENTITY r "5,5">``: xml.etree hands back the expansion and reports a
# 5x5 raster, defusedxml raises EntitiesForbidden -- so a fallback to it is
# not a degraded parse, it is the hole the defusedxml swap was made to
# close. defusedxml is declared in ``[project] dependencies`` with no
# optional-dependencies table anywhere in pyproject.toml, so an install
# without it is broken rather than a supported configuration, and a broken
# install is entitled to the ImportError and its traceback. This is the
# same principle the spatialdata import follows (issue #310); a
# try/except ImportError that warns and carries on is the mirror image of
# the machinery that commit deleted.
import defusedxml.ElementTree as ET
from defusedxml.common import DefusedXmlException

from ...errors import ConversionRefused

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element  # nosec B405 - type hint only

logger = logging.getLogger(__name__)


def find_mis_file_for_d_folder(data_path: Path) -> Optional[Path]:
    """Locate the .mis file that corresponds to a Bruker .d folder.

    A FlexImaging acquisition typically writes the .mis next to the .d folder,
    using the same stem. Falls back to any .mis in the parent directory.

    Args:
        data_path: Path to the Bruker .d directory

    Returns:
        Path to the matching .mis file, or None if none found.
    """
    parent = data_path.parent
    if not parent.exists():
        return None

    matching = parent / f"{data_path.stem}.mis"
    if matching.exists():
        return matching

    candidates = list(parent.glob("*.mis"))
    if candidates:
        return candidates[0]
    return None


def parse_mis_file(path: Path) -> Dict[str, Any]:
    """Parse a Bruker FlexImaging .mis XML file.

    Extracts teaching points, area definitions, raster info, and image
    references from the XML structure.

    A document defusedxml refuses and a document that is simply not
    well-formed are answered differently, on purpose. A malformed or
    truncated .mis costs the optical alignment, which every caller already
    treats as optional -- most acquisitions have no .mis at all -- so it
    stays a warning and an empty result. A refused document is a different
    claim: nothing about that file was read *because Thyra would not read
    it*, and saying so with an empty dict makes a security decision look
    exactly like a .mis that happened to hold nothing. The visible
    consequence was downstream and misleading: no areas, no teaching
    points, no raster, and a later ``--region <name>`` failing as "no such
    region" while the file that caused it went unnamed.

    Args:
        path: Path to the .mis file

    Returns:
        Dictionary with keys: teaching_points, areas, raster, ImageFile,
        OriginalImage, BaseGeometry (all optional depending on file
        content). Empty when the document is not well-formed XML.

    Raises:
        ConversionRefused: If the document declares XML entities or reaches
            for an external reference. Callers do not catch this: it
            travels out of the reader constructor to ``convert_msi``, which
            prints it once and stops.
    """
    metadata: Dict[str, Any] = {}

    try:
        tree = ET.parse(path)  # nosec B314
        root = tree.getroot()

        _extract_basic_elements(root, metadata)
        _extract_teaching_points(root, metadata)
        _extract_raster_info(root, metadata)
        _extract_areas(root, metadata)

    # DefusedXmlException is a ValueError, ET.ParseError a SyntaxError, so
    # the two clauses are disjoint and their order is presentation only.
    except DefusedXmlException as e:
        raise ConversionRefused(
            f"Refused to read the FlexImaging sequence file {path}: {e}. "
            "The document declares XML entities, or points at an external "
            "resource, and Thyra does not expand either -- an entity can "
            "pull a file off this machine into the acquisition metadata, "
            "or expand until the parser runs out of memory. Nothing was "
            "read from the file, so the acquisition areas, the teaching "
            "points and the raster step it carries are all unavailable and "
            "the conversion cannot use it. Re-export the imaging sequence "
            "from FlexImaging, or -- after reading what the declaration "
            "actually does -- remove the DOCTYPE from the file."
        ) from e

    except ET.ParseError as e:
        logger.warning(f"Failed to parse .mis file: {e}")

    return metadata


def _extract_basic_elements(root: "Element", metadata: Dict[str, Any]) -> None:
    """Extract basic text elements from .mis XML."""
    for elem_name in ["Method", "ImageFile", "OriginalImage", "BaseGeometry"]:
        elem = root.find(f".//{elem_name}")
        if elem is not None and elem.text:
            metadata[elem_name] = elem.text


def _extract_teaching_points(root: "Element", metadata: Dict[str, Any]) -> None:
    """Extract teaching point calibration data from .mis XML."""
    teaching_points: List[Dict[str, List[int]]] = []
    for tp in root.findall(".//TeachPoint"):
        if tp.text and ";" in tp.text:
            img_coords, stage_coords = tp.text.split(";")
            img_x, img_y = map(int, img_coords.split(","))
            stage_x, stage_y = map(int, stage_coords.split(","))
            teaching_points.append(
                {"image": [img_x, img_y], "stage": [stage_x, stage_y]}
            )
    if teaching_points:
        metadata["teaching_points"] = teaching_points


def _extract_raster_info(root: "Element", metadata: Dict[str, Any]) -> None:
    """Extract raster dimensions from .mis XML."""
    raster_elem = root.find(".//Raster")
    if raster_elem is not None and raster_elem.text:
        parts = raster_elem.text.split(",")
        if len(parts) == 2:
            metadata["raster"] = [int(parts[0]), int(parts[1])]


def _extract_areas(root: "Element", metadata: Dict[str, Any]) -> None:
    """Extract Area definitions from .mis XML.

    Areas define the image pixel coordinates for each acquisition region.
    Areas may be rectangular (Type=0, 2 points) or polygon (Type=3, N
    points). In both cases the bounding box of all points is stored as p1
    and p2, since alignment only needs the enclosing rectangle.

    Args:
        root: XML root element
        metadata: Dictionary to update with area info
    """
    areas: List[Dict[str, Any]] = []
    for area_elem in root.findall(".//Area"):
        area_name = area_elem.get("Name", "")
        points = area_elem.findall("Point")
        if len(points) >= 2:
            try:
                all_x: List[int] = []
                all_y: List[int] = []
                for p in points:
                    p_text = p.text or ""
                    p_parts = p_text.split(",")
                    all_x.append(int(p_parts[0]))
                    all_y.append(int(p_parts[1]))

                # Bounding box from all points
                areas.append(
                    {
                        "name": area_name,
                        "p1": [min(all_x), min(all_y)],
                        "p2": [max(all_x), max(all_y)],
                    }
                )
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse Area '{area_name}': {e}")
                continue

    if areas:
        metadata["areas"] = areas
        logger.debug(f"Parsed {len(areas)} area definitions from .mis")
