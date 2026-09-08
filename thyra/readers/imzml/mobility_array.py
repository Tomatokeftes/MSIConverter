"""The third binary array of an imzML with ion mobility.

imzML defines two binary arrays per spectrum. TIMSCONVERT and TIMSImaging
(both through the same pyimzML fork) add a third, declared exactly like the
other two: a ``referenceableParamGroup`` carrying a PSI-MS mobility array
term -- ``MS:1003006 mean inverse reduced ion mobility array`` with unit
``MS:1002814`` in both tools -- plus ``IMS:1000101 external data``, and per
spectrum a ``binaryDataArray`` referencing that group with the usual
``IMS:1000102`` offset, ``IMS:1000103`` length and ``IMS:1000104`` encoded
length. Upstream pyimzml matches only the m/z and intensity group ids and
ignores every other array, so the file reads as an ordinary imzML with the
mobility dimension silently dropped. This module finds the group and its
offsets so the reader can keep it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from xml.etree import ElementTree  # nosec B405 - same parser pyimzml uses

import numpy as np
from numpy.typing import NDArray

from ...core.mobility import MOBILITY_ARRAY_TERMS
from ...errors import ConversionRefused

_MZML_NS = "{http://psi.hupo.org/ms/mzml}"
_OFFSET_ACCESSION = "IMS:1000102"
_LENGTH_ACCESSION = "IMS:1000103"
_ENCODED_LENGTH_ACCESSION = "IMS:1000104"
_ZLIB_ACCESSION = "MS:1000574"

# Binary data type terms and the numpy dtype each declares.
_PRECISION_DTYPES: Dict[str, np.dtype] = {
    "MS:1000521": np.dtype(np.float32),
    "MS:1000523": np.dtype(np.float64),
    "MS:1000519": np.dtype(np.int32),
    "MS:1000522": np.dtype(np.int64),
}


@dataclass(frozen=True)
class MobilityArraySpec:
    """How an imzML declares its mobility array."""

    group_id: str
    array_accession: str
    array_name: str
    unit_accession: Optional[str]
    unit_name: Optional[str]
    dtype: np.dtype
    compressed: bool


def detect_mobility_array(metadata: Any) -> Optional[MobilityArraySpec]:
    """Find the referenceableParamGroup that declares a mobility array.

    Args:
        metadata: pyimzml's ``ImzMLParser.metadata``.

    Returns:
        The declaration, or ``None`` when the file has only m/z and
        intensity arrays. Only the first mobility group is taken; a file
        declaring two is not something any writer produces.
    """
    groups = getattr(metadata, "referenceable_param_groups", None)
    if not isinstance(groups, dict):
        return None

    for group_id, group in groups.items():
        cv_params = getattr(group, "cv_params", None) or []
        hit = None
        for param in cv_params:
            # (name, accession, parsed_value, raw_name, raw_value,
            #  unit_name, unit_accession)
            if len(param) >= 7 and param[1] in MOBILITY_ARRAY_TERMS:
                hit = param
                break
        if hit is None:
            continue

        accessions = {p[1] for p in cv_params if len(p) >= 2}
        dtype = None
        for accession, candidate in _PRECISION_DTYPES.items():
            if accession in accessions:
                dtype = candidate
                break
        if dtype is None:
            # The writer both tools use declares 64-bit float; a file that
            # declares nothing gets the same reading pyimzml would give an
            # undeclared m/z array.
            dtype = np.dtype(np.float64)

        return MobilityArraySpec(
            group_id=str(group_id),
            array_accession=str(hit[1]),
            array_name=MOBILITY_ARRAY_TERMS[hit[1]][0],
            unit_accession=hit[6] if hit[6] else None,
            unit_name=hit[5] if hit[5] else None,
            dtype=dtype,
            compressed=_ZLIB_ACCESSION in accessions,
        )
    return None


def _cv_param_int(elem: ElementTree.Element, accession: str) -> Optional[int]:
    for node in elem.iter(f"{_MZML_NS}cvParam"):
        if node.attrib.get("accession") == accession:
            value = node.attrib.get("value")
            if value is None:
                return None
            try:
                return int(value)
            except ValueError:
                return None
    return None


def collect_array_offsets(
    imzml_path: Union[str, Path], group_id: str, n_spectra: int
) -> Tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Per-spectrum offset, length and encoded length of one binary array.

    A second pass over the document, in the order pyimzml enumerated the
    spectra, so index ``i`` here is spectrum ``i`` there. Only done when a
    mobility group exists, and only once per reader.

    Args:
        imzml_path: The imzML file.
        group_id: The ``referenceableParamGroup`` id the array references.
        n_spectra: The number of spectra pyimzml found, for validation.

    Returns:
        ``(offsets, lengths, encoded_lengths)`` as int64 arrays; a spectrum
        that lacks the array has -1 in all three.

    Raises:
        ValueError: If the document holds a different number of spectra
            than pyimzml reported.
    """
    offsets = np.full(n_spectra, -1, dtype=np.int64)
    lengths = np.full(n_spectra, -1, dtype=np.int64)
    encoded = np.full(n_spectra, -1, dtype=np.int64)

    index = 0
    events = ElementTree.iterparse(str(imzml_path), events=("end",))  # nosec B314
    for _event, elem in events:
        if elem.tag != f"{_MZML_NS}spectrum":
            continue
        if index >= n_spectra:
            raise ConversionRefused(
                f"{imzml_path} holds more <spectrum> elements than the "
                f"{n_spectra} pyimzml reported"
            )
        for node in elem.findall(
            f"{_MZML_NS}binaryDataArrayList/{_MZML_NS}binaryDataArray"
        ):
            ref = node.find(f"{_MZML_NS}referenceableParamGroupRef")
            if ref is None or ref.attrib.get("ref") != group_id:
                continue
            offset = _cv_param_int(node, _OFFSET_ACCESSION)
            length = _cv_param_int(node, _LENGTH_ACCESSION)
            if offset is not None and length is not None:
                offsets[index] = offset
                lengths[index] = length
                enc = _cv_param_int(node, _ENCODED_LENGTH_ACCESSION)
                encoded[index] = enc if enc is not None else -1
            break
        index += 1
        # Free the subtree: the whole point of iterparse on a large file.
        elem.clear()

    if index != n_spectra:
        raise ConversionRefused(
            f"{imzml_path} holds {index} <spectrum> elements but pyimzml "
            f"reported {n_spectra}"
        )
    return offsets, lengths, encoded
