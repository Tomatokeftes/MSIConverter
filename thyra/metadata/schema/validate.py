# thyra/metadata/schema/validate.py
"""Validate MSI metadata documents against the schema.

Validation happens in three layers:

1. ``schema_version`` compatibility (semantic-version rules),
2. structural validation through the pydantic models, and
3. ontology checks: PSI-MS/IMS/UO accessions must exist in the shipped
   tables with a matching label; fields with a designated ontology
   (NCBITaxon, UBERON, CHEBI) warn when the accession's prefix is not
   the designated one.

Errors mean the document does not conform; warnings mean it conforms
but says something suspicious.  ``thyra validate`` exits non-zero only
on errors, so warnings do not break CI pipelines that gate on it.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import ValidationError

from ...utils.windows_paths import prepare_zarr_read_path
from ..ontology.cache import ONTOLOGY
from .models import (
    MSI_METADATA_SCHEMA_VERSION,
    MSI_VAR_MOBILITY_COLUMN,
    MSI_VAR_PRECURSOR_COLUMN,
    MSIMetadata,
    OntologyTerm,
)

logger = logging.getLogger(__name__)

# Prefixes checked against the locally shipped ontology tables.
_LOCAL_PREFIXES = ("MS", "IMS", "UO")

# The designated ontology for each ``*_term`` field, as
# (section, field) -> accepted prefixes.  A term from another ontology
# is a warning, not an error: EFO and friends legitimately cover some
# of the same ground.
EXPECTED_TERM_PREFIXES: Dict[Tuple[str, str], Tuple[str, ...]] = {
    ("sample", "organism_term"): ("NCBITaxon",),
    ("sample", "organism_part_term"): ("UBERON",),
    ("preparation", "matrix_term"): ("CHEBI",),
    ("ms_analysis", "polarity_term"): ("MS",),
    ("ms_analysis", "ionisation_source_term"): ("MS",),
    ("ms_analysis", "analyzer_term"): ("MS",),
}


@dataclass(frozen=True)
class ValidationIssue:
    """One finding from validation.

    Attributes:
        severity: ``"error"`` (does not conform) or ``"warning"``
            (conforms but suspicious).
        location: Dotted path into the document, e.g.
            ``"ms_analysis.pixel_size_um.x"``.
        message: Human-readable description.
    """

    severity: str
    location: str
    message: str


def _check_schema_version(doc: Dict[str, Any]) -> List[ValidationIssue]:
    """Semantic-version compatibility of ``schema_version``."""
    issues: List[ValidationIssue] = []
    version = doc.get("schema_version")
    if version is None:
        issues.append(
            ValidationIssue(
                "error",
                "schema_version",
                "schema_version is missing; a stored document must state "
                "which schema version it conforms to",
            )
        )
        return issues
    if not isinstance(version, str) or version.count(".") != 2:
        issues.append(
            ValidationIssue(
                "error",
                "schema_version",
                f"schema_version {version!r} is not a MAJOR.MINOR.PATCH string",
            )
        )
        return issues

    try:
        major, minor, _ = (int(part) for part in version.split("."))
    except ValueError:
        issues.append(
            ValidationIssue(
                "error",
                "schema_version",
                f"schema_version {version!r} is not a MAJOR.MINOR.PATCH string",
            )
        )
        return issues

    impl_major, impl_minor, _ = (
        int(part) for part in MSI_METADATA_SCHEMA_VERSION.split(".")
    )
    if major != impl_major:
        issues.append(
            ValidationIssue(
                "error",
                "schema_version",
                f"document is schema major version {major}, this Thyra "
                f"implements {MSI_METADATA_SCHEMA_VERSION}",
            )
        )
    elif minor > impl_minor:
        issues.append(
            ValidationIssue(
                "warning",
                "schema_version",
                f"document is schema {version}, newer than the implemented "
                f"{MSI_METADATA_SCHEMA_VERSION}; fields added since may be "
                "rejected as unknown",
            )
        )
    return issues


def _check_term(
    term: Optional[OntologyTerm], location: str, prefixes: Tuple[str, ...]
) -> List[ValidationIssue]:
    """Ontology checks for one ``*_term`` field."""
    if term is None:
        return []

    issues: List[ValidationIssue] = []
    prefix = term.accession.split(":", 1)[0]

    if prefix not in prefixes:
        issues.append(
            ValidationIssue(
                "warning",
                location,
                f"accession {term.accession} is from '{prefix}'; the "
                f"designated ontology here is {' / '.join(prefixes)}",
            )
        )

    if prefix in _LOCAL_PREFIXES:
        entry = ONTOLOGY.terms.get(term.accession)
        if entry is None:
            issues.append(
                ValidationIssue(
                    "error",
                    location,
                    f"accession {term.accession} is not a known {prefix} term",
                )
            )
        elif entry[0].lower() != term.name.lower():
            issues.append(
                ValidationIssue(
                    "warning",
                    location,
                    f"name {term.name!r} does not match the {prefix} label "
                    f"{entry[0]!r} for {term.accession}",
                )
            )
    return issues


def _check_ontology_terms(meta: MSIMetadata) -> List[ValidationIssue]:
    """Ontology checks across every ``*_term`` field of the document."""
    issues: List[ValidationIssue] = []
    for (section, field), prefixes in EXPECTED_TERM_PREFIXES.items():
        term = getattr(getattr(meta, section), field)
        issues.extend(_check_term(term, f"{section}.{field}", prefixes))
    return issues


def validate_document(
    doc: Any,
) -> Tuple[Optional[MSIMetadata], List[ValidationIssue]]:
    """Validate one metadata document.

    Args:
        doc: The parsed document (normally a dict read from a store's
            ``uns["msi_metadata"]`` or from a JSON file).

    Returns:
        ``(model, issues)``.  The model is ``None`` when structural
        validation failed; issues carry everything found, errors first
        is not guaranteed -- filter on ``severity``.
    """
    if not isinstance(doc, dict):
        return None, [
            ValidationIssue(
                "error", "", f"document must be a mapping, got {type(doc).__name__}"
            )
        ]

    # On disk, `processing` is a JSON string (a list of objects cannot
    # round-trip through AnnData/zarr); accept both spellings so a raw
    # uns dict validates without going through read_msi_metadata_blocks.
    if isinstance(doc.get("processing"), str):
        doc = dict(doc)
        try:
            doc["processing"] = json.loads(doc["processing"])
        except json.JSONDecodeError as exc:
            return None, [
                ValidationIssue("error", "processing", f"not valid JSON: {exc}")
            ]

    # ``ms_analysis.fragmentation.windows`` is JSON on disk for the same
    # reason; accept both spellings here too.
    analysis = doc.get("ms_analysis")
    if isinstance(analysis, dict) and isinstance(analysis.get("fragmentation"), dict):
        windows = analysis["fragmentation"].get("windows")
        if isinstance(windows, str):
            doc = dict(doc)
            analysis = dict(analysis)
            fragmentation = dict(analysis["fragmentation"])
            try:
                fragmentation["windows"] = json.loads(windows)
            except json.JSONDecodeError as exc:
                return None, [
                    ValidationIssue(
                        "error",
                        "ms_analysis.fragmentation.windows",
                        f"not valid JSON: {exc}",
                    )
                ]
            analysis["fragmentation"] = fragmentation
            doc["ms_analysis"] = analysis

    issues = _check_schema_version(doc)
    if any(issue.severity == "error" for issue in issues):
        return None, issues

    try:
        meta = MSIMetadata.model_validate(doc)
    except ValidationError as exc:
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"])
            issues.append(ValidationIssue("error", location, error["msg"]))
        return None, issues

    issues.extend(_check_ontology_terms(meta))
    return meta, issues


def check_store_var_conventions(
    store_path: Union[str, Path],
) -> Dict[str, List[ValidationIssue]]:
    """Check every table's ``var`` against the fixed column conventions.

    The spec fixes ``var`` column names (see ``MSI_VAR_RESERVED_COLUMNS``)
    and requires ``mz``: present, numeric, finite, strictly increasing.
    Only the ``var`` group is read -- never the intensity matrix -- so
    this stays cheap on stores of any size.

    Args:
        store_path: Path to a converted ``.zarr`` store.

    Returns:
        Mapping of table name to the issues found for it (empty list
        when the table conforms).

    Raises:
        ValueError: If the path is not a SpatialData store (no
            ``tables`` group).
    """
    import numpy as np
    import zarr

    # Same routing as read_msi_metadata_blocks: past the Windows path limit
    # zarr returns fill values instead of failing.
    root = zarr.open_group(str(prepare_zarr_read_path(Path(store_path))), mode="r")
    if "tables" not in root:
        raise ValueError(
            f"{store_path} does not look like a SpatialData store: "
            "it has no 'tables' group"
        )

    results: Dict[str, List[ValidationIssue]] = {}
    for name in sorted(root["tables"].keys()):
        issues: List[ValidationIssue] = []
        results[name] = issues
        try:
            var_group = root["tables"][name]["var"]
        except KeyError:
            issues.append(ValidationIssue("error", "var", "table has no var group"))
            continue
        if "mz" not in var_group:
            issues.append(
                ValidationIssue(
                    "error", "var.mz", "required var column 'mz' is missing"
                )
            )
            continue

        mz = np.asarray(var_group["mz"])
        if not np.issubdtype(mz.dtype, np.number):
            issues.append(
                ValidationIssue(
                    "error",
                    "var.mz",
                    f"'mz' must be numeric, found dtype {mz.dtype}",
                )
            )
            continue
        if mz.size and not bool(np.isfinite(mz).all()):
            issues.append(
                ValidationIssue("error", "var.mz", "'mz' contains non-finite values")
            )
            continue

        if MSI_VAR_MOBILITY_COLUMN in var_group:
            _check_mobility_var(var_group, mz, issues)
        elif MSI_VAR_PRECURSOR_COLUMN in var_group:
            _check_msms_var(var_group, mz, issues)
        elif mz.size > 1 and not bool((np.diff(mz) > 0).all()):
            issues.append(
                ValidationIssue("error", "var.mz", "'mz' is not strictly increasing")
            )
    return results


def _check_mobility_var(var_group: Any, mz: Any, issues: List[ValidationIssue]) -> None:
    """The contract of a mobility-resolved table's ``var``.

    Features are ``(mz, mobility)`` pairs: ``mz`` is non-decreasing (an m/z
    split by mobility repeats), the pairs are unique, and the rows are in
    lexicographic ``(mz, mobility)`` order so an m/z window is one
    contiguous column block.
    """
    mobility = _pair_column(var_group, MSI_VAR_MOBILITY_COLUMN, mz, issues)
    if mobility is None:
        return
    _check_sorted_pairs(mz, mobility, "mz", MSI_VAR_MOBILITY_COLUMN, "mobility", issues)


def _check_msms_var(var_group: Any, mz: Any, issues: List[ValidationIssue]) -> None:
    """The contract of a demultiplexed MS/MS table's ``var``.

    Features are ``(precursor_mz, mz)`` pairs: ``precursor_mz`` is
    non-decreasing (a precursor owns many fragments), ``mz`` increases
    strictly inside each precursor's block, and the rows are in
    lexicographic order so one precursor's fragment spectrum is one
    contiguous column block.  ``mz`` is *not* sorted over the whole table
    -- it restarts at every precursor -- which is exactly why the strict
    single-column contract must not be applied here.
    """
    precursor = _pair_column(var_group, MSI_VAR_PRECURSOR_COLUMN, mz, issues)
    if precursor is None:
        return
    _check_sorted_pairs(
        precursor,
        mz,
        MSI_VAR_PRECURSOR_COLUMN,
        "mz",
        "demultiplexed MS/MS",
        issues,
    )


def _pair_column(
    var_group: Any, name: str, mz: Any, issues: List[ValidationIssue]
) -> Optional[Any]:
    """The companion column of a paired feature axis, or ``None`` if unusable.

    Shared by both paired contracts: whatever the pair means, the second
    column has to be numeric, finite and as long as ``mz`` before the
    ordering of the pairs can be read at all.
    """
    import numpy as np

    values = np.asarray(var_group[name])
    location = f"var.{name}"
    if not np.issubdtype(values.dtype, np.number):
        issues.append(
            ValidationIssue(
                "error",
                location,
                f"'{name}' must be numeric, found dtype {values.dtype}",
            )
        )
        return None
    if values.size != mz.size:
        issues.append(
            ValidationIssue(
                "error",
                location,
                f"'{name}' has {values.size} values for {mz.size} m/z values",
            )
        )
        return None
    if values.size and not bool(np.isfinite(values).all()):
        issues.append(
            ValidationIssue("error", location, f"'{name}' contains non-finite values")
        )
        return None
    return values


def _check_sorted_pairs(
    first: Any,
    second: Any,
    first_name: str,
    second_name: str,
    table_kind: str,
    issues: List[ValidationIssue],
) -> None:
    """``(first, second)`` is lexicographically sorted and its pairs unique."""
    import numpy as np

    if first.size < 2:
        return
    if not bool((np.diff(first) >= 0).all()):
        issues.append(
            ValidationIssue(
                "error",
                f"var.{first_name}",
                f"'{first_name}' is not non-decreasing on a {table_kind} table",
            )
        )
        return
    tied = np.diff(first) == 0
    if bool((np.diff(second)[tied] <= 0).any()):
        issues.append(
            ValidationIssue(
                "error",
                f"var.{second_name}",
                f"({first_name}, {second_name}) pairs are not unique and "
                "sorted lexicographically",
            )
        )
