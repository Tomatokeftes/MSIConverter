# thyra/metadata/schema/builder.py
"""Build the ``msi_metadata`` block from extracted metadata.

Auto-population is best-effort and honest: a field the source does not
report is left unset rather than guessed.  The only inferences made are
facts that follow from the format itself (a PHI raw file is a TOF-SIMS
acquisition) or from vendor metadata that directly encodes the fact
(a Bruker dataset with a ``MaldiFrameLaserInfo`` table is MALDI).
"""

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

from ..types import ComprehensiveMetadata
from .models import (
    Fragmentation,
    IonMobility,
    IsolationWindow,
    MobilityGrid,
    MSAnalysis,
    MSIMetadata,
    PixelSizeUm,
    ProcessingStep,
    Provenance,
)
from .vocab import (
    normalize_analyzer,
    normalize_ionisation_source,
    normalize_polarity,
    term_from_accession,
)

logger = logging.getLogger(__name__)

# Facts that follow from the source format itself.  Kept deliberately
# conservative: only entries where every dataset of that format shares
# the value.  PHI raw files come from TOF-SIMS instruments; the Bruker
# formats Thyra reads (.d with analysis.tsf/.tdf) are all
# timsTOF-family, i.e. TOF analyzers.
_FORMAT_DEFAULTS: Dict[str, Dict[str, str]] = {
    "phi": {"ionisation_source": "SIMS", "analyzer": "TOF"},
    "bruker": {"analyzer": "TOF"},
    "tsf": {"analyzer": "TOF"},
    "tdf": {"analyzer": "TOF"},
}

# Key spellings the extractors use for the instrument model, in
# preference order (imzML: instrument_model; Bruker: instrument_name /
# model; PHI: platform).
_INSTRUMENT_MODEL_KEYS = (
    "instrument_model",
    "instrument_name",
    "model",
    "platform",
)


def _first_string(mapping: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[str]:
    """The first non-empty string value among ``keys``, if any."""
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _polarity_from_cv_params(raw_metadata: Dict[str, Any]) -> Optional[str]:
    """Polarity declared by the raw file's own cvParams, if unambiguous.

    imzML declares polarity as MS:1000130 (positive scan) / MS:1000129
    (negative scan) in the file description; the extractor preserves
    those with their accessions.  A file declaring both (alternating
    polarity) has no single truthful value and returns ``None``.
    """
    cv_params = raw_metadata.get("cvParams")
    if not isinstance(cv_params, list):
        return None
    accessions = {
        entry.get("accession") for entry in cv_params if isinstance(entry, dict)
    }
    positive = "MS:1000130" in accessions
    negative = "MS:1000129" in accessions
    if positive == negative:
        return None
    return "positive" if positive else "negative"


def _build_instrument_fields(
    acquisition: Dict[str, Any],
    instrument: Dict[str, Any],
    format_specific: Dict[str, Any],
    source_format: Optional[str],
) -> Dict[str, Any]:
    """What instrument produced the data: source, analyzer and model.

    Grouped because all three resolve the same way -- what the extractor
    reported, then what the format itself implies (a PHI raw file is a
    TOF-SIMS acquisition), and nothing when neither says. The per-format
    defaults are consulted by all of them, which is what makes this one
    step rather than three.
    """
    fields: Dict[str, Any] = {}
    fmt_defaults = _FORMAT_DEFAULTS.get((source_format or "").lower(), {})

    source = normalize_ionisation_source(
        _first_string(acquisition, ("ionisation_source", "ion_source", "technique"))
    )
    if source is None and format_specific.get("is_maldi"):
        source = normalize_ionisation_source("maldi")
    if source is None and "ionisation_source" in fmt_defaults:
        source = normalize_ionisation_source(fmt_defaults["ionisation_source"])
    if source is not None:
        fields["ionisation_source"], fields["ionisation_source_term"] = source

    analyzer = normalize_analyzer(
        _first_string(instrument, ("analyzer", "mass_analyzer"))
        or _first_string(acquisition, ("analyzer", "mass_analyzer"))
    )
    if analyzer is None and "analyzer" in fmt_defaults:
        analyzer = normalize_analyzer(fmt_defaults["analyzer"])
    if analyzer is not None:
        fields["analyzer"], fields["analyzer_term"] = analyzer

    instrument_model = _first_string(instrument, _INSTRUMENT_MODEL_KEYS)
    if instrument_model is not None:
        fields["instrument_model"] = instrument_model

    return fields


def _build_ms_analysis(
    acquisition: Dict[str, Any],
    instrument: Dict[str, Any],
    format_specific: Dict[str, Any],
    raw_metadata: Dict[str, Any],
    pixel_size_um: Tuple[float, float],
    source_format: Optional[str],
    mobility_resolved_table: Optional[str] = None,
    mobility_grid: Optional[Dict[str, Any]] = None,
    fragmentation: Any = None,
    msms_resolved_table: Optional[str] = None,
) -> MSAnalysis:
    """Assemble the acquisition section from what the extractors report."""
    fields: Dict[str, Any] = {}

    polarity = normalize_polarity(
        acquisition.get("polarity") or _polarity_from_cv_params(raw_metadata)
    )
    if polarity is not None:
        fields["polarity"], fields["polarity_term"] = polarity

    fields.update(
        _build_instrument_fields(
            acquisition, instrument, format_specific, source_format
        )
    )

    ion_mobility = _build_ion_mobility(
        format_specific.get("ion_mobility"), mobility_resolved_table, mobility_grid
    )
    if ion_mobility is not None:
        fields["ion_mobility"] = ion_mobility

    fragmentation_block = _build_fragmentation(fragmentation, msms_resolved_table)
    if fragmentation_block is not None:
        fields["fragmentation"] = fragmentation_block

    return MSAnalysis(
        pixel_size_um=PixelSizeUm(x=pixel_size_um[0], y=pixel_size_um[1]),
        **fields,
    )


def _optional_term(accession: Any) -> Optional[Any]:
    """The ontology term for an accession the extractor reported, if resolvable."""
    if not isinstance(accession, str) or not accession:
        return None
    try:
        return term_from_accession(accession)
    except KeyError:
        logger.debug("Mobility accession %s is not in the local ontology", accession)
        return None


def _build_ion_mobility(
    reported: Any,
    resolved_table: Optional[str] = None,
    grid: Optional[Dict[str, Any]] = None,
) -> Optional[IonMobility]:
    """The mobility block from what a reader's extractor reported.

    The Bruker and imzML extractors report one (``present`` True for TDF
    and for an imzML with a mobility array, False for TSF); readers that
    say nothing leave the field unset, which is honest -- "not reported"
    is not the same as "no mobility". A resolved table written beside
    the summed one is named here whatever the extractor said, since its
    existence proves the dimension.

    ``grid`` is present only when that table was *binned* onto a common
    mobility grid rather than read off a shared feature axis. It is the
    one thing in the store that says which of the two mechanisms filled
    the table, and it is a description: the table itself is the same
    shape either way.
    """
    if not isinstance(reported, dict) or "present" not in reported:
        if resolved_table:
            return IonMobility(
                present=True,
                resolved_table=resolved_table,
                grid=_mobility_grid(grid),
            )
        return None
    present = bool(reported["present"])
    if not present and not resolved_table:
        return IonMobility(present=False)

    fields: Dict[str, Any] = {"present": True}
    if resolved_table:
        fields["resolved_table"] = resolved_table
    grid_block = _mobility_grid(grid)
    if grid_block is not None:
        fields["grid"] = grid_block
    fields.update(_mobility_axis_fields(reported))
    return IonMobility(**fields)


def _mobility_grid(reported: Any) -> Optional[MobilityGrid]:
    """The grid block from what the converter resolved, or ``None``.

    Checked for shape rather than trusted, like everything else here: a
    grid that will not validate is dropped, since an invented one would
    let a consumer map a heatmap box onto channels that do not exist.
    """
    if not isinstance(reported, dict):
        return None
    try:
        return MobilityGrid(
            law=str(reported["law"]),
            lower=float(reported["lower"]),
            upper=float(reported["upper"]),
            n_channels=int(reported["n_channels"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        logger.debug("Mobility grid block is not usable and was dropped: %s", e)
        return None


def _mobility_axis_fields(reported: Dict[str, Any]) -> Dict[str, Any]:
    """The axis description an extractor reported, in the block's field names.

    Everything is optional and checked for shape rather than trusted: an
    unresolvable accession is dropped, never invented, and a malformed
    range or scan count is ignored.
    """
    fields: Dict[str, Any] = {}
    separation = reported.get("separation")
    if isinstance(separation, str) and separation.strip():
        fields["separation"] = separation.strip()
    for field, source in (
        ("separation_term", "separation_accession"),
        ("unit_term", "unit_accession"),
    ):
        term = _optional_term(reported.get(source))
        if term is not None:
            fields[field] = term
    mobility_range = reported.get("one_over_k0_range") or reported.get("range")
    if isinstance(mobility_range, (list, tuple)) and len(mobility_range) == 2:
        try:
            fields["range_lower"] = float(mobility_range[0])
            fields["range_upper"] = float(mobility_range[1])
        except (TypeError, ValueError):
            pass
    num_scans = reported.get("num_scans_max", reported.get("num_scans"))
    if isinstance(num_scans, (int, float)) and num_scans >= 1:
        fields["num_scans"] = int(num_scans)
    return fields


def _build_fragmentation(
    reported: Any, resolved_table: Optional[str] = None
) -> Optional[Fragmentation]:
    """The fragmentation block from what a reader reported.

    ``None`` in, ``None`` out: a reader that cannot tell says nothing,
    and an unset block is honest where ``present=False`` would be a
    claim. Everything is checked for shape rather than trusted -- a
    window without a usable target m/z is dropped rather than invented,
    since a precursor list is exactly the thing a consumer would act on.

    A demultiplexed table written beside the summed one is named here,
    the way :func:`_build_ion_mobility` names the mobility sibling, so
    both kinds are discoverable from this versioned block alone.
    """
    if not isinstance(reported, dict) or "ms_level" not in reported:
        return None
    try:
        ms_level = int(reported["ms_level"])
    except (TypeError, ValueError):
        return None
    if ms_level < 1:
        return None

    present = bool(reported.get("present", ms_level > 1))
    if not present:
        return Fragmentation(present=False, ms_level=1)

    windows = [
        window
        for window in (
            _build_isolation_window(entry) for entry in reported.get("windows") or []
        )
        if window is not None
    ]
    term = _optional_term(reported.get("dissociation_accession"))
    return Fragmentation(
        present=True,
        ms_level=max(ms_level, 2),
        constant_across_pixels=bool(reported.get("constant_across_pixels", True)),
        merges_precursors=len(windows) > 1,
        dissociation_term=term if windows else None,
        windows=windows,
        resolved_table=resolved_table,
    )


def _build_isolation_window(entry: Any) -> Optional[IsolationWindow]:
    """One isolation window, or ``None`` when it carries no usable target."""
    if not isinstance(entry, dict):
        return None
    try:
        target = float(entry["isolation_window_target"])
    except (KeyError, TypeError, ValueError):
        return None
    if not target > 0:
        return None

    fields: Dict[str, Any] = {"target": target}
    for field, key in (
        ("lower_offset", "isolation_window_lower_offset"),
        ("upper_offset", "isolation_window_upper_offset"),
        ("collision_energy", "collision_energy"),
    ):
        value = entry.get(key)
        if isinstance(value, (int, float)):
            fields[field] = float(value)
    begin, end = entry.get("scan_begin"), entry.get("scan_end")
    if isinstance(begin, int) and isinstance(end, int) and end > begin >= 0:
        fields["scan_begin"] = begin
        fields["scan_end"] = end
    return IsolationWindow(**fields)


def build_msi_metadata(
    comprehensive: Optional[ComprehensiveMetadata],
    *,
    pixel_size_um: Tuple[float, float],
    pixel_size_source: Optional[str] = None,
    source_format: Optional[str] = None,
    processing: Optional[List[ProcessingStep]] = None,
    mobility_resolved_table: Optional[str] = None,
    mobility_grid: Optional[Dict[str, Any]] = None,
    fragmentation: Any = None,
    msms_resolved_table: Optional[str] = None,
) -> MSIMetadata:
    """Build an :class:`MSIMetadata` document from extracted metadata.

    Args:
        comprehensive: The reader's comprehensive metadata, or ``None``
            when unavailable -- the document is still built, carrying
            the pixel size and provenance.
        pixel_size_um: Resolved in-plane pixel pitch ``(x_um, y_um)``.
            Required: conversion refuses to run without one, so a block
            without it describes no store Thyra ever wrote.
        pixel_size_source: How the pixel size was determined
            (``"automatic"`` / ``"manual"`` / ``"default"``).
        source_format: Detected input format name (``"imzml"``,
            ``"bruker"``, ...), when known.
        processing: Ordered processing steps performed so far, oldest
            first (see :class:`ProcessingStep`).
        mobility_resolved_table: Element key of the mobility-resolved
            sibling table written beside the summed table, when one was.
        mobility_grid: The common mobility grid that table was binned
            onto, as
            :meth:`thyra.resampling.mobility_grid.MobilityGrid.to_schema_report`
            renders it. ``None`` for a table read off a shared feature
            axis, which was binned onto nothing.
        fragmentation: What the reader reported about fragmentation, as
            :meth:`thyra.core.msms.FragmentationSchedule.to_extractor_report`
            renders it. ``None`` means the reader did not say, which is
            not the same as "MS1" and leaves the block unset.
        msms_resolved_table: Element key of the demultiplexed MS/MS
            sibling table written beside the summed table, when one was.

    Returns:
        The populated document.  Fields the source does not report are
        left unset.
    """
    from thyra import __version__

    acquisition: Dict[str, Any] = {}
    instrument: Dict[str, Any] = {}
    format_specific: Dict[str, Any] = {}
    raw_metadata: Dict[str, Any] = {}
    source_path: Optional[str] = None
    if comprehensive is not None:
        acquisition = dict(comprehensive.acquisition_params or {})
        instrument = dict(comprehensive.instrument_info or {})
        format_specific = dict(comprehensive.format_specific or {})
        raw_metadata = dict(comprehensive.raw_metadata or {})
        essential = comprehensive.essential
        if essential is not None:
            source_path = str(essential.source_path)

    return MSIMetadata(
        ms_analysis=_build_ms_analysis(
            acquisition,
            instrument,
            format_specific,
            raw_metadata,
            pixel_size_um,
            source_format,
            mobility_resolved_table,
            mobility_grid,
            fragmentation,
            msms_resolved_table,
        ),
        processing=list(processing or []),
        provenance=Provenance(
            thyra_version=__version__,
            source_format=source_format,
            source_path=source_path,
            pixel_size_source=cast(
                Optional[Literal["default", "manual", "automatic"]],
                pixel_size_source,
            ),
        ),
    )
