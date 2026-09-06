# thyra/metadata/schema/models.py
"""Pydantic models for the versioned MSI metadata schema.

The schema is the structured, ontology-mapped description of an MSI
dataset that a converted SpatialData store carries in
``table.uns["msi_metadata"]``.  Its base fields mirror the METASPACE
submission form (organism, organ, condition, sample preparation,
matrix, polarity, ionisation source, analyzer, resolving power, pixel
size) so that the metadata a store carries is sufficient for a
METASPACE submission without retyping -- see
:mod:`thyra.metadata.schema.metaspace`.

Ontology mapping: free-text fields are paired with an optional
:class:`OntologyTerm` (``*_term``) carrying a CURIE accession --
NCBITaxon for organism, UBERON for organism part, CHEBI for the MALDI
matrix, and PSI-MS for polarity, ionisation source and analyzer.  The
PSI-MS/IMS/UO accessions are resolvable against the tables shipped in
:mod:`thyra.metadata.ontology`.

Versioning: ``schema_version`` follows semantic-version rules.  A
reader implementing major version N must reject documents with a
different major version, accept documents with an older minor version,
and may warn on a newer minor version.  Purely additive optional
fields bump the minor version; anything else bumps the major version.
The JSON Schema rendering of these models is committed next to this
module and kept in sync by a unit test; regenerate it with
``python -m thyra.metadata.schema.generate``.
"""

import json
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The schema version this code implements and writes.
# 0.2.0: added the optional ``ms_analysis.ion_mobility`` block (additive).
# 0.3.0: ``ion_mobility`` gained ``resolved_table`` and ``grid`` (additive).
# 0.4.0: added the optional ``ms_analysis.fragmentation`` block (additive).
# 0.5.0: ``fragmentation`` gained ``resolved_table`` (additive), so the
#        demultiplexed MS/MS sibling is discoverable from this block the
#        same way ``ion_mobility.resolved_table`` names the mobility one.
MSI_METADATA_SCHEMA_VERSION = "0.5.0"

# Where the block lives inside a converted store:
# ``table.uns["msi_metadata"]``.  This location is a stable contract
# (like ``uns["essential_metadata"]`` and the ``coordinate_systems``
# root attribute); consumers read it from here and nowhere else.
MSI_METADATA_UNS_KEY = "msi_metadata"

# The committed JSON Schema artifact for this schema version.
SCHEMA_JSON_FILENAME = "msi_metadata_schema_v0_5.json"

# Fixed var column conventions for the MSI table.  ``mz`` is required
# and written by every converter; the remaining names are reserved for
# annotation results so downstream consumers can rely on one spelling
# (see docs/metadata-schema.md).  Nothing may reuse these names with a
# different meaning; ``thyra validate`` checks the ``mz`` contract.
MSI_VAR_REQUIRED_COLUMNS = ("mz",)
MSI_VAR_RESERVED_COLUMNS = (
    "mz",
    "mobility",
    "mz_index",
    "mobility_index",
    "precursor_mz",
    "precursor_index",
    "formula",
    "adduct",
    "annotation_source",
    "fdr",
)

# The column that marks a mobility-resolved table (pixels x (m/z, mobility)
# features). The MSI table never carries it; a table that does has a
# non-decreasing ``mz`` with duplicates, validated by the pair instead.
MSI_VAR_MOBILITY_COLUMN = "mobility"

# The column that marks a demultiplexed MS/MS table (pixels x (precursor,
# fragment) features). The MSI table never carries it; a table that does
# has one contiguous column block per precursor, each block a spectrum,
# validated by the pair instead.  A table carries this or ``mobility``,
# never both: the two say different things about what a column is.
#
# Its value is the **isolation window target** (PSI-MS ``MS:1000827``),
# the m/z the quadrupole was set to -- not ``MS:1000744``, a selected ion
# whose m/z was measured. mzPeak keeps the two apart for the same reason,
# and a consumer must not read this column as a monoisotopic mass.
MSI_VAR_PRECURSOR_COLUMN = "precursor_mz"

# Imaging concepts this schema needs that have no PSI CV term yet.
# These are the candidate terms to raise in the mzPeak / PSI-MS imaging
# discussions, so this schema's vocabulary and the future standard's
# converge.  Each entry: (concept, where it lives in Thyra's output).
CANDIDATE_CV_CONCEPTS = (
    (
        "pixel size semantics (raster pitch vs laser spot vs binned size)",
        "ms_analysis.pixel_size_um",
    ),
    (
        "pixel size provenance (measured vs user-supplied vs default)",
        "provenance.pixel_size_source",
    ),
    (
        "coordinate origin and axis handedness",
        "root attrs coordinate_systems.global",
    ),
    (
        "stage offset of the raster origin",
        "root attrs coordinate_systems.global.stage_offset_um",
    ),
    ("ROI / acquisition region identity", "uns['regions']"),
    ("missing / empty pixel semantics", "obs row filtering"),
    (
        "continuous-vs-processed source provenance after conversion",
        "uns['essential_metadata'].spectrum_type",
    ),
    (
        "mass axis resampling provenance (method, axis law, target bins)",
        "processing steps",
    ),
)


def _cv(accession: str, name: str) -> Dict[str, Any]:
    """``json_schema_extra`` payload binding a field to its PSI CV concept.

    The binding is emitted into the JSON Schema artifact, so the claim
    "this field is the CV concept MS:1000443" is machine-readable
    without importing Thyra.  Value-level terms (which analyzer, which
    polarity) are carried per document by the ``*_term`` fields; this
    binds the field itself to the concept it instantiates.
    """
    return {"cv": {"accession": accession, "name": name}}


_CURIE_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*:\S+$"
_SEMVER_PATTERN = r"^\d+\.\d+\.\d+$"


class _SchemaModel(BaseModel):
    """Base for all schema models.

    Unknown keys are rejected rather than silently carried, so a typo
    in a hand-authored document fails validation instead of being
    stored as an unread field.
    """

    model_config = ConfigDict(extra="forbid")


class OntologyTerm(_SchemaModel):
    """A controlled-vocabulary reference: CURIE accession plus label."""

    accession: str = Field(
        pattern=_CURIE_PATTERN,
        description=(
            "CURIE identifier, e.g. 'MS:1000075', 'NCBITaxon:10090', "
            "'UBERON:0002107' or 'CHEBI:90695'."
        ),
    )
    name: str = Field(
        min_length=1,
        description="The term's primary label in its ontology.",
    )


class PixelSizeUm(_SchemaModel):
    """In-plane raster pitch in micrometres."""

    x: float = Field(
        gt=0,
        description="Pixel size along x in micrometres.",
        json_schema_extra=_cv("IMS:1000046", "pixel size (x)"),
    )
    y: float = Field(
        gt=0,
        description="Pixel size along y in micrometres.",
        json_schema_extra=_cv("IMS:1000047", "pixel size y"),
    )


class ResolvingPower(_SchemaModel):
    """Mass resolving power quoted at a reference m/z."""

    value: float = Field(gt=0, description="Resolving power (FWHM definition).")
    at_mz: float = Field(gt=0, description="The m/z the value is quoted at.")


class SampleInformation(_SchemaModel):
    """What the sample is (METASPACE ``Sample_Information``)."""

    organism: Optional[str] = Field(
        default=None, description="Species, e.g. 'Mus musculus'."
    )
    organism_term: Optional[OntologyTerm] = Field(
        default=None, description="NCBITaxon term for the organism."
    )
    organism_part: Optional[str] = Field(
        default=None, description="Organ or organism part, e.g. 'liver'."
    )
    organism_part_term: Optional[OntologyTerm] = Field(
        default=None, description="UBERON term for the organism part."
    )
    condition: Optional[str] = Field(
        default=None, description="E.g. 'wildtype', 'diseased'."
    )
    sample_growth_conditions: Optional[str] = Field(
        default=None, description="E.g. intervention or treatment."
    )


class SamplePreparation(_SchemaModel):
    """How the sample was prepared (METASPACE ``Sample_Preparation``)."""

    sample_stabilisation: Optional[str] = Field(
        default=None, description="Preservation method, e.g. 'fresh frozen'."
    )
    tissue_modification: Optional[str] = Field(
        default=None, description="Chemical modification, e.g. 'derivatised'."
    )
    matrix: Optional[str] = Field(
        default=None,
        description=(
            "MALDI matrix, e.g. '2,5-dihydroxybenzoic acid (DHB)'. "
            "'none' for matrix-free techniques (DESI, SIMS)."
        ),
    )
    matrix_term: Optional[OntologyTerm] = Field(
        default=None, description="CHEBI term for the matrix compound."
    )
    matrix_application: Optional[str] = Field(
        default=None, description="Matrix application device or protocol."
    )
    solvent: Optional[str] = Field(default=None, description="Solvent used.")


class MobilityGrid(_SchemaModel):
    """The common mobility grid a mobility-resolved table was binned onto.

    Describes the channels of a grid table (pixels x (m/z, mobility)
    built from per-pixel mobility sources): the law the channel edges
    follow, the range they span and how many there are. Absent until
    such a table is written; the store's ``uns["mobility_heatmap"]`` uses
    the same channel count and edges so the two align by index.
    """

    law: str = Field(
        min_length=1,
        description=(
            "How channel edges are spaced across the range, e.g. 'linear' "
            "(equal width in the axis unit)."
        ),
    )
    lower: float = Field(description="Lower edge of the first channel.")
    upper: float = Field(description="Upper edge of the last channel.")
    n_channels: int = Field(ge=1, description="Number of mobility channels.")

    @model_validator(mode="after")
    def _range_has_extent(self) -> "MobilityGrid":
        if not self.upper > self.lower:
            raise ValueError("mobility grid upper must exceed lower")
        return self


class IonMobility(_SchemaModel):
    """Whether, and how, the acquisition separated ions by mobility.

    Written so a consumer can tell a spectrum that was *summed over* a
    mobility ramp (Bruker TDF, TIMS engaged) from one that never had a
    mobility dimension (TSF, imzML, most other sources).  The MSI table
    Thyra writes collapses the ramp; the fields here describe what was
    collapsed.  Mobility is a spectral coordinate, never a spatial one,
    so nothing about it appears in the store's coordinate systems.
    """

    present: bool = Field(
        description=(
            "True when the source carries a mobility dimension, e.g. a Bruker "
            "TDF acquisition with TIMS engaged."
        ),
    )
    separation: Optional[str] = Field(
        default=None,
        description=(
            "The mobility quantity the instrument records, e.g. 'inverse "
            "reduced ion mobility' (TIMS) or 'ion mobility drift time'."
        ),
        json_schema_extra=_cv("MS:1002892", "ion mobility attribute"),
    )
    separation_term: Optional[OntologyTerm] = Field(
        default=None,
        description=(
            "PSI-MS term for the quantity: MS:1002815 (inverse reduced ion "
            "mobility) or MS:1002476 (ion mobility drift time)."
        ),
    )
    unit_term: Optional[OntologyTerm] = Field(
        default=None,
        description=(
            "Unit of range_lower and range_upper, e.g. MS:1002814 "
            "volt-second per square centimeter for 1/K0."
        ),
    )
    range_lower: Optional[float] = Field(
        default=None, description="Lower bound of the acquired mobility range."
    )
    range_upper: Optional[float] = Field(
        default=None, description="Upper bound of the acquired mobility range."
    )
    num_scans: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Mobility scans per frame (the TIMS ramp length); the largest "
            "when frames differ."
        ),
    )
    resolved_table: Optional[str] = Field(
        default=None,
        min_length=1,
        description=(
            "Element key of the mobility-resolved sibling table (pixels x "
            "(m/z, mobility)) written beside this summed table, when one was."
        ),
    )
    grid: Optional[MobilityGrid] = Field(
        default=None,
        description=(
            "The common mobility grid the resolved table was binned onto, "
            "when it was built from per-pixel mobility values."
        ),
    )

    @model_validator(mode="after")
    def _absent_means_empty(self) -> "IonMobility":
        """A source without mobility cannot describe a mobility axis."""
        if not self.present and any(
            value is not None
            for value in (
                self.separation,
                self.separation_term,
                self.unit_term,
                self.range_lower,
                self.range_upper,
                self.num_scans,
                self.resolved_table,
                self.grid,
            )
        ):
            raise ValueError(
                "ion_mobility.present is False but mobility axis fields are set"
            )
        return self


class IsolationWindow(_SchemaModel):
    """One precursor isolation, named the way mzPeak names it.

    The window spans ``[target - lower_offset, target + upper_offset]``.
    Offsets rather than a single width is mzPeak's shape (and mzML's);
    a source reporting one full width has it halved into two equal
    offsets, which is the reading every mzML writer for those
    instruments takes.
    """

    target: float = Field(
        gt=0,
        description="Isolation window target m/z.",
        json_schema_extra=_cv("MS:1000827", "isolation window target m/z"),
    )
    lower_offset: Optional[float] = Field(
        default=None,
        ge=0,
        description="How far below the target the window reaches, in m/z.",
        json_schema_extra=_cv("MS:1000828", "isolation window lower offset"),
    )
    upper_offset: Optional[float] = Field(
        default=None,
        ge=0,
        description="How far above the target the window reaches, in m/z.",
        json_schema_extra=_cv("MS:1000829", "isolation window upper offset"),
    )
    collision_energy: Optional[float] = Field(
        default=None,
        description="Collision energy in electronvolts (UO:0000266).",
        json_schema_extra=_cv("MS:1000045", "collision energy"),
    )
    scan_begin: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "First mobility scan this window occupies, when the source "
            "separates its windows along the mobility ramp (Bruker PASEF)."
        ),
    )
    scan_end: Optional[int] = Field(
        default=None,
        ge=0,
        description="One past the last mobility scan this window occupies.",
    )

    @model_validator(mode="after")
    def _scan_range_is_ordered(self) -> "IsolationWindow":
        if self.scan_begin is not None and self.scan_end is not None:
            if self.scan_end <= self.scan_begin:
                raise ValueError("scan_end must exceed scan_begin")
        return self


class Fragmentation(_SchemaModel):
    """Whether, and how, the acquisition fragmented its ions.

    An MS/MS imaging run measures fragments, so the m/z axis of the table
    means fragment m/z. Nothing about the axis says so, which is why this
    block exists: without it an MS/MS store is indistinguishable from an
    MS1 one.

    The MSI table sums a frame that isolated several precursors into one
    spectrum. ``merges_precursors`` says when that has happened, so a
    consumer knows the spectrum is a chimera rather than discovering it
    from the peaks, and ``resolved_table`` names the sibling table that
    holds those precursors split apart when one was written.
    """

    present: bool = Field(
        description="True when the stored spectra are fragment spectra.",
    )
    ms_level: int = Field(
        ge=1,
        description="MS level of the stored spectra: 1 for a survey scan.",
        json_schema_extra=_cv("MS:1000511", "ms level"),
    )
    constant_across_pixels: bool = Field(
        default=True,
        description=(
            "Whether every pixel was fragmented on the same schedule. A "
            "scheduled method makes it so; a data-dependent one does not, "
            "and then the windows below are not a global precursor axis."
        ),
    )
    merges_precursors: bool = Field(
        default=False,
        description=(
            "Whether one stored spectrum sums fragments of more than one "
            "precursor, so its peaks cannot be attributed to a single one."
        ),
    )
    dissociation_term: Optional[OntologyTerm] = Field(
        default=None,
        description="PSI-MS dissociation method, e.g. MS:1000133 (CID).",
    )
    windows: List[IsolationWindow] = Field(
        default_factory=list,
        description="The precursor schedule, empty when none was reported.",
    )
    resolved_table: Optional[str] = Field(
        default=None,
        min_length=1,
        description=(
            "Element key of the demultiplexed sibling table (pixels x "
            "(precursor, fragment)) written beside this summed table, when "
            "one was."
        ),
    )

    @model_validator(mode="after")
    def _absent_means_ms1(self) -> "Fragmentation":
        """A run that fragmented nothing cannot describe a precursor."""
        if not self.present:
            if self.ms_level != 1:
                raise ValueError("fragmentation.present is False but ms_level > 1")
            if self.windows or self.dissociation_term is not None:
                raise ValueError(
                    "fragmentation.present is False but precursor fields are set"
                )
        elif self.ms_level < 2:
            raise ValueError("fragmentation.present is True but ms_level < 2")
        if self.merges_precursors and len(self.windows) < 2:
            raise ValueError("merges_precursors needs more than one isolation window")
        return self


class MSAnalysis(_SchemaModel):
    """How the data was acquired (METASPACE ``MS_Analysis``).

    ``pixel_size_um`` is the one field that is always known at
    conversion time (conversion refuses to run without a pixel size),
    so it is the one required field of the section.
    """

    polarity: Optional[Literal["positive", "negative"]] = Field(
        default=None,
        description="Ion polarity mode.",
        json_schema_extra=_cv("MS:1000465", "scan polarity"),
    )
    polarity_term: Optional[OntologyTerm] = Field(
        default=None,
        description="PSI-MS scan polarity term (MS:1000130 / MS:1000129).",
    )
    ionisation_source: Optional[str] = Field(
        default=None,
        description="E.g. 'MALDI', 'DESI', 'SIMS'.",
        json_schema_extra=_cv("MS:1000008", "ionization type"),
    )
    ionisation_source_term: Optional[OntologyTerm] = Field(
        default=None, description="PSI-MS ionisation type term."
    )
    analyzer: Optional[str] = Field(
        default=None,
        description="E.g. 'TOF', 'Orbitrap', 'FTICR'.",
        json_schema_extra=_cv("MS:1000443", "mass analyzer type"),
    )
    analyzer_term: Optional[OntologyTerm] = Field(
        default=None, description="PSI-MS mass analyzer type term."
    )
    instrument_model: Optional[str] = Field(
        default=None,
        description="Instrument model as reported by the source.",
        json_schema_extra=_cv("MS:1000031", "instrument model"),
    )
    detector_resolving_power: Optional[ResolvingPower] = Field(
        default=None,
        description="Resolving power at a reference m/z.",
        json_schema_extra=_cv("MS:1000800", "mass resolving power"),
    )
    pixel_size_um: PixelSizeUm = Field(
        description="In-plane raster pitch in micrometres."
    )
    ion_mobility: Optional[IonMobility] = Field(
        default=None,
        description=(
            "Whether the source separated ions by mobility and, if so, over "
            "what range; the MSI table is summed over that dimension."
        ),
    )
    fragmentation: Optional[Fragmentation] = Field(
        default=None,
        description=(
            "Whether the stored spectra are fragment spectra and, if so, of "
            "which precursors; unset when the source does not say."
        ),
    )

    @model_validator(mode="after")
    def _polarity_and_term_agree(self) -> "MSAnalysis":
        """A polarity string and term that contradict are worse than either alone."""
        if self.polarity is not None and self.polarity_term is not None:
            expected = {
                "positive": "MS:1000130",
                "negative": "MS:1000129",
            }[self.polarity]
            if self.polarity_term.accession != expected:
                raise ValueError(
                    f"polarity '{self.polarity}' contradicts polarity_term "
                    f"{self.polarity_term.accession} (expected {expected})"
                )
        return self


class SoftwareRef(_SchemaModel):
    """A software agent, the way mzQC records analysis software."""

    name: str = Field(min_length=1, description="Software name, e.g. 'thyra'.")
    version: str = Field(min_length=1, description="Software version.")
    uri: Optional[str] = Field(default=None, description="Homepage or repository URL.")


class ProcessingStep(_SchemaModel):
    """One processing action between the raw data and this store.

    Modeled on mzQC provenance: an ordered list of steps, each naming
    the software that performed it and the parameters it ran with.  The
    converter records its own steps (conversion, mass axis resampling);
    downstream tools append theirs (normalisation, peak picking,
    annotation) when they modify the store.
    """

    name: str = Field(
        min_length=1,
        description=(
            "What was done, e.g. 'conversion', 'mass axis resampling', "
            "'normalisation', 'peak picking', 'annotation'."
        ),
    )
    software: SoftwareRef = Field(description="The software that did it.")
    parameters: Dict[str, Union[str, int, float, bool]] = Field(
        default_factory=dict,
        description="The parameters the step ran with.",
    )


class Provenance(_SchemaModel):
    """Who wrote the block and from what.

    Deliberately small: the store's ``uns["essential_metadata"]`` and
    root attributes already record dimensions, mass range, spectrum
    type and conversion timestamp.  This section records only what is
    needed to interpret the *metadata block itself*, and contains no
    timestamps so that two conversions of the same input produce an
    identical block (the uns parity tests rely on this).
    """

    thyra_version: str = Field(
        min_length=1, description="Thyra version that wrote the block."
    )
    source_format: Optional[str] = Field(
        default=None,
        description="Detected input format, e.g. 'imzml', 'bruker', 'phi'.",
    )
    source_path: Optional[str] = Field(
        default=None, description="Path of the source data at conversion time."
    )
    pixel_size_source: Optional[Literal["default", "manual", "automatic"]] = Field(
        default=None,
        description=(
            "How the pixel size was determined: read from the source "
            "metadata ('automatic'), supplied by the user ('manual'), or "
            "the 1.0 um fallback ('default')."
        ),
    )


class MSIMetadata(_SchemaModel):
    """The versioned MSI metadata document.

    ``sample`` and ``preparation`` cannot be auto-populated from raw
    files and default to empty; ``ms_analysis`` and ``provenance`` are
    written by the converter for every store.
    """

    schema_version: str = Field(
        default=MSI_METADATA_SCHEMA_VERSION,
        pattern=_SEMVER_PATTERN,
        description="Semantic version of the schema this document conforms to.",
    )
    sample: SampleInformation = Field(
        default_factory=SampleInformation,
        description="What the sample is; user-supplied.",
    )
    preparation: SamplePreparation = Field(
        default_factory=SamplePreparation,
        description="How the sample was prepared; user-supplied.",
    )
    ms_analysis: MSAnalysis = Field(
        description="How the data was acquired; auto-populated where possible."
    )
    processing: List[ProcessingStep] = Field(
        default_factory=list,
        description="Ordered processing history, oldest first (mzQC-style).",
    )
    provenance: Provenance = Field(
        description="Who wrote this block and from what source."
    )

    def to_uns_dict(self) -> Dict[str, Any]:
        """Serialise for storage in ``table.uns``.

        ``None`` fields are dropped, and the ``sample`` / ``preparation``
        / ``processing`` sections are omitted entirely when empty --
        following the store convention that a section the source has
        nothing for is omitted rather than written empty, so consumers
        can tell "not available" from "available and empty".

        ``processing`` is stored as a JSON string: it is a list of
        objects, which AnnData/zarr cannot round-trip (the same reason
        ``uns["regions"]`` is JSON).  ``read_msi_metadata_blocks`` and
        ``validate_document`` both decode it transparently.
        ``ms_analysis.fragmentation.windows`` is a list of objects too and
        gets the same treatment: stored raw it comes back as a numpy array
        of Python ``repr`` strings, which is neither parseable nor safe to
        deepcopy on numpy 2.1-2.2.
        """
        data: Dict[str, Any] = self.model_dump(mode="json", exclude_none=True)
        for section in ("sample", "preparation", "processing"):
            if not data.get(section):
                data.pop(section, None)
        if "processing" in data:
            data["processing"] = json.dumps(data["processing"])
        fragmentation = data.get("ms_analysis", {}).get("fragmentation")
        if isinstance(fragmentation, dict) and "windows" in fragmentation:
            fragmentation["windows"] = json.dumps(fragmentation["windows"])
        return data


def field_cv_bindings() -> Dict[str, Dict[str, str]]:
    """Every field-to-CV binding, keyed by its path in the JSON Schema.

    Collected from the emitted JSON Schema rather than the models, so
    the result is exactly what an external consumer of the committed
    artifact sees.
    """
    bindings: Dict[str, Dict[str, str]] = {}

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            cv = node.get("cv")
            if isinstance(cv, dict):
                bindings[path] = cv
            for key, child in node.items():
                if key != "cv":
                    _walk(child, f"{path}.{key}" if path else key)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                _walk(child, f"{path}[{index}]")

    _walk(MSIMetadata.model_json_schema(), "")
    return bindings
