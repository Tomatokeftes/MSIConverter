# thyra/metadata/schema/__init__.py
"""The versioned, ontology-mapped MSI metadata schema.

See :mod:`thyra.metadata.schema.models` for the schema itself,
``docs/metadata-schema.md`` for the storage contract, and
``thyra validate`` / ``thyra export-metaspace`` for the CLI.
"""

from .builder import build_msi_metadata
from .metaspace import to_metaspace
from .models import (
    MSI_METADATA_SCHEMA_VERSION,
    MSI_METADATA_UNS_KEY,
    MSI_VAR_PRECURSOR_COLUMN,
    MSI_VAR_PRECURSOR_INDEX_COLUMN,
    MSI_VAR_REQUIRED_COLUMNS,
    MSI_VAR_RESERVED_COLUMNS,
    Fragmentation,
    IonMobility,
    IsolationWindow,
    MobilityGrid,
    MSAnalysis,
    MSIMetadata,
    OntologyTerm,
    PixelSizeUm,
    ProcessingStep,
    Provenance,
    ResolvingPower,
    SampleInformation,
    SamplePreparation,
    SoftwareRef,
)
from .store_io import deep_merge, read_msi_metadata_blocks
from .validate import ValidationIssue, check_store_var_conventions, validate_document

__all__ = [
    "MSI_METADATA_SCHEMA_VERSION",
    "MSI_METADATA_UNS_KEY",
    "MSI_VAR_PRECURSOR_COLUMN",
    "MSI_VAR_PRECURSOR_INDEX_COLUMN",
    "MSI_VAR_REQUIRED_COLUMNS",
    "MSI_VAR_RESERVED_COLUMNS",
    "Fragmentation",
    "IonMobility",
    "IsolationWindow",
    "MSAnalysis",
    "MSIMetadata",
    "MobilityGrid",
    "OntologyTerm",
    "PixelSizeUm",
    "ProcessingStep",
    "Provenance",
    "ResolvingPower",
    "SampleInformation",
    "SamplePreparation",
    "SoftwareRef",
    "ValidationIssue",
    "build_msi_metadata",
    "check_store_var_conventions",
    "deep_merge",
    "read_msi_metadata_blocks",
    "to_metaspace",
    "validate_document",
]
