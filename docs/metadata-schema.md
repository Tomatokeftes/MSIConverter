# Metadata Schema

Every store Thyra writes carries a versioned, ontology-mapped metadata
block: `table.uns["msi_metadata"]`. Its base fields mirror the
[METASPACE](https://metaspace2020.org) submission form, so the metadata a
converted dataset carries is, by construction, what a METASPACE submission
needs -- filling it costs nothing extra.

The schema exists because MSI has had no structured metadata convention
the way other spatial omics modalities do. Vendor files spell the same
fact many ways, imzML metadata is free-form cvParams, and every pipeline
re-invents its own dictionary. This block fixes the names, maps them to
ontologies, and ships a validator.

!!! info "Where this sits in the format stack"
    Raw/archival exchange is the domain of
    [mzPeak](https://github.com/HUPO-PSI/mzPeak) (the HUPO-PSI working
    draft succeeding mzML); SpatialData is the analysis-ready layer and
    Thyra's output. Thyra is the bridge between the two, and this
    schema is the analysis-layer metadata contract: PSI CV-anchored
    like the raw layer, so nothing is lost crossing the bridge. Raw
    ragged spectra stay in the raw layer; they are deliberately not
    part of Thyra's output.

```python
import spatialdata as sd

sdata = sd.read_zarr("output.zarr")
block = sdata.tables["msi_dataset_z0"].uns["msi_metadata"]

print(block["schema_version"])                       # "0.5.0"
print(block["ms_analysis"]["pixel_size_um"])         # {"x": 20.0, "y": 20.0}
print(block["ms_analysis"]["ionisation_source"])     # "MALDI"
print(block["ms_analysis"]["ionisation_source_term"])
# {"accession": "MS:1000075",
#  "name": "matrix-assisted laser desorption ionization"}
```

---

## The document

Four sections. `ms_analysis` and `provenance` are written by the converter
for every store; `sample` and `preparation` describe things no raw file
records (what the tissue was, how it was prepared) and are supplied by
you -- see [Completing the metadata](#completing-the-metadata).

| Section | Field | Type | Ontology |
|---------|-------|------|----------|
| (root) | `schema_version` | `MAJOR.MINOR.PATCH` string, required | -- |
| `sample` | `organism`, `organism_term` | text + term | NCBITaxon |
| | `organism_part`, `organism_part_term` | text + term | UBERON |
| | `condition` | text | -- |
| | `sample_growth_conditions` | text | -- |
| `preparation` | `sample_stabilisation` | text | -- |
| | `tissue_modification` | text | -- |
| | `matrix`, `matrix_term` | text + term | CHEBI |
| | `matrix_application` | text | -- |
| | `solvent` | text | -- |
| `ms_analysis` | `polarity`, `polarity_term` | `"positive"`/`"negative"` + term | PSI-MS |
| | `ionisation_source`, `ionisation_source_term` | text + term | PSI-MS |
| | `analyzer`, `analyzer_term` | text + term | PSI-MS |
| | `instrument_model` | text | -- |
| | `detector_resolving_power` | `{value, at_mz}` | -- |
| | `pixel_size_um` | `{x, y}`, **required** | -- |
| | `ion_mobility` | `{present, separation, separation_term, unit_term, range_lower, range_upper, num_scans, resolved_table, grid}` | PSI-MS (`MS:1002815` / `MS:1002476`, unit `MS:1002814`) |
| | `fragmentation` | `{present, ms_level, constant_across_pixels, merges_precursors, dissociation_term, windows, resolved_table}` | PSI-MS (`MS:1000511`; windows `MS:1000827` / `828` / `829`, `MS:1000045`, `MS:1000133`) |
| `processing` | list of `{name, software {name, version, uri}, parameters}` | ordered steps, oldest first | -- |
| `provenance` | `thyra_version` | text, required | -- |
| | `source_format` | `"imzml"`, `"bruker"`, ... | -- |
| | `source_path` | text | -- |
| | `pixel_size_source` | `"automatic"` / `"manual"` / `"default"` | -- |

An ontology term is always the pair
`{"accession": "MS:1000075", "name": "matrix-assisted laser desorption ionization"}` --
a [CURIE](https://www.w3.org/TR/curie/) plus the term's label, so the block
is readable without resolving anything.

`pixel_size_um` is the one acquisition field that is required: conversion
refuses to run without a pixel size, so a document without it describes no
store Thyra ever wrote.

!!! note "Unknown fields are rejected"
    Validation refuses keys the schema does not define, so a typo fails
    loudly instead of becoming an unread field. Anything genuinely
    vendor-specific already has a home in the sections beside this block
    (`format_specific`, `acquisition_params`, `raw_metadata` -- see
    [Output Format](output-format.md#provenance)).

### What is auto-populated

Auto-population is deliberately honest: a field the source does not report
is left unset, never guessed. The only inferences are facts that follow
from the format itself.

| Source | polarity | ionisation source | analyzer | instrument model |
|--------|----------|-------------------|----------|------------------|
| imzML | -- | -- | from the `<analyzer>` component cvParam | from the instrumentConfiguration (model term or `MS:1000031` value) |
| Bruker `.d` | -- | MALDI, when the laser tables are present | TOF (timsTOF-family formats) | from the DB |

Bruker `.d` also fills `ion_mobility`: `present: true` for a TDF acquisition
(TIMS engaged), with the acquired 1/K0 range and the ramp length in scans,
and `present: false` for TSF. Other formats leave the field unset, which
means "not reported", not "no mobility". The MSI table is always summed over
the ramp; how it was summed is the `tdf_spectrum` parameter of the
`conversion` processing step (see [Supported Formats](supported-formats.md#bruker-timstof)).
`resolved_table` names the mobility-resolved sibling table when one was
written beside the summed table, and `grid` (`{law, lower, upper,
n_channels}`) describes the common mobility grid such a table was binned onto
when it was built from per-pixel mobility values; both are unset otherwise.
The arrays that describe the axis itself and the mass-mobility heatmap live
outside this block, in `uns["mobility_axis"]` and `uns["mobility_heatmap"]`
(see [Output Format](output-format.md#ion-mobility)): this block is versioned
and carries no arrays.
| PHI ToF-SIMS | from the header | SIMS | TOF | platform name |
| Waters `.raw` | -- | -- | -- | from `_HEADER.TXT` |

Bruker `.d` also fills `fragmentation` from `Frames.MsMsType` and whichever
precursor table the acquisition uses -- `PasefFrameMsMsInfo` for PASEF frames,
`FrameMsMsInfo` for single-precursor ones. `present: false` records a survey
acquisition; the field is left unset when the database cannot be asked at all,
which means "not reported", not "MS1". `windows` is stored as a JSON string
(a list of objects does not round-trip through AnnData/zarr) and decoded by
`read_msi_metadata_blocks` and `thyra validate`. The field names and terms are
mzPeak's, so an archive and a store describe a precursor the same way; see
[Output Format](output-format.md#fragmentation-msms) for the array block
beside it and for what `merges_precursors` means for the stored spectrum.
`resolved_table` names the demultiplexed sibling table when one was written,
mirroring `ion_mobility.resolved_table`, so both kinds of sibling are
discoverable from this block alone.

A precursor's m/z here and in `var["precursor_mz"]` is the **isolation window
target** (`MS:1000827`) -- the m/z the quadrupole was set to -- and not
`MS:1000744`, a selected ion whose m/z was measured. mzPeak keeps the two in
separate files for the same reason; do not read either as a monoisotopic mass.

Everything else -- organism, tissue, condition, matrix, resolving power --
cannot come from a raw file and stays empty until you provide it.

### Processing history

`processing` is the dataset's processing provenance, modeled on
[mzQC](https://github.com/HUPO-PSI/mzQC): an ordered list of steps, each
naming the software that performed it and the parameters it ran with.
The converter records its own steps -- `conversion` always, and
`mass axis resampling` with the resolved resampling parameters when
resampling was enabled. Downstream tools (normalisation, peak picking,
annotation) append theirs when they modify the store.

```python
[
  {"name": "conversion",
   "software": {"name": "thyra", "version": "3.5.0"}},
  {"name": "mass axis resampling",
   "software": {"name": "thyra", "version": "3.5.0"},
   "parameters": {"method": "nearest_neighbor", "target_bins": 50000,
                  "reference_mz": 1000.0}}
]
```

On disk the list is stored as a JSON string (AnnData/zarr cannot
round-trip a list of objects -- the same reason `uns["regions"]` is
JSON); `read_msi_metadata_blocks` and `validate_document` decode it
transparently.

---

## PSI CV alignment

Beyond the per-document `*_term` values, every schema **field** that
instantiates a PSI CV concept is bound to that concept's accession in
the JSON Schema itself (a `cv` annotation on the field definition), so
the claim "Thyra's output is annotated with the same PSI CV terms as
the raw file it came from" is machine-checkable from the committed
artifact alone:

| Field | CV concept |
|-------|-----------|
| `ms_analysis.pixel_size_um.x` | `IMS:1000046` pixel size (x) |
| `ms_analysis.pixel_size_um.y` | `IMS:1000047` pixel size y |
| `ms_analysis.polarity` | `MS:1000465` scan polarity |
| `ms_analysis.ionisation_source` | `MS:1000008` ionization type |
| `ms_analysis.analyzer` | `MS:1000443` mass analyzer type |
| `ms_analysis.instrument_model` | `MS:1000031` instrument model |
| `ms_analysis.detector_resolving_power` | `MS:1000800` mass resolving power |

On the input side, every imzML file-description cvParam is preserved in
`uns["raw_metadata"]["cvParams"]` **with its accession** (and unit
accession where the source set one) -- the name alone cannot be resolved
back to the CV concept. The list is stored as a JSON string (see
[Output Format](output-format.md#provenance) for why); `json.loads`
hands back the list of terms. Polarity declared there (`MS:1000130` /
`MS:1000129`) auto-populates the schema field.

### Candidate CV terms

Several imaging concepts this schema needs have no CV term yet. They
are tracked in `CANDIDATE_CV_CONCEPTS` as the vocabulary to raise in
the PSI/mzPeak imaging discussions, so this schema and the future
standard converge:

- pixel size semantics (raster pitch vs laser spot vs binned size)
- pixel size provenance (measured vs user-supplied vs default)
- coordinate origin and axis handedness
- stage offset of the raster origin
- ROI / acquisition region identity
- missing / empty pixel semantics
- continuous-vs-processed source provenance after conversion
- mass axis resampling provenance (method, axis law, target bins)

### LinkML rendering

The schema is also rendered as LinkML at
`thyra/metadata/schema/msi_metadata.linkml.yaml` -- classes, slots,
required flags, `slot_uri` CV bindings and the polarity enum with
`meaning:` accessions. The pydantic models remain the source of truth
for what Thyra writes and validates; the YAML is the discussion artifact
for LinkML-native settings (the PSI/mzPeak imaging work, the planned
spec repository), and a unit test keeps the two from drifting. A later
migration to LinkML-as-source changes no field names and nothing on
disk.

---

## var column conventions

The MSI table's `.var` column names are fixed by the spec so every
consumer can rely on one spelling:

| Column | Written by | Meaning |
|--------|-----------|---------|
| `mz` | every converter, **required** | The common mass axis. Numeric, finite, strictly increasing -- except on a sibling table, where the pair is what is unique and sorted: `(mz, mobility)` on a mobility-resolved one, `(precursor_mz, precursor_mobility, mz)` on a demultiplexed MS/MS one, where `precursor_index` identifies the block. |
| `mobility` | the converter, on mobility-resolved tables only | The feature's ion mobility (1/K0 or drift time; see `uns["mobility_axis"]`). Its presence is what marks the table as mobility-resolved. |
| `precursor_mz` | the converter, on demultiplexed MS/MS tables only | The isolated m/z the feature's fragments came from (see `uns["msms_schedule"]`). Its presence is what marks the table as demultiplexed; such a table never carries `mobility` as well. |
| `mz_index` | the converter, on sibling tables only | Column of the feature's m/z on the summed MSI table's axis |
| `mobility_index` | the converter, on mobility-resolved tables only | Rank of the feature's mobility among the table's distinct mobility values |
| `precursor_mobility` | the converter, on demultiplexed MS/MS tables only | The 1/K0 the precursor was isolated at (the middle of its mobility window). What tells two precursors sharing an m/z apart -- an isomer pair -- so they are never merged |
| `precursor_index` | the converter, on demultiplexed MS/MS tables only | The precursor's position in **this store's** precursor axis, and the identity of its column block. Means nothing outside the store: align two stores on `(precursor_mz, precursor_mobility)` |
| `formula` | annotation tools | Molecular formula of the annotation |
| `adduct` | annotation tools | Adduct, e.g. `+H`, `-H`, `+Na` |
| `annotation_source` | annotation tools | Tool/database that produced the annotation |
| `fdr` | annotation tools | False discovery rate of the annotation |

Readers may add extra per-channel columns (a non-m/z native axis such as
flight time stays alongside `mz`), and annotation tools may add columns
beyond these -- but the reserved names above must never be reused with a
different meaning. `thyra validate` checks the `mz` contract on every
table of a store.

---

## Storage contract

- The block lives at `table.uns["msi_metadata"]`, in every table the
  store has. This location is stable, like `uns["essential_metadata"]`
  and the `coordinate_systems` root attribute; consumers read it from
  here and nowhere else.
- It is written identically by every converter write path (the uns
  parity tests assert this), and contains no timestamps, so converting
  the same input twice produces the same block.
- Sections with nothing in them are omitted rather than written empty,
  following the store-wide convention.

## Versioning

`schema_version` follows semantic-version rules:

- Adding an optional field bumps the **minor** version.
- Renaming, removing, retyping, or making a field required bumps the
  **major** version.
- A validator implementing major version N rejects documents with a
  different major version, accepts older minors, and warns on newer
  minors.

Versions so far: 0.1.0 (initial), 0.2.0 (`ms_analysis.ion_mobility`
added), 0.3.0 (`ion_mobility.resolved_table` and `ion_mobility.grid` added),
0.4.0 (`ms_analysis.fragmentation` added), 0.5.0
(`fragmentation.resolved_table` added).

The JSON Schema rendering is committed at
`thyra/metadata/schema/msi_metadata_schema_v0_5.json` and ships in the
wheel, so non-Python consumers can validate documents without importing
Thyra:

```python
from importlib import resources
import json

schema = json.loads(
    resources.files("thyra.metadata.schema")
    .joinpath("msi_metadata_schema_v0_5.json")
    .read_text()
)
```

A unit test keeps the artifact in sync with the models; regenerate it with
`python -m thyra.metadata.schema.generate` after a model change.

---

## `thyra validate`

```
thyra validate PATH [--merge USER.json] [--json]
```

`PATH` is a converted `.zarr` store or a standalone metadata `.json`
document. Only the metadata block (and, for stores, the `var` axis) is
read -- validating a 100 GB store is instant. For stores it also checks
the [var column contract](#var-column-conventions) and that every table
carries a metadata block. Exit status is `0` when every document
conforms (warnings allowed) and `1` otherwise, so it can gate CI.

On Windows, a store whose files sit past the 260-character path limit is
read through an extended-length path automatically; without that, Zarr
would return empty terms for the keys it cannot open and the document
would fail validation for the wrong reason (see
[long output paths](getting-started.md#windows-long-output-paths)).

Errors mean the document does not conform: structural violations, unknown
PSI-MS/IMS/UO accessions, version incompatibility. Warnings mean it
conforms but says something suspicious: a term label that does not match
its accession, or a term from an unexpected ontology.

```bash
thyra validate output.zarr
# msi_dataset_z0: OK

thyra validate output.zarr --json   # machine-readable report on stdout
```

## `thyra export-metaspace`

```
thyra export-metaspace PATH [--merge USER.json] [--table NAME] [-o OUT.json]
```

Writes the METASPACE submission metadata JSON (default:
`<input>.metaspace.json` next to the input; `-o -` for stdout). Required
fields the store cannot know are emitted empty and reported as warnings on
stderr -- the output is a truthful starting point, never a fabricated
record. The one inference: matrix-free sources (DESI, SIMS) truthfully get
`MALDI_Matrix: "none"`.

```bash
thyra export-metaspace output.zarr
# warning: Sample_Information.Organism is required ... and is not set
# Wrote METASPACE metadata for msi_dataset_z0 to output.metaspace.json
```

## Completing the metadata

The fields only you can know go in a small JSON overlay, merged at
validation or export time with `--merge`:

```json
{
  "sample": {
    "organism": "Mus musculus",
    "organism_term": {"accession": "NCBITaxon:10090", "name": "Mus musculus"},
    "organism_part": "liver",
    "organism_part_term": {"accession": "UBERON:0002107", "name": "liver"},
    "condition": "wildtype"
  },
  "preparation": {
    "sample_stabilisation": "fresh frozen",
    "matrix": "2,5-dihydroxybenzoic acid (DHB)",
    "matrix_term": {"accession": "CHEBI:17189", "name": "2,5-dihydroxybenzoic acid"},
    "matrix_application": "ImagePrep"
  },
  "ms_analysis": {
    "detector_resolving_power": {"value": 130000, "at_mz": 400}
  }
}
```

```bash
thyra validate output.zarr --merge sample.json
thyra export-metaspace output.zarr --merge sample.json
```

The overlay merges key-by-key over the stored block, so it only needs the
fields you are adding. Keep it next to the store; it applies unchanged to
every dataset from the same study.

## Python API

```python
from thyra.metadata.schema import (
    MSIMetadata,             # the pydantic model
    read_msi_metadata_blocks,  # {table_name: block} from a store
    validate_document,       # (model | None, issues)
    to_metaspace,            # (submission_dict, warnings)
)

blocks = read_msi_metadata_blocks("output.zarr")
meta, issues = validate_document(blocks["msi_dataset_z0"])
submission, warnings = to_metaspace(meta)
```
