# API Reference

Thyra's Python API centres on two functions: `convert_msi`, which does the
work, and `preview_msi`, which tells you what an input is without converting
it. For most use cases those are all you need. The remaining sections document
configuration types, metadata objects, and base classes for advanced users who
want to inspect results or extend Thyra with new formats.

---

## Converting Data

The primary entry point. Detects the input format, reads metadata, and writes
a SpatialData/Zarr directory.

### Basic usage

```python
from thyra import convert_msi

# Minimal -- auto-detects format and pixel size
success = convert_msi("input.imzML", "output.zarr")

# With explicit parameters
success = convert_msi(
    "data/experiment.d",
    "output/experiment.zarr",
    dataset_id="hippocampus",
    pixel_size_um=10.0,
)
```

!!! warning "The Python API does not resample by default"
    `resampling_config` defaults to `None`, and nothing builds one for you, so
    the call above keeps the original mass axis. The `thyra` command-line tool
    is the opposite -- it resamples unless you pass `--no-resample`, because
    the default is applied in the CLI layer rather than in `convert_msi`. Pass
    a `resampling_config` explicitly to get the CLI's behaviour from Python.

### With resampling configuration

```python
success = convert_msi(
    "input.imzML",
    "output.zarr",
    resampling_config={
        "method": "nearest_neighbor",
        "axis_type": "orbitrap",
        "target_bins": 50000,
    },
)
```

### Multi-region dataset (select one region)

`region` takes either a `.mis` Area Name or a DB RegionNumber. A string is
matched against the area names first and only parsed as a number if no name
matches, so the two forms below are not interchangeable -- area `'03'` need not
be RegionNumber 3.

```python
# By the name flexImaging gives the area
success = convert_msi(
    "data/slide.d",
    "output/tissue_only.zarr",
    region="03",
)

# By the database's own region number, which starts at 0
success = convert_msi(
    "data/slide.d",
    "output/tissue_only.zarr",
    region=0,
)
```

Thyra logs the RegionNumber-to-Area-Name mapping at `INFO` when it opens a
multi-region dataset. See [`--region`](cli.md#conversion) for the detail.

### Large datasets

Nothing to switch on: every conversion makes two passes over the source and
writes each table from memory-mapped arrays, so the matrix is never held in
RAM. The `streaming` keyword older code passes is accepted and selects
nothing.

### Full signature

::: thyra.convert.convert_msi

---

## Previewing an Input

`preview_msi` answers "what is this file?" without converting it. It detects
the format, builds the reader, and reads metadata only -- no spectra are
decoded and no store is written -- so it is cheap enough to call while a user
waits. It is the entry point behind the Ousia import wizard's per-sample
preview card.

Two properties make it usable directly from UI code:

- **It never raises.** Any failure -- a path that does not exist, an
  unrecognised format, a truncated file -- comes back as an `MsiPreview` with
  `readable=False` and `error` set to the message. Check `readable` before
  reading the numeric fields; they hold zeroes when it is `False`.
- **It is fast.** The design budget is under 500 ms for inputs up to about
  50 GB, because it never touches the spectra.

```python
from pathlib import Path

from thyra import preview_msi

p = preview_msi(Path("example_data/synthetic_brain.imzML"))

if p.readable:
    print(p.grid_dims)        # (48, 36)
    print(p.n_pixels)         # 1728
    print(p.mz_range)         # (250.0, 1200.0)
    print(p.pixel_size_um)    # 25.0
    print(p.instrument_type)  # AxisType.CONSTANT
else:
    print("cannot read:", p.error)
```

Failure looks the same shape, which is the point:

```python
p = preview_msi(Path("no/such/file.imzML"))
print(p.readable, p.error)
# False Path does not exist: no\such\file.imzML
```

`instrument_type` is the `AxisType` the resampling decision tree would pick for
this input, so a caller can show the default before the user commits to it.
`has_escdat_folder` reports whether `<path>/EscDat/` exists, which is how a
downstream tool decides whether EscDat-derived registration is available.

::: thyra.preview.preview_msi
    options:
      show_root_heading: true
      heading_level: 3

::: thyra.preview.MsiPreview
    options:
      show_root_heading: true
      heading_level: 3

---

## Resampling Configuration

When you pass `resampling_config` to `convert_msi`, the dictionary keys map
to the fields of `ResamplingConfig`. You can pass a plain dict (as shown in
the examples above) or construct the dataclass directly:

```python
from thyra.resampling.types import ResamplingConfig, ResamplingMethod, AxisType

config = ResamplingConfig(
    method=ResamplingMethod.TIC_PRESERVING,
    axis_type=AxisType.ORBITRAP,
    target_bins=50000,
)

success = convert_msi("input.imzML", "output.zarr", resampling_config=config)
```

::: thyra.resampling.types.ResamplingConfig
    options:
      show_root_heading: true
      heading_level: 3

::: thyra.resampling.types.ResamplingMethod
    options:
      show_root_heading: true
      heading_level: 3
      members: true

::: thyra.resampling.types.AxisType
    options:
      show_root_heading: true
      heading_level: 3
      members: true

---

## Metadata Types

Readers expose metadata through two dataclasses. `EssentialMetadata` contains
everything needed for conversion decisions (grid size, mass range, memory
estimate). `ComprehensiveMetadata` wraps essential metadata and adds
vendor-specific details for provenance and QC.

```python
from thyra.readers.imzml import ImzMLReader

with ImzMLReader("sample.imzML") as reader:
    meta = reader.get_essential_metadata()
    print(f"Grid: {meta.dimensions}")
    print(f"m/z range: {meta.mass_range}")
    print(f"Spectra: {meta.n_spectra}")
    print(f"Est. memory: {meta.estimated_memory_gb:.1f} GB")
```

::: thyra.metadata.types.EssentialMetadata
    options:
      show_root_heading: true
      heading_level: 3

::: thyra.metadata.types.ComprehensiveMetadata
    options:
      show_root_heading: true
      heading_level: 3

---

## Metadata Schema

The versioned, ontology-mapped `uns["msi_metadata"]` block every store
carries. See [Metadata Schema](metadata-schema.md) for the storage
contract, the CLI (`thyra validate`, `thyra export-metaspace`), and the
versioning rules.

```python
from thyra.metadata.schema import (
    read_msi_metadata_blocks,
    to_metaspace,
    validate_document,
)

blocks = read_msi_metadata_blocks("output.zarr")
meta, issues = validate_document(blocks["msi_dataset_z0"])
submission, warnings = to_metaspace(meta)
```

::: thyra.metadata.schema.models.MSIMetadata
    options:
      show_root_heading: true
      heading_level: 3

::: thyra.metadata.schema.validate.validate_document
    options:
      show_root_heading: true
      heading_level: 3

::: thyra.metadata.schema.store_io.read_msi_metadata_blocks
    options:
      show_root_heading: true
      heading_level: 3

::: thyra.metadata.schema.metaspace.to_metaspace
    options:
      show_root_heading: true
      heading_level: 3

::: thyra.metadata.schema.builder.build_msi_metadata
    options:
      show_root_heading: true
      heading_level: 3

---

## Reader Base Class

All format readers (ImzML, Bruker, Waters, PHI) inherit from this base class.
If you are writing a custom reader for a new format, subclass `BaseMSIReader`
and implement the abstract methods below. See
[Supported Formats](supported-formats.md) for which optional methods are worth
implementing and what each one buys you.

::: thyra.core.base_reader.BaseMSIReader
    options:
      members:
        - get_essential_metadata
        - get_comprehensive_metadata
        - get_common_mass_axis
        - get_mass_axis_annotations
        - get_optical_image_paths
        - get_peak_counts_per_pixel
        - iter_spectra
        - get_region_map
        - get_region_info
        - has_shared_mass_axis
        - has_ion_mobility
        - get_mobility_axis
        - iter_mobility_spectra
        - get_fragmentation
        - iter_precursor_spectra
        - has_frame_scans
        - iter_frame_scans
        - close

A reader whose summed spectrum, mobility point cloud and precursor spectra
all derive from one raw read per pixel can hand that read over as a record,
so a conversion that writes several tables reads the source once per pass
(design decision D5). The Bruker TDF reader does; the record's contract is:

::: thyra.core.frames.FrameScans

---

## Converter Base Class

All output converters inherit from this base class. Currently only
SpatialData output is supported, but the architecture allows adding new output
formats by subclassing `BaseMSIConverter`.

::: thyra.core.base_converter.BaseMSIConverter
    options:
      members:
        - convert
        - pixel_size_um
        - pixel_size_source
        - dataset_id
        - handle_3d

---

## Format Detection and Plugin Registry

Thyra uses a registry to map file extensions and directory structures to the
correct reader and converter classes. The public functions below let you detect
formats programmatically or register your own reader/converter.

### Detecting a format

```python
from pathlib import Path
from thyra.core.registry import detect_format

fmt = detect_format(Path("experiment.imzML"))  # "imzml"
fmt = detect_format(Path("data.d"))            # "bruker" or "rapiflex"
fmt = detect_format(Path("data.raw"))          # "waters"
```

### Registering a custom reader

```python
from thyra.core.registry import register_reader
from thyra.core.base_reader import BaseMSIReader

@register_reader("my_format")
class MyFormatReader(BaseMSIReader):
    ...
```

::: thyra.core.registry.detect_format
    options:
      show_root_heading: true
      heading_level: 3

::: thyra.core.registry.register_reader
    options:
      show_root_heading: true
      heading_level: 3

::: thyra.core.registry.register_converter
    options:
      show_root_heading: true
      heading_level: 3
